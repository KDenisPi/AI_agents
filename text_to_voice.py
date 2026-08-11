"""
Speech synthesis via an Orpheus TTS model on Ollama plus the SNAC vocoder.

Orpheus is a Llama-architecture model fine-tuned to emit audio codec
tokens rather than prose: its reply is a run of <custom_token_N> markers
encoding SNAC codebook entries. TextToVoice asks the model for those
tokens, decodes them back into SNAC's three codebooks, and runs them
through the vocoder to produce a 24 kHz mono .wav.

Two things keep this out of OllamaClient:
  - the request goes to /api/generate with raw=True. The model on the
    server carries the stock Llama 3.1 chat template, so /api/chat would
    wrap the text in <|start_header_id|> scaffolding and the model would
    never see the prompt format it was trained on - it returns nothing
    usable. raw=True bypasses templating so we can supply that format.
  - there is no conversation. Every call stands alone, so there is no
    history to keep, prune, or persist.

The SNAC weights (~80 MB) are fetched from HuggingFace on first use and
then held in memory, so the first call is much slower than the rest.
"""

import logging
import re
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import requests
import torch
from snac import SNAC

# oneDNN's AArch64 JIT backend (also named Xbyak) has a bug that mis-encodes
# an immediate operand for some conv1d shapes on this CPU, crashing with
# "bad err=15 ... illegal immediate parameter (range error)" - seen on Grace
# CPUs. The vocoder is small, so the non-JIT fallback conv costs little.
torch.backends.mkldnn.enabled = False

logger = logging.getLogger("text-to-voice")

DEFAULT_OUTPUT_DIR = Path("voice_output")

SNAC_REPO = "hubertsiuzdak/snac_24khz"
# Fixed by the vocoder's architecture, not preferences - snac_24khz emits
# 24 kHz mono, and its three codebooks are consumed 1:2:4 per frame.
SAMPLE_RATE = 24000
TOKENS_PER_FRAME = 7
CODEBOOK_SIZE = 4096
# <custom_token_0..9> are control markers (start/end of speech); real audio
# tokens start at 10, which is why decoding subtracts it.
FIRST_AUDIO_TOKEN = 10

# Voices the en-3b checkpoint was trained on. Anything else still generates,
# but in an unpredictable voice, so it is worth catching the typo.
VOICES = ("tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe")
DEFAULT_VOICE = "tara"

# Orpheus was trained on single utterances and simply stops early on longer
# input - measured here at ~50s of audio from 1044 characters, with the tail
# missing. It is not a context-window error and Ollama reports an ordinary
# "stop", so there is nothing to detect after the fact: the only reliable
# fix is to not ask for too much at once. Longer text is split and the
# pieces joined back together.
MAX_CHUNK_CHARS = 400
# Silence inserted between those pieces, so two separately generated
# sentences don't butt up against each other.
CHUNK_GAP_SECONDS = 0.15

# How long a synthesized wav is kept before cleanup() removes it.
DEFAULT_RETENTION_HOURS = 24

_TOKEN_RE = re.compile(r"<custom_token_(\d+)>")

# One process-wide vocoder shared by every caller, so its use has to be
# serialised: torch modules are not safe to call concurrently, and two
# threads racing the lazy load would fetch and build it twice. Callers can
# be genuinely concurrent - ai_agent_server.py runs requests in parallel
# threads. Only the decode is held here; the slow part of synthesize(), the
# generation request to Ollama, stays outside so it still overlaps.
_vocoder_lock = threading.Lock()


def _safe(name: str) -> str:
    """Filename-safe form of a caller-supplied name, so a voice or label
    can't reach outside the output directory."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


@lru_cache(maxsize=1)
def _vocoder() -> tuple[SNAC, str]:
    """The SNAC model, downloaded once and cached for the process.

    Uses the GPU when torch was built with CUDA, which is worth having:
    vocoding is a measurable share of each request (5.7s of a 39.7s call
    on CPU, ~4s per 21s of audio) and it scales with the length of the
    answer. Falls back to CPU otherwise - a CPU-only torch build, or a
    driver too old for the build's CUDA version, both just mean slower,
    never broken.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SNAC.from_pretrained(SNAC_REPO).eval().to(device)
    logger.info("Loaded SNAC vocoder %s on %s", SNAC_REPO, device)
    return model, device


def build_prompt(text: str, voice: str = DEFAULT_VOICE) -> str:
    """The raw prompt Orpheus expects: a start-of-speech marker, the voice
    and text, then end-of-turn and start-of-audio. Sent with raw=True, so
    this is verbatim what the model sees."""
    return f"<custom_token_3><|begin_of_text|>{voice}: {text}<|eot_id|><custom_token_4>"


def parse_snac_tokens(response: str) -> list[int]:
    """SNAC codebook entries from a raw Orpheus reply.

    Each token's codebook slot comes from its *position*: the nth audio
    token belongs to slot n % 7 and its id is offset by that slot's block
    of 4096. The model also emits low control tokens (its own
    <custom_token_5><custom_token_1> start-of-audio marker); those are
    dropped first so they don't shift every position after them.
    """
    tokens = [int(n) for n in _TOKEN_RE.findall(response)]
    audio = [n for n in tokens if n >= FIRST_AUDIO_TOKEN]

    codes: list[int] = []
    for position, token in enumerate(audio):
        code = token - FIRST_AUDIO_TOKEN - (position % TOKENS_PER_FRAME) * CODEBOOK_SIZE
        if not 0 <= code < CODEBOOK_SIZE:
            # Position determines the slot, so one out-of-range token means
            # the stream has desynchronised and everything after it decodes
            # against the wrong codebook. Keep the good prefix, drop the rest.
            logger.warning(
                "Token %d at position %d decodes to %d, outside [0, %d) - "
                "truncating after %d good token(s)",
                token, position, code, CODEBOOK_SIZE, len(codes),
            )
            break
        codes.append(code)

    # The vocoder needs whole frames; a partial trailing one is unusable.
    return codes[: len(codes) - len(codes) % TOKENS_PER_FRAME]


def _to_codebooks(codes: list[int], device: str) -> list[torch.Tensor]:
    """Deal a flat token stream into SNAC's three codebooks. Within each
    7-token frame the layout is fixed: slot 0 is the coarse code, slots 1
    and 4 the mid codes, and slots 2, 3, 5, 6 the fine ones."""
    coarse, mid, fine = [], [], []
    for frame_start in range(0, len(codes), TOKENS_PER_FRAME):
        frame = codes[frame_start : frame_start + TOKENS_PER_FRAME]
        coarse.append(frame[0])
        mid += [frame[1], frame[4]]
        fine += [frame[2], frame[3], frame[5], frame[6]]

    return [
        torch.tensor(level, dtype=torch.int32, device=device).unsqueeze(0)
        for level in (coarse, mid, fine)
    ]


_SENTENCE_END = re.compile(r"(?<=[.,!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Break `text` into individual sentences on sentence-ending punctuation,
    and also on commas - splitting mid-sentence too, so a short answer with
    only one or two real sentences still yields more than one piece for
    synthesize() to generate concurrently.
    """
    return [s for s in (piece.strip() for piece in _SENTENCE_END.split(text.strip())) if s]


def split_text(text: str, limit: int = MAX_CHUNK_CHARS) -> list[str]:
    """Break `text` into pieces small enough for one synthesize call.

    One chunk per sentence - never merged into fewer, larger ones - so
    synthesize() can generate independent sentences concurrently instead of
    paying for one big serial call. A single sentence longer than `limit`
    is split on whitespace instead, which is audible but still better than
    losing it.
    """
    chunks: list[str] = []
    for sentence in split_sentences(text):
        if not sentence:
            continue

        if len(sentence) <= limit:
            chunks.append(sentence)
            continue

        while len(sentence) > limit:
            cut = sentence.rfind(" ", 0, limit)
            if cut <= 0:  # one unbroken run of characters - cut mid-word
                cut = limit
            chunks.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if sentence:
            chunks.append(sentence)

    return chunks


def _join(segments: list[np.ndarray]) -> np.ndarray:
    """Concatenate chunk audio with a short gap, so sentences don't run
    into each other where two separate generations meet."""
    if len(segments) == 1:
        return segments[0]

    gap = np.zeros(int(SAMPLE_RATE * CHUNK_GAP_SECONDS), dtype=np.float32)
    joined: list[np.ndarray] = []
    for index, segment in enumerate(segments):
        if index:
            joined.append(gap)
        joined.append(segment)
    return np.concatenate(joined)


def _write_wav(path: Path, samples: np.ndarray) -> None:
    """16-bit mono PCM. Uses the stdlib wave module rather than soundfile
    to keep this off libsndfile."""
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767).astype("<i2")
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes(pcm.tobytes())


class TextToVoiceError(RuntimeError):
    """The model returned nothing that could be decoded into audio."""


class TextToVoice:
    """
    Usage:
        voice = TextToVoice("http://192.168.1.57:11434", "sematre/orpheus:en-3b")
        path = voice.synthesize("The outside temperature is 27 degrees.")

    Unlike OllamaClient this holds no conversation - each synthesize() call
    is independent, so one instance can be reused for everything.
    """

    def __init__(
        self,
        url: str,
        model: str,
        voice: str = DEFAULT_VOICE,
        output_dir: str | Path = DEFAULT_OUTPUT_DIR,
        options: dict | None = None,
        timeout: float = 180,
        retention_hours: float = DEFAULT_RETENTION_HOURS,
        keep_alive: str | int = -1,
        max_workers: int = 4,
    ):
        self.url = url.rstrip("/")
        self._model = model
        self.voice = voice
        # Synthesized audio is only useful until the client has fetched it,
        # and at roughly 1.5 MB per 30s answer it would otherwise grow
        # without limit. 0 or less keeps everything.
        self.retention_hours = retention_hours
        # Left empty by default so the model's own Modelfile settings apply -
        # Orpheus ships tuned temperature/top_p/repeat_penalty values, and
        # overriding them tends to make the audio worse, not better.
        self.options = options or {}
        # Generating even a short line takes far longer than a chat reply:
        # roughly 84 audio tokens per second of speech.
        self.timeout = timeout
        # -1 keeps the model loaded indefinitely, matching how the chat
        # models (OllamaClient) are kept resident - Ollama's own 5m default
        # would otherwise unload it between manual/spaced-out runs, paying
        # a multi-second reload on the next call.
        self.keep_alive = keep_alive
        # Chunks are independent model calls (see synthesize()), so they can
        # run concurrently instead of one after another - this bounds how
        # many are ever in flight against the Ollama host at once.
        self.max_workers = max_workers

        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        if voice not in VOICES:
            logger.warning(
                "Voice %r is not one of the trained voices %s - output voice is undefined",
                voice, ", ".join(VOICES),
            )

    @property
    def model(self) -> str:
        """Read-only - the model is fixed for the life of the client."""
        return self._model

    def cleanup(self, retention_hours: float | None = None) -> int:
        """Delete wavs in the output directory last modified more than
        retention_hours ago, returning how many were removed.

        Called from synthesize(), so anything generating audio prunes as it
        goes and no separate job is needed. Files written to an explicit
        path outside the output directory are the caller's to manage.
        """
        hours = self.retention_hours if retention_hours is None else retention_hours
        if hours <= 0:
            return 0

        cutoff = time.time() - hours * 3600
        removed = freed = 0
        # Non-recursive, and *.wav only. This deletes things, so the reach
        # stays exactly the files this class writes - no walking into
        # subdirectories a user may have put here, no other extensions.
        for wav in sorted(self._output_dir.glob("*.wav")):
            try:
                stat = wav.stat()
                if not wav.is_file() or stat.st_mtime >= cutoff:
                    continue
                wav.unlink()
            except OSError as e:
                # Raced with another process, or not ours to remove. Cleanup
                # is housekeeping and must never fail the synthesis it runs
                # alongside.
                logger.warning("Could not remove %s: %s", wav, e)
                continue
            removed += 1
            freed += stat.st_size

        if removed:
            logger.info(
                "Removed %d wav file(s) older than %gh from %s, freeing %.1f MB",
                removed, hours, self._output_dir, freed / 1e6,
            )
        return removed

    def _generate(self, text: str, voice: str) -> list[int]:
        """SNAC codes for one chunk of text, straight from the model."""
        response = requests.post(
            f"{self.url}/api/generate",
            json={
                "model": self.model,
                "prompt": build_prompt(text, voice),
                # Without this Ollama applies the model's chat template and
                # the Orpheus prompt format never reaches the model.
                "raw": True,
                "stream": False,
                "options": self.options,
                "keep_alive": self.keep_alive,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()

        codes = parse_snac_tokens(response.json().get("response", ""))
        if not codes:
            raise TextToVoiceError(
                f"{self.model} returned no usable SNAC tokens for {text[:60]!r} - "
                "check that it is an Orpheus-style TTS model"
            )
        return codes

    def _decode(self, codes: list[int]) -> np.ndarray:
        with _vocoder_lock:
            model, device = _vocoder()
            with torch.inference_mode():
                audio = model.decode(_to_codebooks(codes, device))
        return audio.squeeze().float().cpu().numpy()

    def _generate_timed(
        self, index: int, total: int, chunk: str, voice: str
    ) -> tuple[list[int], float]:
        mark = time.perf_counter()
        codes = self._generate(chunk, voice)
        duration = time.perf_counter() - mark
        logger.info(
            "  chunk %d/%d (%d char(s)) -> %d token(s) in %.2fs generating",
            index + 1, total, len(chunk), len(codes), duration,
        )
        return codes, duration

    def synthesize(
        self,
        text: str,
        path: str | Path | None = None,
        voice: str | None = None,
        stats: dict | None = None,
        max_workers: int | None = None,
    ) -> Path:
        """Speak `text` into a .wav and return where it was written.

        Text longer than MAX_CHUNK_CHARS is synthesized in pieces and
        joined, so the answer isn't cut off. Defaults to a timestamped file
        in the output directory; pass `path` to choose one. Raises
        TextToVoiceError if the model returned no usable audio tokens.

        Chunks are independent Ollama calls, so up to `max_workers` of them
        (self.max_workers by default) run at once instead of paying each
        chunk's generation time in series - decoding stays one at a time
        (see _vocoder_lock) since it is cheap by comparison and torch
        modules are not safe to call concurrently. Results are joined back
        in chunk order regardless of which finishes generating first.

        Pass `stats` (an empty dict) to have it filled in with timing and
        token counts - useful for comparing concurrent calls against each
        other. Left alone by default.
        """
        voice = voice or self.voice
        started = time.perf_counter()
        # Before generating, not after: a run that fails still prunes, and
        # the file about to be written is never a candidate.
        self.cleanup()

        chunks = split_text(text)
        if not chunks:
            raise TextToVoiceError("No text to speak")

        # Timed apart so the log says which half the time went to: the model
        # generating tokens on the Ollama host, or this process vocoding them.
        # They answer different questions - one is a remote GPU's problem, the
        # other is this machine's. generating is a sum of per-chunk time, so
        # it can exceed wall-clock elapsed once chunks overlap - it still
        # answers "how much model time did this cost", just not serially.
        workers = max(1, min(max_workers or self.max_workers, len(chunks)))
        count = len(chunks)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            generated = list(pool.map(
                self._generate_timed,
                range(count), [count] * count, chunks, [voice] * count,
            ))

        segments, frames, generating, vocoding = [], 0, 0.0, 0.0
        for codes, duration in generated:
            generating += duration
            frames += len(codes) // TOKENS_PER_FRAME
            mark = time.perf_counter()
            segments.append(self._decode(codes))
            vocoding += time.perf_counter() - mark
        samples = _join(segments)

        if path is None:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            path = self._output_dir / f"{_safe(voice)}-{stamp}.wav"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_wav(path, samples)

        seconds = len(samples) / SAMPLE_RATE
        elapsed = time.perf_counter() - started
        logger.info(
            "synthesize(%s, voice=%s) %d char(s) in %d chunk(s), %d frame(s) -> "
            "%.1fs of audio in %.2fs (%.1fx realtime, %.2fs generating, "
            "%.2fs vocoding on %s) -> %s",
            self.model, voice, len(text), len(chunks), frames, seconds,
            elapsed, seconds / elapsed if elapsed else 0, generating,
            vocoding, _vocoder()[1], path,
        )

        if stats is not None:
            stats.update(
                chunks=len(chunks),
                tokens=frames * TOKENS_PER_FRAME,
                audio_seconds=seconds,
                elapsed=elapsed,
                generating=generating,
                vocoding=vocoding,
            )
        return path


DEMO_TEXT_FULL = "Interior room has a temperature of 26.45°C and relative humidity of 58%. \
	Exterior temperature is slightly cooler at 20.96°C, with similar humidity.\
	Interior room's conditions are about 5.49°C warmer than outside."

DEMO_TEXT_1 = split_sentences(DEMO_TEXT_FULL)

# Independent of len(DEMO_TEXT_1): a pool smaller than the text list queues
# the extra calls instead of running everything at once.
DEMO_MAX_WORKERS = 4 if len(DEMO_TEXT_1) >= 4 else len(DEMO_TEXT_1)


def _demo_run(index: int, config) -> tuple[int, Path | None, dict, float, str | None]:
    """One simultaneous synthesize() call for demo(). A fresh TextToVoice
    per call, matching independent concurrent requests rather than one
    instance shared across threads."""
    voice = TextToVoice(
        config.ollama_url, config.ollama_model_text_to_voice, voice=config.ollama_voice
    )
    stats: dict = {}
    started = time.perf_counter()
    try:
        path = voice.synthesize(DEMO_TEXT_1[index], stats=stats)
    except Exception as e:  # noqa: BLE001 - reported per-instance, not raised
        return index, None, stats, time.perf_counter() - started, str(e)
    return index, path, stats, time.perf_counter() - started, None


def demo():
    from Config import Config

    config = Config.from_env()
    config.configure_logging()

    logger.info("Start")

    count = len(DEMO_TEXT_1)
    started = time.perf_counter()
    generating = vocoding = 0.0
    failed = 0
    with ThreadPoolExecutor(max_workers=DEMO_MAX_WORKERS) as pool:
        results = pool.map(_demo_run, range(count), [config] * count)

        for index, path, stats, wall_time, error in results:
            if error is not None:
                failed += 1
                print(f"[{index}] failed after {wall_time:.2f}s: {error}")
                continue
            generating += stats["generating"]
            vocoding += stats["vocoding"]
            print(
                f"[{index}] wrote {path} in {wall_time:.2f}s wall "
                f"({stats['tokens']} tokens, {stats['chunks']} chunk(s), "
                f"{stats['elapsed']:.2f}s total, {stats['generating']:.2f}s "
                f"generating, {stats['vocoding']:.2f}s vocoding)"
            )

    elapsed = time.perf_counter() - started
    print(
        f"Total: {count} item(s), {failed} failed, {elapsed:.2f}s wall "
        f"(sum {generating:.2f}s generating, {vocoding:.2f}s vocoding across "
        f"{DEMO_MAX_WORKERS} worker(s))"
    )
    logger.info("Finish")


if __name__ == "__main__":
    demo()
