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
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import requests
import torch
from snac import SNAC

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


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def split_text(text: str, limit: int = MAX_CHUNK_CHARS) -> list[str]:
    """Break `text` into pieces small enough for one synthesize call.

    Splits on sentence boundaries so the joins land where a speaker would
    pause anyway; a single sentence longer than `limit` is split on
    whitespace instead, which is audible but still better than losing it.
    """
    chunks: list[str] = []
    pending = ""

    def flush() -> None:
        nonlocal pending
        if pending:
            chunks.append(pending)
            pending = ""

    for sentence in _SENTENCE_END.split(text.strip()):
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(sentence) > limit:
            flush()
            while len(sentence) > limit:
                cut = sentence.rfind(" ", 0, limit)
                if cut <= 0:  # one unbroken run of characters - cut mid-word
                    cut = limit
                chunks.append(sentence[:cut].strip())
                sentence = sentence[cut:].strip()
            pending = sentence
        elif not pending:
            pending = sentence
        elif len(pending) + 1 + len(sentence) <= limit:
            pending = f"{pending} {sentence}"
        else:
            flush()
            pending = sentence

    flush()
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

    def synthesize(
        self, text: str, path: str | Path | None = None, voice: str | None = None
    ) -> Path:
        """Speak `text` into a .wav and return where it was written.

        Text longer than MAX_CHUNK_CHARS is synthesized in pieces and
        joined, so the answer isn't cut off. Defaults to a timestamped file
        in the output directory; pass `path` to choose one. Raises
        TextToVoiceError if the model returned no usable audio tokens.
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
        # other is this machine's.
        segments, frames, generating, vocoding = [], 0, 0.0, 0.0
        for chunk in chunks:
            mark = time.perf_counter()
            codes = self._generate(chunk, voice)
            generating += time.perf_counter() - mark

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
        return path


def demo():
    from Config import Config

    config = Config.from_env()
    config.configure_logging()

    voice = TextToVoice(
        config.ollama_url, config.ollama_model_text_to_voice, voice=config.ollama_voice
    )
    print("Wrote", voice.synthesize("Hello. The outside temperature is 27 degrees."))


if __name__ == "__main__":
    demo()
