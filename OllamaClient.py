"""
One Ollama model + one conversation. Keeps its own message history and
persists it to disk after every exchange, so many instances can run side
by side (each with its own model/url/session) without stepping on each
other's state, and a new instance can resume a prior conversation just by
passing back the same session_id.
"""

import functools
import json
import logging
import re
import time
import uuid
from pathlib import Path

import requests

logger = logging.getLogger("ollama")

DEFAULT_CONTEXT_DIR = Path("ollama_sessions")

# Rough token budget for the verbatim history chat() sends, and how many of
# the most recent messages are always kept verbatim once that budget is
# exceeded. Both are conservative defaults sized for a small model's default
# context window - tune them (via the constructor) to the actual num_ctx of
# whatever model you point this at.
DEFAULT_MAX_HISTORY_TOKENS = 3000
DEFAULT_KEEP_RECENT_MESSAGES = 10


def _estimate_tokens(messages: list[dict]) -> int:
    """~4 chars/token is a coarse but model-agnostic estimate - good enough
    to decide when to prune without calling out to a tokenizer."""
    chars = sum(len(m.get("content", "")) for m in messages)
    return chars // 4


def _slug(model: str) -> str:
    """Filename-safe form of a model name - 'llama3.1:8b' -> 'llama3.1_8b'.
    Model names are caller-supplied and may contain ':' or '/', so this also
    keeps them from reaching outside the context dir.
    """
    return re.sub(r"[^A-Za-z0-9._-]", "_", model)


def session_id_for(model: str, name: str) -> str:
    """A session id naming both the conversation and the model running it -
    session_id_for("llama3.1:8b", "kitchen") -> "llama3.1_8b-kitchen".
    For callers driving several models through one logical session: each
    needs an id of its own, or they share a history file and overwrite
    each other.
    """
    return f"{_slug(model)}-{name}"


def _format_tokens(metrics: dict | None) -> str:
    """Token counts and generation rate from an Ollama response, as a log
    suffix. Empty string if the server did not report them.
    """
    if not metrics:
        return ""

    generated = metrics.get("eval_count")
    prompt = metrics.get("prompt_eval_count")
    counts = [f"{n} {label}" for n, label in ((generated, "gen"), (prompt, "prompt")) if n]
    if not counts:
        return ""

    suffix = " - " + " + ".join(counts) + " tokens"
    # eval_duration is generation time alone, in nanoseconds - a fairer rate
    # than dividing by wall time, which also covers queueing and the prompt.
    eval_ns = metrics.get("eval_duration")
    if generated and eval_ns:
        suffix += f", {generated / (eval_ns / 1e9):.1f} tok/s"
    return suffix


def _timed(method):
    """Log which model was called, how long it took, and what it produced.
    Wraps in try/finally so a timeout or HTTP error is timed too - those are
    the calls whose duration is most worth seeing. For methods of a class
    exposing .model, which may record an Ollama response dict in
    self._last_metrics to have its token counts logged too.
    """

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        self._last_metrics = None  # never report the previous call's numbers
        start = time.perf_counter()
        ok = False
        try:
            result = method(self, *args, **kwargs)
            ok = True
            return result
        finally:
            logger.log(
                logging.INFO if ok else logging.WARNING,
                "%s(%s) %s in %.2fs%s",
                method.__name__,
                self.model,
                "ok" if ok else "failed",
                time.perf_counter() - start,
                _format_tokens(self._last_metrics),
            )

    return wrapper


class OllamaClient:
    """
    Usage:
        bot = OllamaClient("http://192.168.1.57:11434", "llama3.1:8b")
        print(bot.chat("How do I check disk usage on my Pi?"))
        print(bot.chat("What about just the SD card partition?"))

        # Resume the same conversation later (new process, new instance):
        bot2 = OllamaClient("http://192.168.1.57:11434", "llama3.1:8b", session_id=bot.session_id)

    An auto-generated session_id is prefixed with the model, so the file it
    saves to says which model wrote it - ollama_sessions/llama3.1_8b-<hex>.json.
    """

    def __init__(
        self,
        url: str,
        model: str,
        system: str | None = None,
        session_id: str | None = None,
        context_dir: str | Path = DEFAULT_CONTEXT_DIR,
        options: dict | None = None,
        timeout: float = 60,
        max_history_tokens: int | None = DEFAULT_MAX_HISTORY_TOKENS,
        keep_recent_messages: int = DEFAULT_KEEP_RECENT_MESSAGES,
    ):
        self.url = url.rstrip("/")
        self._model = model
        self.options = options or {}
        self.timeout = timeout
        # The model goes in the id, not just the filename, so session_id stays
        # the one key to the file - ids issued before this still resolve, and
        # resuming under a different model still finds the conversation.
        self._session_id = session_id or session_id_for(model, uuid.uuid4().hex)

        self._context_dir = Path(context_dir)
        self._context_dir.mkdir(parents=True, exist_ok=True)
        self._context_file = self._context_dir / f"{self.session_id}.json"

        # Not persisted with the rest of the history - re-sent fresh on every
        # request, so changing it doesn't mean rewriting old session files.
        self._system = system
        # None disables pruning entirely. Rounded up to even so the kept tail
        # always starts on a user turn, matching chat()'s user/assistant
        # alternation.
        self.max_history_tokens = max_history_tokens
        self.keep_recent_messages = keep_recent_messages + (keep_recent_messages % 2)
        self._last_metrics: dict | None = None
        self._summary, self._messages = self._load()

    @property
    def model(self) -> str:
        """Read-only - the model is fixed for the life of the client."""
        return self._model

    @property
    def session_id(self) -> str:
        """Read-only - the context file is resolved from it once, at
        construction, so reassigning it would only change the label while
        the client kept writing the original file. Point a new client at
        the conversation instead: OllamaClient(url, model, session_id=...).
        """
        return self._session_id

    @property
    def context(self) -> list[dict]:
        """Read-only view of the verbatim (unsummarized) conversation
        history - what chat() would send is this plus summary and system."""
        return list(self._messages)

    @property
    def summary(self) -> str | None:
        """Read-only - the folded-in summary of turns older than the kept
        tail, or None if history hasn't needed pruning yet."""
        return self._summary

    @_timed
    def chat(self, prompt: str) -> str:
        """Send `prompt` plus the saved history: system prompt, then the
        summary of anything pruned off the front, then the verbatim recent
        turns. Appends both sides of the exchange to that history, pruning
        the oldest turns into the summary first if it's grown past
        max_history_tokens."""
        self._messages.append({"role": "user", "content": prompt})
        self._enforce_history_budget()

        history = self._messages
        if self._summary:
            history = [
                {"role": "system", "content": f"Earlier in this conversation: {self._summary}"}
            ] + history
        reply = self._request(history)

        self._messages.append({"role": "assistant", "content": reply})
        self._save()
        return reply

    @_timed
    def chat_once(self, prompt: str) -> str:
        """Send `prompt` with the system prompt only - no saved history sent,
        and nothing about this call is appended to it or persisted. For
        calls that don't need prior turns, e.g. summarizing freshly queried
        data - use chat() for anything that should feel like a conversation.
        """
        return self._request([{"role": "user", "content": prompt}])

    def reset(self):
        """Drop conversation history and any accumulated summary. The
        system prompt isn't part of either - it's re-sent fresh from
        self._system on every call - so there's nothing of it to keep."""
        self._messages = []
        self._summary = None
        self._save()

    def _enforce_history_budget(self) -> None:
        """Once self._messages passes max_history_tokens, fold everything
        but the most recent keep_recent_messages into self._summary via one
        extra model call. Ollama would otherwise silently drop whatever
        falls off the front once the model's context window fills - this
        keeps that loss intentional and recorded instead of implicit."""
        if self.max_history_tokens is None:
            return
        if _estimate_tokens(self._messages) <= self.max_history_tokens:
            return
        if len(self._messages) <= self.keep_recent_messages:
            return  # nothing older than the kept tail to fold in

        # chat() calls this right after appending the new user prompt, so
        # self._messages is transiently odd-length (...assistant, user).
        # Round the boundary down to even so `recent` still starts on a
        # user turn rather than a dangling assistant one.
        boundary = len(self._messages) - self.keep_recent_messages
        boundary -= boundary % 2
        if boundary <= 0:
            return
        older, recent = self._messages[:boundary], self._messages[boundary:]

        summary = self._summarize(older)
        self._summary = summary
        self._messages = recent
        logger.info(
            "chat(%s): folded %d older message(s) into summary (session %s)",
            self.model, len(older), self.session_id,
        )

    def _summarize(self, messages: list[dict]) -> str:
        """One-off, unlogged-by-@_timed call asking the same model to
        compress `messages` (plus any existing summary, so repeated pruning
        keeps compounding instead of losing everything before the last
        round) into a short paragraph of prior context."""
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        if self._summary:
            transcript = f"Summary so far: {self._summary}\n{transcript}"
        instruction = (
            "Summarize the earlier conversation below in a short paragraph, "
            "keeping any facts, numbers, or decisions a reader would need to "
            "follow later messages. Do not add anything not present below.\n\n"
            + transcript
        )
        response = requests.post(
            f"{self.url}/api/chat",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": instruction}],
                "stream": False,
                "options": self.options,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    def _request(self, messages: list[dict]) -> str:
        request_messages = messages
        if self._system:
            request_messages = [{"role": "system", "content": self._system}] + messages

        response = requests.post(
            f"{self.url}/api/chat",
            json={
                "model": self.model,
                "messages": request_messages,
                "stream": False,
                "options": self.options,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()

        payload = response.json()
        self._last_metrics = payload  # picked up by @_timed for the log line
        return payload["message"]["content"]

    def _load(self) -> tuple[str | None, list[dict]]:
        if not self._context_file.exists():
            return None, []

        data = json.loads(self._context_file.read_text())
        if isinstance(data, list):
            # Pre-summary session file: just the message list.
            summary, messages = None, data
        else:
            summary, messages = data.get("summary"), data.get("messages", [])

        # Older session files may have a persisted system message from
        # before it moved to self._system - drop it so it isn't resent as
        # a stray turn and duplicated alongside the fresh one.
        messages = [m for m in messages if m.get("role") != "system"]
        return summary, messages

    def _save(self):
        payload = {"summary": self._summary, "messages": self._messages}
        self._context_file.write_text(json.dumps(payload, indent=2))


def demo():
    bot = OllamaClient("http://<pi-ip>:11434", "llama3.2", system="You are a concise assistant.")
    print("Session ID:", bot.session_id)

    print(bot.chat("How do I check disk usage on my Pi?"))
    print(bot.chat("What about just the SD card partition?"))


if __name__ == "__main__":
    demo()
