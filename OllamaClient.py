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

        self._last_metrics: dict | None = None
        self._messages: list[dict] = self._load()
        if not self._messages and system:
            self._messages.append({"role": "system", "content": system})

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
        """Read-only view of the current conversation history."""
        return list(self._messages)

    @_timed
    def chat(self, prompt: str) -> str:
        self._messages.append({"role": "user", "content": prompt})

        response = requests.post(
            f"{self.url}/api/chat",
            json={
                "model": self.model,
                "messages": self._messages,
                "stream": False,
                "options": self.options,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()

        payload = response.json()
        self._last_metrics = payload  # picked up by @_timed for the log line
        reply = payload["message"]["content"]
        self._messages.append({"role": "assistant", "content": reply})
        self._save()
        return reply

    def reset(self):
        """Drop conversation history, keeping the system prompt if one was set."""
        self._messages = [m for m in self._messages if m["role"] == "system"]
        self._save()

    def _load(self) -> list[dict]:
        if self._context_file.exists():
            return json.loads(self._context_file.read_text())
        return []

    def _save(self):
        self._context_file.write_text(json.dumps(self._messages, indent=2))


def demo():
    bot = OllamaClient("http://<pi-ip>:11434", "llama3.2", system="You are a concise assistant.")
    print("Session ID:", bot.session_id)

    print(bot.chat("How do I check disk usage on my Pi?"))
    print(bot.chat("What about just the SD card partition?"))


if __name__ == "__main__":
    demo()
