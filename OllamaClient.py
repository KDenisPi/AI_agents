"""
One Ollama model + one conversation. Keeps its own message history and
persists it to disk after every exchange, so many instances can run side
by side (each with its own model/url/session) without stepping on each
other's state, and a new instance can resume a prior conversation just by
passing back the same session_id.
"""

import json
import uuid
from pathlib import Path

import requests

DEFAULT_CONTEXT_DIR = Path("ollama_sessions")


class OllamaClient:
    """
    Usage:
        bot = OllamaClient("http://pi-host:11434", "llama3.2")
        print(bot.chat("How do I check disk usage on my Pi?"))
        print(bot.chat("What about just the SD card partition?"))

        # Resume the same conversation later (new process, new instance):
        bot2 = OllamaClient("http://pi-host:11434", "llama3.2", session_id=bot.session_id)
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
        self.session_id = session_id or uuid.uuid4().hex

        self._context_dir = Path(context_dir)
        self._context_dir.mkdir(parents=True, exist_ok=True)
        self._context_file = self._context_dir / f"{self.session_id}.json"

        self._messages: list[dict] = self._load()
        if not self._messages and system:
            self._messages.append({"role": "system", "content": system})

    @property
    def model(self) -> str:
        """Read-only - the model is fixed for the life of the client."""
        return self._model

    @property
    def context(self) -> list[dict]:
        """Read-only view of the current conversation history."""
        return list(self._messages)

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

        reply = response.json()["message"]["content"]
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
