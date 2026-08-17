"""
Raw, template-free single-shot text completion against either an Ollama or
a bare llama-server instance. "Raw" means the prompt is sent exactly as
given - no chat template applied - and nothing is kept between calls: no
history, no session. That's what a model trained on its own from-scratch
prompt format (e.g. Orpheus's SNAC-token format, see text_to_voice.py)
needs, and it's the one place Ollama and llama-server genuinely diverge -
unlike the OpenAI-compatible /v1/chat/completions endpoint (OllamaClient),
which is identical on both, raw completion is a different, backend-specific
endpoint on each:
    Ollama:       POST /api/generate   {"raw": true, ...}
    llama-server: POST /completion     (never applies a chat template at all)

Usage:
    client = RawCompletionClient("http://192.168.1.57:11434", "sematre/orpheus:en-3b")
    text = client.generate(prompt)

    client = RawCompletionClient("http://192.168.1.57:8080", backend="llama-server")
    text = client.generate(prompt)
"""

import logging
import time

import requests

logger = logging.getLogger("raw-completion")

BACKENDS = ("ollama", "llama-server")


class RawCompletionClient:
    def __init__(
        self,
        url: str,
        model: str | None = None,
        backend: str = "ollama",
        options: dict | None = None,
        timeout: float = 180,
        # -1 keeps the model loaded indefinitely between calls - Ollama only,
        # since a llama-server process has exactly one model loaded for its
        # whole lifetime and there is nothing to keep alive.
        keep_alive: str | int = -1,
    ):
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
        self.url = url.rstrip("/")
        # llama-server serves the one model it was started with and ignores
        # any name given to it - model only matters for the "ollama" backend.
        self.model = model
        self.backend = backend
        self.options = options or {}
        self.timeout = timeout
        self.keep_alive = keep_alive

    def generate(self, prompt: str) -> str:
        start = time.perf_counter()
        ok = False
        try:
            if self.backend == "llama-server":
                text = self._generate_llama_server(prompt)
            else:
                text = self._generate_ollama(prompt)
            ok = True
            return text
        finally:
            logger.log(
                logging.INFO if ok else logging.WARNING,
                "generate(%s/%s) %s in %.2fs",
                self.backend, self.model, "ok" if ok else "failed",
                time.perf_counter() - start,
            )

    def _generate_ollama(self, prompt: str) -> str:
        response = requests.post(
            f"{self.url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                # Without this Ollama applies the model's chat template and
                # a from-scratch prompt format never reaches the model.
                "raw": True,
                "stream": False,
                "options": self.options,
                "keep_alive": self.keep_alive,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json().get("response", "")

    def _generate_llama_server(self, prompt: str) -> str:
        # llama.cpp's own /completion, not the OpenAI-compatible
        # /v1/completions - it never applies a chat template (there is no
        # "raw" flag to set because there is nothing to opt out of).
        response = requests.post(
            f"{self.url}/completion",
            json={"prompt": prompt, "stream": False, **self.options},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json().get("content", "")
