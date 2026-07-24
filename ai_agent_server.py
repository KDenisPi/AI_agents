"""
Async HTTP API in front of AiAgent. A GET is answered immediately - 200 if
the request was accepted, or an error if it couldn't be even started (e.g.
Ollama isn't reachable) - and the actual answer follows later as a POST back
to the client, once the model call finishes. This keeps a slow model call
from holding an HTTP connection open, and matches a voice-assistant client
that wants to move on immediately and get spoken to when the answer is
ready.

Client to AI_AGENT:
    GET /api/current[?request_id=<id>][&voice[=<name>]]
        -> 200 (request accepted, answer to follow)
        -> 503 (model not available right now)
        -> 500 (something else went wrong before work could start)

    request_id is optional, opaque, and only used to echo back in the
    callback below - the server does not interpret it. Pass one if more
    than one request may be in flight, so the client can match the
    eventual callback to the request that caused it.

    voice is a bare flag - "&voice" is enough - and asks for the answer to
    be spoken as well as written. "&voice=leo" picks a specific voice from
    text_to_voice.VOICES instead of config.ollama_voice. Synthesis roughly
    doubles the time to the callback.

AI_AGENT to Client (config.ai_client_callback_url, always POST):
    POST /api/response
        {"request_id": "<id or omitted>", "text": "...", "audio_url": null}
        or, if the request failed after being accepted:
        {"request_id": "<id or omitted>", "error": "..."}

    audio_url is null unless the request asked for voice. When it did, it
    is a link to a WAV served from /audio (see config.voice_output_dir) -
    root-relative by default, absolute if config.ai_agent_public_url is
    set. If synthesis alone fails the text answer is still delivered, with
    audio_url null and the reason in an extra "audio_error" field: a
    spoken answer is a nice-to-have, and losing it is no reason to throw
    away an answer that already cost a model call.

Every endpoint below follows this same accept-then-callback shape; see
handle_current for the pattern to copy when adding another.

See ai_client/simple_client.py for a minimal client exercising both sides.

Install:
    pip install starlette uvicorn requests

Run:
    python ai_agent_server.py
"""

import asyncio
import contextlib
import logging
from pathlib import Path
from urllib.parse import quote

import requests
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from ai_agent import AiAgent
from Config import Config

logger = logging.getLogger("ai-agent-server")

# Where synthesized WAVs are served from, mounted on config.voice_output_dir.
AUDIO_URL_PREFIX = "/audio"

# asyncio.create_task() only holds a weak reference via the event loop - an
# otherwise-unreferenced task can be garbage-collected mid-flight. Keeping a
# strong reference here until each task finishes avoids that.
_background_tasks: set[asyncio.Task] = set()


def _spawn(coro) -> None:
    """Fire-and-forget a coroutine, without losing it to garbage collection
    before it completes."""
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


def _ollama_reachable(config: Config, timeout: float = 2) -> bool:
    """Cheap pre-check so a GET can fail fast with 503 instead of accepting
    work the model call is just going to fail anyway."""
    try:
        response = requests.get(f"{config.ollama_url}/api/tags", timeout=timeout)
        return response.ok
    except requests.RequestException:
        return False


def _audio_url(config: Config, path: Path) -> str:
    """Client-facing link to a synthesized WAV. Root-relative unless a
    public base URL is configured - see Config.ai_agent_public_url."""
    url = f"{AUDIO_URL_PREFIX}/{quote(path.name)}"
    base = config.ai_agent_public_url.rstrip("/")
    return f"{base}{url}" if base else url


async def _post_callback(config: Config, payload: dict) -> None:
    try:
        await asyncio.to_thread(
            requests.post,
            config.ai_client_callback_url,
            json=payload,
            timeout=config.ai_client_callback_timeout,
        )
    except requests.RequestException as e:
        logger.error("Could not deliver callback to %s: %s", config.ai_client_callback_url, e)


def make_app(config: Config) -> Starlette:
    async def handle_current(request: Request) -> Response:
        request_id = request.query_params.get("request_id")
        # "voice" is a bare flag, so its value is the empty string - falsy,
        # which is why presence is tested by membership and not .get().
        # "voice=leo" names a voice; plain "voice" leaves it to the agent.
        wants_audio = "voice" in request.query_params
        voice = request.query_params.get("voice") or None
        try:
            reachable = await asyncio.to_thread(_ollama_reachable, config)
        except Exception as e:
            logger.exception("Could not check Ollama availability")
            return JSONResponse({"error": str(e)}, status_code=500)
        if not reachable:
            return JSONResponse({"error": "model not available"}, status_code=503)

        _spawn(
            _run_current(config, request.app.state.agent, request_id, wants_audio, voice)
        )
        body = {"accepted": True}
        if request_id:
            body["request_id"] = request_id
        return JSONResponse(body, status_code=200)

    async def _run_current(
        config: Config,
        agent: AiAgent,
        request_id: str | None,
        wants_audio: bool,
        voice: str | None,
    ) -> None:
        try:
            text = await asyncio.to_thread(agent.summarize_current)
            payload = {"text": text, "audio_url": None}
        except Exception as e:
            logger.exception("summarize_current failed")
            payload = {"error": str(e)}
        else:
            if wants_audio:
                try:
                    path = await asyncio.to_thread(agent.say, text, voice=voice)
                    payload["audio_url"] = _audio_url(config, path)
                except Exception as e:
                    # The text answer has already cost a model call - send it
                    # rather than lose it to a failed nice-to-have.
                    logger.exception("Speech synthesis failed")
                    payload["audio_error"] = str(e)
        if request_id:
            payload["request_id"] = request_id
        await _post_callback(config, payload)

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette):
        app.state.agent = AiAgent(config)
        try:
            yield
        finally:
            app.state.agent.close()

    # StaticFiles refuses to mount a directory that doesn't exist yet, and
    # routes are built before the agent (which would create it) exists.
    Path(config.voice_output_dir).mkdir(parents=True, exist_ok=True)

    return Starlette(
        routes=[
            Route("/api/current", handle_current, methods=["GET"]),
            Mount(
                AUDIO_URL_PREFIX,
                StaticFiles(directory=config.voice_output_dir),
                name="audio",
            ),
        ],
        lifespan=lifespan,
    )


config = Config.from_env()
if config.log_file == Config.log_file:  # LOG_FILE not overridden via env
    config.log_file = "logs/ai_agent_server.log"
config.configure_logging()
app = make_app(config)

if __name__ == "__main__":
    uvicorn.run(app, host=config.ai_agent_host, port=config.ai_agent_port)
