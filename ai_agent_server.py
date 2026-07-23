"""
Async HTTP API in front of AiAgent. A GET is answered immediately - 200 if
the request was accepted, or an error if it couldn't be even started (e.g.
Ollama isn't reachable) - and the actual answer follows later as a POST back
to the client, once the model call finishes. This keeps a slow model call
from holding an HTTP connection open, and matches a voice-assistant client
that wants to move on immediately and get spoken to when the answer is
ready.

Client to AI_AGENT:
    GET /api/current[?request_id=<id>]
        -> 200 (request accepted, answer to follow)
        -> 503 (model not available right now)
        -> 500 (something else went wrong before work could start)

    request_id is optional, opaque, and only used to echo back in the
    callback below - the server does not interpret it. Pass one if more
    than one request may be in flight, so the client can match the
    eventual callback to the request that caused it.

AI_AGENT to Client (config.ai_client_callback_url, always POST):
    POST /api/response
        {"request_id": "<id or omitted>", "text": "...", "audio_url": null}
        or, if the request failed after being accepted:
        {"request_id": "<id or omitted>", "error": "..."}

    audio_url is always null for now - reserved for a future text-to-speech
    step that would drop a WAV under a static folder and link it here.

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

import requests
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from ai_agent import AiAgent
from Config import Config

logger = logging.getLogger("ai-agent-server")

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
        try:
            reachable = await asyncio.to_thread(_ollama_reachable, config)
        except Exception as e:
            logger.exception("Could not check Ollama availability")
            return JSONResponse({"error": str(e)}, status_code=500)
        if not reachable:
            return JSONResponse({"error": "model not available"}, status_code=503)

        _spawn(_run_current(config, request.app.state.agent, request_id))
        body = {"accepted": True}
        if request_id:
            body["request_id"] = request_id
        return JSONResponse(body, status_code=200)

    async def _run_current(config: Config, agent: AiAgent, request_id: str | None) -> None:
        try:
            text = await asyncio.to_thread(agent.summarize_current)
            payload = {"text": text, "audio_url": None}
        except Exception as e:
            logger.exception("summarize_current failed")
            payload = {"error": str(e)}
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

    return Starlette(
        routes=[Route("/api/current", handle_current, methods=["GET"])],
        lifespan=lifespan,
    )


config = Config.from_env()
if config.log_file == Config.log_file:  # LOG_FILE not overridden via env
    config.log_file = "logs/ai_agent_server.log"
config.configure_logging()
app = make_app(config)

if __name__ == "__main__":
    uvicorn.run(app, host=config.ai_agent_host, port=config.ai_agent_port)
