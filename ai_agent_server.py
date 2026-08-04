"""
Async HTTP API in front of AiAgent. A GET is answered immediately - 200 if
the request was accepted, or an error if it couldn't be even started (e.g.
Ollama isn't reachable) - and the actual answer follows later as a POST back
to the client, once the model call finishes. This keeps a slow model call
from holding an HTTP connection open, and matches a voice-assistant client
that wants to move on immediately and get spoken to when the answer is
ready.

Client to AI_AGENT:
    GET /api/current[?request_id=<id>][&voice[=<name>]][&callback]
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

    callback is a bare flag - "&callback" is enough - and asks for the
    answer to also be POSTed to config.ai_client_callback_url once ready
    (see below). Omit it for a client that only polls /api/poll - a
    browser page, say - so the server doesn't waste an attempt POSTing to
    a URL nothing is listening on.

    GET /api/current_battery[?request_id=<id>][&voice[=<name>]][&callback]
        Same 200/503/500 contract as /api/current. Summarizes battery
        levels (AiAgent.summarize_current_battery). voice and callback
        work exactly as they do on /api/current.

    GET /api/outside_today[?request_id=<id>][&voice[=<name>]][&graph][&callback]
        Same 200/503/500 contract as /api/current. Summarizes today's
        outside temperature+humidity (AiAgent.summarize_outside_for_today,
        i.e. MetricStorage.today_outside() - 08:00-21:00 or now). voice and
        callback work exactly as they do on /api/current.

        graph is a bare flag - "&graph" is enough - and asks for a PNG per
        metric alongside the text.

    GET /api/outside_last_hours[?request_id=<id>][&hours=<n>][&graph][&callback]
        Same 200/503/500 contract as /api/current, plus 400 if hours is
        given but not a positive integer. Summarizes the last <hours> of
        outside temperature+humidity
        (AiAgent.summarize_history_outside_last_hours, i.e.
        MetricStorage.get_history_outside_last_hours). hours defaults to 24.
        Graphs only, no voice - graph works exactly as it does on
        /api/outside_today; there is no voice flag on this endpoint.
        callback works exactly as it does on /api/current.

AI_AGENT to Client (config.ai_client_callback_url, POST - only for a
request that asked for it with &callback, see above):
    POST /api/response
        {"request_id": "<id or omitted>", "text": "...", "audio_url": null, "graphs": null}
        or, if the request failed after being accepted:
        {"request_id": "<id or omitted>", "error": "..."}

    audio_url is null unless the request asked for voice. When it did, it
    is an absolute link to a WAV served from /audio (see
    config.voice_output_dir), built from the address the client used to
    reach this server - override it with config.ai_agent_public_url. WAVs
    are pruned after config.voice_retention_hours, so the link is not
    permanent. If synthesis alone fails the text answer is still delivered, with
    audio_url null and the reason in an extra "audio_error" field: a
    spoken answer is a nice-to-have, and losing it is no reason to throw
    away an answer that already cost a model call.

    graphs is null unless the request asked for one (only /api/outside_today
    and /api/outside_last_hours do today). When it did, it is
    {"<metric>": "<url>", ...} - one link
    per metric, served from /graphs (see config.graph_output_dir), built
    and pruned the same way audio_url is. {} means graph was requested but
    there was nothing plottable (see ai_server_graph.GraphError) - the text
    answer still went out.

GET /api/poll[?request_id=<id>]
    Alternative to the callback above, for a client with no way to accept
    an incoming POST - a browser page, for instance. Reports on the single
    most recently accepted request; there is no history of older ones:
        -> 200 {"pending": true, "request_id": "<id>"}   still running
        -> 200 <same body POST /api/response above would have carried>
        -> 404 nothing accepted yet, or request_id doesn't match the most
           recently accepted one (e.g. a page reload polling for a request
           from before it)

    request_id is optional here the same way it is on the GET endpoints
    above, but omitting it only makes sense with a single poller: with it,
    a mismatch is how a client notices it's looking at the wrong request
    instead of silently showing someone else's answer. Polling and the
    callback above are not exclusive - a request can ask for both with
    &callback. Neither happens on its own: polling means calling this
    endpoint yourself, and the callback POST requires &callback on the
    original request.

Every endpoint below follows this same accept-then-callback shape; see
handle_current for the pattern to copy when adding another.

See ai_client/simple_client.py for a minimal client exercising both sides.

Also serves the touchscreen client (ai_client/web/) as static files from "/",
same origin as the API above - the browser can reach both without CORS.

Install:
    pip install starlette uvicorn requests

Run:
    python ai_agent_server.py
"""

import asyncio
import contextlib
import logging
import uuid
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
# Where rendered graphs are served from, mounted on config.graph_output_dir.
GRAPH_URL_PREFIX = "/graphs"
# The touchscreen client's static files (ai_client/web/), served from "/" -
# same origin as the API above, so the browser needs no CORS configuration
# to reach it.
CLIENT_WEB_DIR = Path(__file__).parent / "ai_client" / "web"

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
    work the model call is just going to fail anyway. Hits /v1/models
    rather than an Ollama-native path, since OllamaClient now talks
    OpenAI-compatible /v1/chat/completions and may be pointed at a
    llama.cpp server instead of Ollama."""
    try:
        response = requests.get(f"{config.ollama_url}/v1/models", timeout=timeout)
        return response.ok
    except requests.RequestException:
        return False


def _track_pending(app: Starlette, request_id: str | None) -> str:
    """Record a newly accepted request as the one /api/poll will report on,
    clearing any previous result. Returns the id to poll with - the
    caller's own request_id if it gave one, otherwise a server-generated
    one (moot, since nothing that skipped request_id can ask for it back).
    """
    tracked_id = request_id or uuid.uuid4().hex
    app.state.poll_request_id = tracked_id
    app.state.poll_result = None
    return tracked_id


def _store_result(app: Starlette, tracked_id: str, payload: dict) -> None:
    """Make `payload` - the same dict about to be POSTed as the callback -
    available from /api/poll, unless a newer request has since been
    accepted and made this one stale."""
    if app.state.poll_request_id == tracked_id:
        app.state.poll_result = {**payload, "request_id": tracked_id}


def _audio_url(config: Config, path: Path, base_url: str) -> str:
    """Absolute link to a synthesized WAV.

    Built from the address the client actually used to reach us, so the
    link works without configuring anything - config.ai_agent_host is a
    bind address ("0.0.0.0") and cannot be turned into a fetchable URL.
    That address comes from the request's Host header, so set
    config.ai_agent_public_url to pin it wherever the header can't be
    trusted or doesn't match how clients reach the server, e.g. behind a
    proxy terminating on a different name.
    """
    base = (config.ai_agent_public_url or base_url).rstrip("/")
    return f"{base}{AUDIO_URL_PREFIX}/{quote(path.name)}"


def _graph_urls(config: Config, paths: dict[str, Path], base_url: str) -> dict[str, str]:
    """One absolute link per metric, same base-URL logic as _audio_url."""
    base = (config.ai_agent_public_url or base_url).rstrip("/")
    return {
        metric: f"{base}{GRAPH_URL_PREFIX}/{quote(path.name)}"
        for metric, path in paths.items()
    }


async def _add_audio(
    config: Config, agent: AiAgent, payload: dict, text: str, voice: str | None, base_url: str
) -> None:
    """Synthesize `text` and add its audio_url to `payload` in place. On
    failure, adds audio_error instead - the text answer has already cost a
    model call, so a synthesis failure must not lose it."""
    try:
        path = await asyncio.to_thread(agent.say, text, voice=voice)
        payload["audio_url"] = _audio_url(config, path, base_url)
    except Exception as e:
        logger.exception("Speech synthesis failed")
        payload["audio_error"] = str(e)


def _log_request(request: Request) -> None:
    """One INFO line per incoming GET, every query param included verbatim -
    uvicorn's own access log has the raw query string too, but interleaved
    with unrelated traffic and not worth eyeballing to answer "did this
    request actually ask for a graph/voice/callback". Cheap enough to log
    unconditionally."""
    logger.info("%s %s", request.url.path, dict(request.query_params))


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
        _log_request(request)
        request_id = request.query_params.get("request_id")
        # "voice" is a bare flag, so its value is the empty string - falsy,
        # which is why presence is tested by membership and not .get().
        # "voice=leo" names a voice; plain "voice" leaves it to the agent.
        wants_audio = "voice" in request.query_params
        voice = request.query_params.get("voice") or None
        wants_callback = "callback" in request.query_params
        try:
            reachable = await asyncio.to_thread(_ollama_reachable, config)
        except Exception as e:
            logger.exception("Could not check Ollama availability")
            return JSONResponse({"error": str(e)}, status_code=500)
        if not reachable:
            return JSONResponse({"error": "model not available"}, status_code=503)

        tracked_id = _track_pending(request.app, request_id)
        _spawn(
            _run_current(
                config,
                request.app,
                tracked_id,
                request_id,
                wants_audio,
                voice,
                wants_callback,
                # Captured here because the callback is built long after the
                # request object is gone.
                str(request.base_url),
            )
        )
        body = {"accepted": True}
        if request_id:
            body["request_id"] = request_id
        return JSONResponse(body, status_code=200)

    async def _run_current(
        config: Config,
        app: Starlette,
        tracked_id: str,
        request_id: str | None,
        wants_audio: bool,
        voice: str | None,
        wants_callback: bool,
        base_url: str,
    ) -> None:
        agent = app.state.agent
        try:
            text = await asyncio.to_thread(agent.summarize_current)
            payload = {"text": text, "audio_url": None}
        except Exception as e:
            logger.exception("summarize_current failed")
            payload = {"error": str(e)}
        else:
            if wants_audio:
                await _add_audio(config, agent, payload, text, voice, base_url)
        if request_id:
            payload["request_id"] = request_id
        _store_result(app, tracked_id, payload)
        if wants_callback:
            await _post_callback(config, payload)

    async def handle_current_battery(request: Request) -> Response:
        _log_request(request)
        request_id = request.query_params.get("request_id")
        # Same reasoning as handle_current above.
        wants_audio = "voice" in request.query_params
        voice = request.query_params.get("voice") or None
        wants_callback = "callback" in request.query_params
        try:
            reachable = await asyncio.to_thread(_ollama_reachable, config)
        except Exception as e:
            logger.exception("Could not check Ollama availability")
            return JSONResponse({"error": str(e)}, status_code=500)
        if not reachable:
            return JSONResponse({"error": "model not available"}, status_code=503)

        tracked_id = _track_pending(request.app, request_id)
        _spawn(
            _run_current_battery(
                config,
                request.app,
                tracked_id,
                request_id,
                wants_audio,
                voice,
                wants_callback,
                # Captured here because the callback is built long after the
                # request object is gone.
                str(request.base_url),
            )
        )
        body = {"accepted": True}
        if request_id:
            body["request_id"] = request_id
        return JSONResponse(body, status_code=200)

    async def _run_current_battery(
        config: Config,
        app: Starlette,
        tracked_id: str,
        request_id: str | None,
        wants_audio: bool,
        voice: str | None,
        wants_callback: bool,
        base_url: str,
    ) -> None:
        agent = app.state.agent
        try:
            text = await asyncio.to_thread(agent.summarize_current_battery)
            payload = {"text": text, "audio_url": None}
        except Exception as e:
            logger.exception("summarize_current_battery failed")
            payload = {"error": str(e)}
        else:
            if wants_audio:
                await _add_audio(config, agent, payload, text, voice, base_url)
        if request_id:
            payload["request_id"] = request_id
        _store_result(app, tracked_id, payload)
        if wants_callback:
            await _post_callback(config, payload)

    async def handle_outside_today(request: Request) -> Response:
        _log_request(request)
        request_id = request.query_params.get("request_id")
        # Same reasoning as handle_current above.
        wants_audio = "voice" in request.query_params
        voice = request.query_params.get("voice") or None
        wants_graph = "graph" in request.query_params
        wants_callback = "callback" in request.query_params
        try:
            reachable = await asyncio.to_thread(_ollama_reachable, config)
        except Exception as e:
            logger.exception("Could not check Ollama availability")
            return JSONResponse({"error": str(e)}, status_code=500)
        if not reachable:
            return JSONResponse({"error": "model not available"}, status_code=503)

        tracked_id = _track_pending(request.app, request_id)
        _spawn(
            _run_outside_today(
                config,
                request.app,
                tracked_id,
                request_id,
                wants_audio,
                voice,
                wants_graph,
                wants_callback,
                # Captured here because the callback is built long after the
                # request object is gone.
                str(request.base_url),
            )
        )
        body = {"accepted": True}
        if request_id:
            body["request_id"] = request_id
        return JSONResponse(body, status_code=200)

    async def _run_outside_today(
        config: Config,
        app: Starlette,
        tracked_id: str,
        request_id: str | None,
        wants_audio: bool,
        voice: str | None,
        wants_graph: bool,
        wants_callback: bool,
        base_url: str,
    ) -> None:
        agent = app.state.agent
        try:
            text, graphs = await asyncio.to_thread(
                agent.summarize_outside_for_today, wants_graph
            )
            payload = {
                "text": text,
                "audio_url": None,
                "graphs": _graph_urls(config, graphs, base_url) if wants_graph else None,
            }
        except Exception as e:
            logger.exception("summarize_outside_for_today failed")
            payload = {"error": str(e)}
        else:
            if wants_audio:
                await _add_audio(config, agent, payload, text, voice, base_url)
        if request_id:
            payload["request_id"] = request_id
        _store_result(app, tracked_id, payload)
        if wants_callback:
            await _post_callback(config, payload)

    async def handle_outside_last_hours(request: Request) -> Response:
        _log_request(request)
        request_id = request.query_params.get("request_id")
        wants_graph = "graph" in request.query_params
        wants_callback = "callback" in request.query_params
        # hours is optional and defaults to 24. Reject a non-integer or
        # non-positive value up front rather than silently summarizing the
        # wrong span - this is the only endpoint that can 400.
        hours_param = request.query_params.get("hours")
        if hours_param is None:
            hours = 24
        else:
            try:
                hours = int(hours_param)
            except ValueError:
                return JSONResponse({"error": "hours must be an integer"}, status_code=400)
            if hours <= 0:
                return JSONResponse({"error": "hours must be positive"}, status_code=400)
        try:
            reachable = await asyncio.to_thread(_ollama_reachable, config)
        except Exception as e:
            logger.exception("Could not check Ollama availability")
            return JSONResponse({"error": str(e)}, status_code=500)
        if not reachable:
            return JSONResponse({"error": "model not available"}, status_code=503)

        tracked_id = _track_pending(request.app, request_id)
        _spawn(
            _run_outside_last_hours(
                config,
                request.app,
                tracked_id,
                request_id,
                hours,
                wants_graph,
                wants_callback,
                # Captured here because the callback is built long after the
                # request object is gone.
                str(request.base_url),
            )
        )
        body = {"accepted": True}
        if request_id:
            body["request_id"] = request_id
        return JSONResponse(body, status_code=200)

    async def _run_outside_last_hours(
        config: Config,
        app: Starlette,
        tracked_id: str,
        request_id: str | None,
        hours: int,
        wants_graph: bool,
        wants_callback: bool,
        base_url: str,
    ) -> None:
        agent = app.state.agent
        try:
            text, graphs = await asyncio.to_thread(
                agent.summarize_history_outside_last_hours, hours, None, wants_graph
            )
            payload = {
                "text": text,
                "audio_url": None,
                "graphs": _graph_urls(config, graphs, base_url) if wants_graph else None,
            }
        except Exception as e:
            logger.exception("summarize_history_outside_last_hours failed")
            payload = {"error": str(e)}
        if request_id:
            payload["request_id"] = request_id
        _store_result(app, tracked_id, payload)
        if wants_callback:
            await _post_callback(config, payload)

    async def handle_poll(request: Request) -> Response:
        tracked_id = request.app.state.poll_request_id
        if tracked_id is None:
            return JSONResponse({"error": "no request has been accepted yet"}, status_code=404)

        request_id = request.query_params.get("request_id")
        if request_id and request_id != tracked_id:
            return JSONResponse({"error": "unknown or stale request_id"}, status_code=404)

        result = request.app.state.poll_result
        if result is None:
            return JSONResponse({"pending": True, "request_id": tracked_id}, status_code=200)
        return JSONResponse(result, status_code=200)

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette):
        app.state.agent = AiAgent(config)
        app.state.poll_request_id = None
        app.state.poll_result = None
        try:
            yield
        finally:
            app.state.agent.close()

    # StaticFiles refuses to mount a directory that doesn't exist yet, and
    # routes are built before the agent (which would create it) exists.
    Path(config.voice_output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.graph_output_dir).mkdir(parents=True, exist_ok=True)

    return Starlette(
        routes=[
            Route("/api/current", handle_current, methods=["GET"]),
            Route("/api/current_battery", handle_current_battery, methods=["GET"]),
            Route("/api/outside_today", handle_outside_today, methods=["GET"]),
            Route("/api/outside_last_hours", handle_outside_last_hours, methods=["GET"]),
            Route("/api/poll", handle_poll, methods=["GET"]),
            Mount(
                AUDIO_URL_PREFIX,
                StaticFiles(directory=config.voice_output_dir),
                name="audio",
            ),
            Mount(
                GRAPH_URL_PREFIX,
                StaticFiles(directory=config.graph_output_dir),
                name="graphs",
            ),
            # Catch-all for everything not matched above - must stay last,
            # or it would swallow the /api/* routes and the mounts before
            # they ever get a chance to match.
            Mount("/", StaticFiles(directory=CLIENT_WEB_DIR, html=True), name="client"),
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
