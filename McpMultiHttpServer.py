"""
Run multiple MCP servers under a single process/port, distinguished by
URL path instead of separate ports. Lighter on RAM than N separate
uvicorn processes - useful on constrained hardware like a Pi.

Install:
    pip install mcp uvicorn starlette

Run:
    python multi_mcp_server.py
    -> everything on :8000, routed by path:
       http://host:8000/filesystem/sse
       http://host:8000/ollama/sse
"""

import contextlib
from typing import Any
import uvicorn
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.types import Tool, TextContent
from starlette.applications import Starlette
from starlette.responses import Response
from starlette.routing import Mount, Route

from Config import Config
from McpWeather import McpWeatherServer

def make_mcp_asgi_app(server: Server, base_path: str) -> tuple[Route, Mount]:
    """Wrap an MCP Server into Starlette routes mounted under base_path."""
    sse = SseServerTransport(f"{base_path}/messages/")

    async def handle_sse(request):
        async with sse.connect_sse(
            request.scope, request.receive, request._send
        ) as (read_stream, write_stream):
            await server.run(
                read_stream, write_stream, server.create_initialization_options()
            )
        return Response()

    sse_route = Route(f"{base_path}/sse", endpoint=handle_sse)
    messages_mount = Mount(f"{base_path}/messages", app=sse.handle_post_message)
    return sse_route, messages_mount

def make_streamable_http_mount(
    server: Server, base_path: str
) -> tuple[StreamableHTTPSessionManager, Mount]:
    """Stateless Streamable HTTP endpoint under base_path/mcp/.

    One plain POST per JSON-RPC request, JSON response back -
    no session id, no initialize handshake, curl-friendly.
    """
    manager = StreamableHTTPSessionManager(
        app=server,
        json_response=True,  # answer with JSON instead of an SSE stream
        stateless=True,      # no session tracking, each POST independent
    )
    return manager, Mount(f"{base_path}/mcp", app=manager.handle_request)

config = Config.from_env()
if config.log_file == Config.log_file:  # LOG_FILE not overridden via env
    config.log_file = "logs/mcp_server.log"
config.configure_logging()
servers = [McpWeatherServer(config)]

# ---- Combine everything into one Starlette app on one port ----
routes = []
managers = []
for server in servers:
    sse_route, messages_mount = make_mcp_asgi_app(server.server, server.base_path)
    routes.append(sse_route)
    routes.append(messages_mount)

    manager, mcp_mount = make_streamable_http_mount(server.server, server.base_path)
    managers.append(manager)
    routes.append(mcp_mount)

@contextlib.asynccontextmanager
async def lifespan(app: Starlette):
    # StreamableHTTPSessionManager needs its task group running for
    # the whole lifetime of the app.
    async with contextlib.AsyncExitStack() as stack:
        for manager in managers:
            await stack.enter_async_context(manager.run())
        yield

app = Starlette(routes=routes, lifespan=lifespan)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)