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

from typing import Any
import uvicorn
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import Tool, TextContent
from starlette.applications import Starlette
from starlette.responses import Response
from starlette.routing import Mount, Route

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

servers = [McpWeatherServer()]

# ---- Combine everything into one Starlette app on one port ----
routes = []
for server in servers:
    sse_route, messages_mount = make_mcp_asgi_app(server.server, server.base_path)
    routes.append(sse_route)
    routes.append(messages_mount)

app = Starlette(routes=routes)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)