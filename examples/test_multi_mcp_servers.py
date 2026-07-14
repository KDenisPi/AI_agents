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
from starlette.routing import Mount, Route


def make_mcp_asgi_app(server: Server, base_path: str) -> tuple[Route, Mount]:
    """Wrap an MCP Server into Starlette routes mounted under base_path."""
    sse = SseServerTransport(f"{base_path}/messages")

    async def handle_sse(request):
        async with sse.connect_sse(
            request.scope, request.receive, request._send
        ) as (read_stream, write_stream):
            await server.run(
                read_stream, write_stream, server.create_initialization_options()
            )

    sse_route = Route(f"{base_path}/sse", endpoint=handle_sse)
    messages_mount = Mount(f"{base_path}/messages", app=sse.handle_post_message)
    return sse_route, messages_mount


# ---- Server 1: filesystem-style tools ----
filesystem_server = Server("filesystem-server")

@filesystem_server.list_tools()
async def fs_list_tools() -> list[Tool]:
    return [Tool(
        name="list_files",
        description="Lists files in a fixed demo directory.",
        inputSchema={"type": "object", "properties": {}},
    )]

@filesystem_server.call_tool()
async def fs_call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    return [TextContent(type="text", text="file1.txt, file2.txt")]


# ---- Server 2: Ollama-related tools ----
ollama_server = Server("ollama-server")

@ollama_server.list_tools()
async def ollama_list_tools() -> list[Tool]:
    return [Tool(
        name="list_models",
        description="Lists models loaded in the local Ollama instance.",
        inputSchema={"type": "object", "properties": {}},
    )]

@ollama_server.call_tool()
async def ollama_call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    return [TextContent(type="text", text="llama3.2:1b, qwen2.5:0.5b")]


# ---- Combine everything into one Starlette app on one port ----
routes = []
for server, path in [(filesystem_server, "/filesystem"), (ollama_server, "/ollama")]:
    sse_route, messages_mount = make_mcp_asgi_app(server, path)
    routes.append(sse_route)
    routes.append(messages_mount)

app = Starlette(routes=routes)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)