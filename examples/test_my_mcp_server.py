"""
Template for building your own MCP server in Python.

Install:
    pip install mcp

Run:
    python my_mcp_server.py

This uses the low-level `mcp.server.Server` class directly (not the FastMCP
convenience wrapper) so you can see exactly what's happening - request
handlers, tool schemas, and the stdio transport loop. Once you're comfortable,
FastMCP (`from mcp.server.fastmcp import FastMCP`) gives you the same result
with decorators and less code - worth switching to once you understand this.
"""

import asyncio
import logging
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("my-mcp-server")


class MyMCPServer:
    """
    Base template for an MCP server.

    Subclass this (or just edit it directly) and:
      1. Add tool definitions in `_build_tool_list`
      2. Add matching logic in `_dispatch_tool_call`
    Each tool needs: a name, description, and a JSON Schema for its inputs.
    """

    def __init__(self, name: str = "my-mcp-server"):
        self.name = name
        self.server = Server(name)
        self._register_handlers()

    # ---- Wiring: connects MCP protocol events to our methods ----
    def _register_handlers(self):
        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            return self._build_tool_list()

        @self.server.call_tool()
        async def call_tool(tool_name: str, arguments: dict[str, Any]) -> list[TextContent]:
            return await self._dispatch_tool_call(tool_name, arguments)

    # ---- Define what tools this server offers ----
    def _build_tool_list(self) -> list[Tool]:
        return [
            Tool(
                name="say_hello",
                description="Returns a greeting for the given name.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Name to greet"},
                    },
                    "required": ["name"],
                },
            ),
            Tool(
                name="add_numbers",
                description="Adds two numbers together.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "a": {"type": "number"},
                        "b": {"type": "number"},
                    },
                    "required": ["a", "b"],
                },
            ),
            # Add more Tool(...) entries here for each capability you expose.
        ]

    # ---- Route each tool call to actual logic ----
    async def _dispatch_tool_call(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> list[TextContent]:
        try:
            if tool_name == "say_hello":
                result = self._say_hello(arguments["name"])
            elif tool_name == "add_numbers":
                result = self._add_numbers(arguments["a"], arguments["b"])
            else:
                raise ValueError(f"Unknown tool: {tool_name}")

            return [TextContent(type="text", text=str(result))]

        except Exception as e:
            logger.exception("Tool call failed: %s", tool_name)
            return [TextContent(type="text", text=f"Error: {e}")]

    # ---- Actual tool implementations ----
    def _say_hello(self, name: str) -> str:
        return f"Hello, {name}!"

    def _add_numbers(self, a: float, b: float) -> float:
        return a + b

    # ---- Entry point: run over stdio ----
    async def run(self):
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options(),
            )


if __name__ == "__main__":
    server = MyMCPServer(name="my-mcp-server")
    asyncio.run(server.run())