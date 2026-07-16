import asyncio
import logging
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("my-mcp-server")

class McpWeatherServer:
    """
    Base template for an MCP server.
    """
    def __init__(self, name: str = "mcp-weather-server"):
        self.name = name
        self.server = Server(name)
        self._register_handlers()
        self._base_path = "/weather"

    @property
    def mcp_server(self) -> Server:
        return self.server

    @property
    def base_path(self) -> str:
        return self._base_path

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
                name="weather_info",
                description="Returns weather information from local Weather station.",
                inputSchema={
                    "type": "object",
                },
            ),
        ]

    async def _dispatch_tool_call(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> list[TextContent]:
        try:
            if tool_name == "weather_info":
                result = self._get_weather_info()
            else:
                raise ValueError(f"Unknown tool: {tool_name}")

            return [TextContent(type="text", text=str(result))]

        except Exception as e:
            logger.exception("Tool call failed: %s", tool_name)
            return [TextContent(type="text", text=f"Error: {e}")]


    def _get_weather_info(self) -> str:
        # Placeholder for actual weather info retrieval logic
        return "Weather info: Sunny, 25°C"
