import asyncio
import logging
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from Config import Config
from HttpClient import MyHttpClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp-weather")

class McpWeatherServer:
    """
    Base template for an MCP server.
    """
    def __init__(self, config: Config, name: str = "mcp-weather-server", dry_run: bool = False):
        self.name = name
        self.server = Server(name)
        self._register_handlers()
        self._base_path = "/weather"
        self._http_client = MyHttpClient(config.weather_station_url)
        self._dry_run = dry_run

    @property
    def mcp_server(self) -> Server:
        return self.server

    @property
    def base_path(self) -> str:
        return self._base_path

    @property
    def dry_run(self) -> bool:
        return self._dry_run

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
                result = await self._get_weather_info()
            else:
                raise ValueError(f"Unknown tool: {tool_name}")

            return [TextContent(type="text", text=str(result))]

        except Exception as e:
            logger.exception("Tool call failed: %s", tool_name)
            return [TextContent(type="text", text=f"Error: {e}")]


    async def _get_weather_info(self) -> str:
        if self.dry_run:
            return "{}"
        result = await self._http_client.request()
        return result
