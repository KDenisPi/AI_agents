"""
Hubitat Elevation as an MCP server - wraps HubitatClient's read-only Maker
API calls as MCP tools. Server-side counterpart to WeatherMcpSource.py's
weather station wrapper (see McpWeather.py, which this mirrors), hosted by
McpMultiHttpServer.py rather than run standalone.

Read-only by design: no send_command/setLabel/etc. here, same as
McpWeather.py only ever exposing the single read-only weather_info tool -
Hubitat's write endpoints (device control, hub variable writes, room/mode
changes) are out of scope for this MCP surface.

Builds its own HubitatClient instance from config, entirely separate from
the one HubitatSource.py/Collector.py/CollectorClickhouse.py construct for
the readings pipeline - the two share no state at runtime, so nothing here
touches that collection path.
"""

import asyncio
import json
import logging
from typing import Any

from mcp.server import Server
from mcp.types import Tool, TextContent

from Config import Config
from HubitatClient import HubitatClient

logger = logging.getLogger("mcp-hubitat")

_DEVICE_ID_SCHEMA = {
    "type": "object",
    "properties": {
        "device_id": {"type": "string", "description": "Hubitat device id"},
    },
    "required": ["device_id"],
}


class McpHubitatServer:
    """
    Base template for an MCP server.
    """
    def __init__(self, config: Config, name: str = "mcp-hubitat-server", dry_run: bool = False):
        self.name = name
        self.server = Server(name)
        self._register_handlers()
        self._base_path = "/hubitat"
        self._client = HubitatClient(
            config.hubitat_ip,
            app_id=config.hubitat_app_id,
            access_token=config.hubitat_token,
        )
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
                name="get_all_devices",
                description="List every device known to this Maker API instance (id, name, label, type).",
                inputSchema={"type": "object"},
            ),
            Tool(
                name="get_all_devices_full_info",
                description="Every device with full details: current attributes, capabilities, and commands.",
                inputSchema={"type": "object"},
            ),
            Tool(
                name="get_device_info",
                description="Full info for one device: current attributes, capabilities, and available commands.",
                inputSchema=_DEVICE_ID_SCHEMA,
            ),
            Tool(
                name="get_device_event_history",
                description="Recent event history for one device.",
                inputSchema=_DEVICE_ID_SCHEMA,
            ),
            Tool(
                name="get_device_commands",
                description="Commands one device supports.",
                inputSchema=_DEVICE_ID_SCHEMA,
            ),
            Tool(
                name="get_device_capabilities",
                description="Capabilities one device supports.",
                inputSchema=_DEVICE_ID_SCHEMA,
            ),
            Tool(
                name="get_device_attribute",
                description="Current value of one named attribute on one device (e.g. attribute='temperature' or 'switch').",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "device_id": {"type": "string", "description": "Hubitat device id"},
                        "attribute": {
                            "type": "string",
                            "description": "Attribute name, e.g. 'switch' or 'temperature'",
                        },
                    },
                    "required": ["device_id", "attribute"],
                },
            ),
            Tool(
                name="get_list_hub_variables",
                description="All hub variables defined on this hub.",
                inputSchema={"type": "object"},
            ),
        ]

    async def _dispatch_tool_call(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> list[TextContent]:
        try:
            result = await self._call_tool(tool_name, arguments)
            return [TextContent(type="text", text=json.dumps(result))]
        except Exception as e:
            logger.exception("Tool call failed: %s", tool_name)
            return [TextContent(type="text", text=f"Error: {e}")]

    async def _call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        if self.dry_run:
            return {}

        # HubitatClient uses blocking requests - keep it off the event loop,
        # same as HubitatSource.collect() does for the readings pipeline.
        if tool_name == "get_all_devices":
            return await asyncio.to_thread(self._client.list_devices)
        if tool_name == "get_all_devices_full_info":
            return await asyncio.to_thread(self._client.list_devices_all)
        if tool_name == "get_device_info":
            return await asyncio.to_thread(self._client.get_device, arguments["device_id"])
        if tool_name == "get_device_event_history":
            return await asyncio.to_thread(self._client.get_device_events, arguments["device_id"])
        if tool_name == "get_device_commands":
            return await asyncio.to_thread(self._client.get_device_commands, arguments["device_id"])
        if tool_name == "get_device_capabilities":
            return await asyncio.to_thread(self._client.get_device_capabilities, arguments["device_id"])
        if tool_name == "get_device_attribute":
            return await asyncio.to_thread(
                self._client.get_device_attribute, arguments["device_id"], arguments["attribute"]
            )
        if tool_name == "get_list_hub_variables":
            return await asyncio.to_thread(self._client.list_hub_variables)

        raise ValueError(f"Unknown tool: {tool_name}")
