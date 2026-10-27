"""
AiAgent's summaries as MCP tools - server-side counterpart to
McpWeather.py/McpHubitat.py, hosted by McpMultiHttpServer.py rather than
run standalone.

Unlike those two, this does not talk to a data source directly - it is a
thin client of the already-running ai_agent_server.py (AiAgent itself
needs Ollama, ClickHouse, and - for voice - numpy/torch/snac, none of
which belong in the lightweight mcp-server container). Each tool call
replays ai_agent_server.py's accept-then-poll contract (see that module's
docstring) against config.ai_agent_internal_url: GET the endpoint, then
poll /api/poll with the same request_id until it stops being pending.

voice and graph are deliberately never requested here - a client calling
these tools (e.g. OpenClaw) has its own text-to-speech and rendering, and
synthesizing audio server-side only helps a caller that can fetch
/audio, which is not a safe assumption to make about every MCP client.
"""

import asyncio
import logging
import uuid
from typing import Any

import aiohttp
from mcp.server import Server
from mcp.types import Tool, TextContent, ToolAnnotations

from Config import Config

logger = logging.getLogger("mcp-ai-agent")

# Every tool here only reads a summary that ai_agent_server.py itself
# computes read-only (it never writes anything back to the DB), so all of
# them get the same read-only annotations - same as McpWeather.py/
# McpHubitat.py.
_READ_ONLY = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)

_HOURS_SCHEMA = {
    "type": "object",
    "properties": {
        "hours": {
            "type": "integer",
            "description": "How many hours of history to summarize (default 24).",
            "minimum": 1,
        },
    },
}

# How long to wait for an accepted request to finish before giving up -
# these are model calls (AiAgent.model_small), not database reads, so a
# single one can legitimately take tens of seconds.
_POLL_TIMEOUT_SECONDS = 60
_POLL_INTERVAL_SECONDS = 1.5


class AiAgentCallError(Exception):
    """Raised when ai_agent_server.py rejects, fails, or never finishes a
    request this server made on a tool caller's behalf."""


class McpAiAgentServer:
    """
    Base template for an MCP server.
    """
    def __init__(self, config: Config, name: str = "mcp-ai-agent-server", dry_run: bool = False):
        self.name = name
        self.server = Server(name)
        self._register_handlers()
        self._base_path = "/agent"
        self._api_base = config.ai_agent_internal_url.rstrip("/")
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
                name="current_status",
                description="Plain-language summary of the current readings across all sources.",
                inputSchema={"type": "object"},
                annotations=_READ_ONLY,
            ),
            Tool(
                name="current_battery",
                description="Plain-language summary of current battery levels.",
                inputSchema={"type": "object"},
                annotations=_READ_ONLY,
            ),
            Tool(
                name="outside_today",
                description="Summary of today's outside temperature and humidity.",
                inputSchema={"type": "object"},
                annotations=_READ_ONLY,
            ),
            Tool(
                name="outside_last_hours",
                description="Summary of outside temperature and humidity over the last N hours.",
                inputSchema=_HOURS_SCHEMA,
                annotations=_READ_ONLY,
            ),
            Tool(
                name="power_last_hours",
                description=(
                    "Historical power usage per device over time - typical and peak "
                    "wattage over the last N hours (default 24, i.e. roughly 'today'). "
                    "This is the only source with power usage over time / trends; a "
                    "device's current/instantaneous wattage from a smart-outlet "
                    "attribute is not a substitute for this and cannot show usage "
                    "'today' or 'over the last N hours' on its own."
                ),
                inputSchema=_HOURS_SCHEMA,
                annotations=_READ_ONLY,
            ),
        ]

    _ENDPOINTS = {
        "current_status": "/api/current",
        "current_battery": "/api/current_battery",
        "outside_today": "/api/outside_today",
        "outside_last_hours": "/api/outside_last_hours",
        "power_last_hours": "/api/power_last_hours",
    }

    async def _dispatch_tool_call(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> list[TextContent]:
        try:
            if self.dry_run:
                return [TextContent(type="text", text="{}")]
            endpoint = self._ENDPOINTS.get(tool_name)
            if endpoint is None:
                raise ValueError(f"Unknown tool: {tool_name}")
            text = await self._summarize(endpoint, arguments)
            return [TextContent(type="text", text=text)]
        except Exception as e:
            logger.exception("Tool call failed: %s", tool_name)
            return [TextContent(type="text", text=f"Error: {e}")]

    async def _summarize(self, endpoint: str, arguments: dict[str, Any]) -> str:
        """Replay ai_agent_server.py's accept-then-poll contract for one
        endpoint and return the resulting text, or raise AiAgentCallError."""
        request_id = uuid.uuid4().hex
        params = {"request_id": request_id}
        if "hours" in arguments and arguments["hours"] is not None:
            params["hours"] = str(arguments["hours"])

        async with aiohttp.ClientSession(base_url=self._api_base) as session:
            async with session.get(endpoint, params=params) as response:
                body = await response.json()
                if response.status != 200:
                    raise AiAgentCallError(body.get("error", f"HTTP {response.status}"))

            deadline = asyncio.get_event_loop().time() + _POLL_TIMEOUT_SECONDS
            while True:
                async with session.get(
                    "/api/poll", params={"request_id": request_id}
                ) as response:
                    if response.status == 404:
                        raise AiAgentCallError(
                            "lost the request while polling - "
                            "another request to ai-server overtook it"
                        )
                    result = await response.json()
                if not result.get("pending", False):
                    break
                if asyncio.get_event_loop().time() >= deadline:
                    raise AiAgentCallError(
                        f"timed out after {_POLL_TIMEOUT_SECONDS}s waiting for an answer"
                    )
                await asyncio.sleep(_POLL_INTERVAL_SECONDS)

        if "error" in result:
            raise AiAgentCallError(result["error"])
        return result["text"]
