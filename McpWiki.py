"""
Wikidata questions as MCP tools - server-side counterpart to McpAiAgent.py,
hosted by McpMultiHttpServer.py at /wiki rather than run standalone.

Like McpAiAgent.py, this answers nothing itself. It is a thin client of the
wiki repo's agent/wiki_service.py (../wiki), reached at config.wiki_url: the
question pipeline behind it needs duckdb, a 2.1 GB database and Ollama,
none of which belong in the lightweight mcp-server container. The wiki repo
imports OllamaClient from this one, so the two only meet over HTTP and the
dependency stays one-way.

Four tools, one per step of that pipeline, each a single POST:

    wiki_sentence  /sentence  the question with spelling and grammar fixed
    wiki_intent    /intent    the question read into a structured intent,
                              nothing looked up
    wiki_query     /query     a structured intent - written by the calling
                              client's own model from the tool's schema - in,
                              the answer and its rows out
    wiki_ask       /ask       the whole pipeline: question in, answer out

The tool list itself - names, descriptions, input schemas - comes from the
service's GET /tools rather than being written out here, so the intent
schema has a single owner: the wiki repo, next to the composer that reads
it. It is cached for _TOOLS_TTL_SECONDS, and only once fetched: a service
that is down when a client lists tools costs that listing the wiki tools,
not every listing until mcp-server restarts.

Every result is the service's JSON, passed through as compact text - the
calling model reads "answer" for the sentence and "rows" for the facts.
"""

import json
import logging
import time
from typing import Any

import aiohttp
from mcp.server import Server
from mcp.types import Tool, TextContent

from Config import Config

logger = logging.getLogger("mcp-wiki")

_ROUTES = {
    "wiki_sentence": "/sentence",
    "wiki_intent": "/intent",
    "wiki_query": "/query",
    "wiki_ask": "/ask",
}

# A question is two model calls and a DuckDB query - typically 5-15s. The
# service caps a query at 120s and queues a request for up to 120s more when
# its slots are busy, so anything past this is a service that is not coming
# back with an answer.
_CALL_TIMEOUT_SECONDS = 300
_TOOLS_TIMEOUT_SECONDS = 10
_TOOLS_TTL_SECONDS = 300


class WikiCallError(Exception):
    """Raised when wiki_service.py rejects or fails a request this server
    made on a tool caller's behalf."""


class McpWikiServer:
    """
    Base template for an MCP server.
    """
    def __init__(self, config: Config, name: str = "mcp-wiki-server", dry_run: bool = False):
        self.name = name
        self.server = Server(name)
        self._register_handlers()
        self._base_path = "/wiki"
        self._api_base = config.wiki_url.rstrip("/")
        self._dry_run = dry_run
        self._tools: list[Tool] = []
        self._tools_fetched = 0.0

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
            return await self._build_tool_list()

        @self.server.call_tool()
        async def call_tool(tool_name: str, arguments: dict[str, Any]) -> list[TextContent]:
            return await self._dispatch_tool_call(tool_name, arguments)

    # ---- Define what tools this server offers ----
    async def _build_tool_list(self) -> list[Tool]:
        """The service's own tool definitions, cached once fetched."""
        if self._tools and time.monotonic() - self._tools_fetched < _TOOLS_TTL_SECONDS:
            return self._tools
        try:
            timeout = aiohttp.ClientTimeout(total=_TOOLS_TIMEOUT_SECONDS)
            async with aiohttp.ClientSession(base_url=self._api_base, timeout=timeout) as session:
                async with session.get("/tools") as response:
                    body = await response.json()
                    if response.status != 200:
                        raise WikiCallError(body.get("error", f"HTTP {response.status}"))
            tools = [Tool(**tool) for tool in body["tools"] if tool.get("name") in _ROUTES]
        except Exception as e:
            # Stale tools beat none: a service restarting under a cached
            # list still answers the calls once it is back.
            logger.warning("wiki service %s: no tool list (%s)", self._api_base, e)
            return self._tools
        self._tools, self._tools_fetched = tools, time.monotonic()
        return tools

    async def _dispatch_tool_call(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> list[TextContent]:
        try:
            if self.dry_run:
                return [TextContent(type="text", text="{}")]
            route = _ROUTES.get(tool_name)
            if route is None:
                raise ValueError(f"Unknown tool: {tool_name}")
            text = await self._post(route, arguments or {})
            return [TextContent(type="text", text=text)]
        except Exception as e:
            logger.exception("Tool call failed: %s", tool_name)
            return [TextContent(type="text", text=f"Error: {e}")]

    async def _post(self, route: str, arguments: dict[str, Any]) -> str:
        """POST the tool's arguments to the service and return its JSON answer
        as compact text, or raise WikiCallError with the service's reason."""
        timeout = aiohttp.ClientTimeout(total=_CALL_TIMEOUT_SECONDS)
        async with aiohttp.ClientSession(base_url=self._api_base, timeout=timeout) as session:
            async with session.post(route, json=arguments) as response:
                body = await response.json()
                if response.status != 200:
                    raise WikiCallError(body.get("error", f"HTTP {response.status}"))
        return json.dumps(body, ensure_ascii=False, separators=(",", ":"))
