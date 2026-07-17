"""
Persistent MCP client session - opened once, reused for many tool calls
within an agent's lifetime. Better fit than a per-call session when your
Ollama-driven agent will call tools repeatedly during one run.
"""

import asyncio
from contextlib import AsyncExitStack
from mcp import ClientSession
from mcp.client.sse import sse_client


class MCPClientPersistent:
    """
    Wraps one MCP server connection, keeping the session alive across
    multiple tool calls instead of re-handshaking every time.

    Usage:
        async with MCPClientPersistent("http://pi-host:8000/ollama/sse") as client:
            tools = await client.list_tools()
            result = await client.call_tool("list_models", {})
            ... call more tools as needed, session stays open ...
        # connect()/close() also work directly if you need to manage lifetime by hand
    """

    def __init__(self, server_url: str):
        self.server_url = server_url
        self._session_id: str | None = None
        self._session: ClientSession | None = None
        self._stack = AsyncExitStack()
        self._tools: list = []

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def is_session_id(self) -> bool:
        """Return True if the session is open and has a session ID."""
        return self._session_id is not None

    @property
    def is_session(self) -> bool:
        """Return True if the session is open and active."""
        return self._session is not None

    @property
    def tools(self) -> list:
        """Return the list of tools available in the session."""
        if len(self._tools) == 0:
            self.list_tools()
        return self._tools

    async def connect(self):
        read, write = await self._stack.enter_async_context(
            sse_client(self.server_url, on_session_created=self._on_session_created)
        )
        self._session = await self._stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()  # handshake happens once, here

    def _on_session_created(self, session_id: str):
        self._session_id = session_id  # server assigns this once the SSE endpoint event arrives

    async def list_tools(self):
        self._tools = await self._session.list_tools()
        return self._tools

    async def call_tool(self, name: str, arguments: dict):
        result = await self._session.call_tool(name, arguments=arguments)
        return "\n".join(b.text for b in result.content if hasattr(b, "text"))

    async def close(self):
        await self._stack.aclose()
        self._session = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *exc_info):
        await self.close()


async def demo():
    async with MCPClientPersistent("http://<pi-ip>:8000/ollama/sse") as client:
        print("Session ID:", client.session_id)

        tools = await client.list_tools()
        print("Tools:", [t.name for t in tools.tools])

        # Reuses the same handshake for multiple calls - no re-init cost
        print(await client.call_tool("list_models", {}))
        print(await client.call_tool("list_models", {}))


if __name__ == "__main__":
    asyncio.run(demo())