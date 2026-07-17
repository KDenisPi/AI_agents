import asyncio
import aiohttp

import logging
from typing import Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("my-http")

class MyHttpClient:
    """HTTP client"""

    def __init__(self, url : str = None, json : bool = False):
        self._url = url
        self._data = None
        self._json = json
        self._method = "GET"

    @property
    def url(self) -> str:
        return self._url
    
    @property
    def json(self) -> bool:
        return self._json
    
    @json.setter
    def json(self, json_frm : bool) -> None:
        self._json = json_frm

    @property
    def method(self) -> str:
        return self._method
    
    @method.setter
    def method(self, http_method : str) -> None:
        self._method = http_method

    async def _fetch_url(self, session) -> Any:
        try:
            async with session.get(self.url) if self._method == "GET" \
                else session.post(self.url, data=self._data) if not self._json \
                else session.post(self.url, json=self._data)  as response:
                    # Await the response body text or .json() 
                    data = await response.text()
                    print(f"Fetched {self.url} with status: {response.status}")
                    return data
        except Exception as e:
            print(f"Error fetching {self.url}: {e}")
            return None
    
    async def _req(self) -> Any:
        async with aiohttp.ClientSession() as session:
            return await self._fetch_url(session)

    async def request(self) -> Any:
        """Async request - await this from a running event loop."""
        return await self._req()

    def request_sync(self) -> Any:
        """Blocking request for use outside an event loop (CLI, scripts)."""
        return asyncio.run(self._req())