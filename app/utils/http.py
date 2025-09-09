import asyncio
from typing import Any, Dict, Optional

import httpx


class AsyncHttpClient:
    def __init__(self, base_url: str, timeout_seconds: float = 30.0):
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout_seconds)

    async def post_json(self, path: str, json: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        resp = await self._client.post(path, json=json, headers=headers)
        resp.raise_for_status()
        return resp.json()

    async def get_json(self, path: str, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        resp = await self._client.get(path, headers=headers)
        resp.raise_for_status()
        return resp.json()

    async def aclose(self) -> None:
        await self._client.aclose()


