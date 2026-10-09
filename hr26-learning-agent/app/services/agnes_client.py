"""
Agnes AI API Client
- Keeps API key server-side
- Rate-limit aware (free plan: 10 RPM text)
- Supports chat/completions + tool calling
- Simple exponential backoff
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, AsyncGenerator
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from app.config import get_settings


class RateLimiter:
    """Simple token-bucket style limiter for RPM."""
    def __init__(self, rpm: int = 10):
        self.rpm = rpm
        self.interval = 60.0 / rpm
        self.last_call = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            wait = self.interval - (now - self.last_call)
            if wait > 0:
                await asyncio.sleep(wait)
            self.last_call = time.monotonic()


class AgnesClient:
    def __init__(self):
        self.settings = get_settings()
        self.base_url = self.settings.agnes_base_url.rstrip("/")
        self.model = self.settings.agnes_model
        self.rate_limiter = RateLimiter(rpm=self.settings.rate_limit_rpm)
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.settings.agnes_api_key}",
                    "Content-Type": "application/json",
                },
                timeout=120.0,
            )
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TimeoutException)),
    )
    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str | Dict] = None,
        temperature: float = 0.4,
        max_tokens: int = 4096,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """
        Call /v1/chat/completions (OpenAI-compatible).
        Also works with Anthropic-compatible /v1/messages if needed later.
        """
        await self.rate_limiter.acquire()
        client = await self._get_client()

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if tools:
            payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice

        resp = await client.post("/chat/completions", json=payload)

        if resp.status_code == 429:
            # Rate limited  force longer wait
            await asyncio.sleep(6)
            resp.raise_for_status()

        resp.raise_for_status()
        return resp.json()

    async def chat_completion_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
        temperature: float = 0.4,
        max_tokens: int = 4096,
    ) -> AsyncGenerator[str, None]:
        """Streaming helper (SSE style)."""
        await self.rate_limiter.acquire()
        client = await self._get_client()

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools

        async with client.stream("POST", "/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    yield data

    async def generate_image(
        self,
        prompt: str,
        size: str = "1024x1024",
        n: int = 1,
    ) -> Dict[str, Any]:
        """Optional image generation via Agnes Image 2.5 Flash."""
        await self.rate_limiter.acquire()
        client = await self._get_client()

        payload = {
            "model": self.settings.agnes_image_model,
            "prompt": prompt,
            "n": n,
            "size": size,
        }
        resp = await client.post("/images/generations", json=payload)
        resp.raise_for_status()
        return resp.json()


# Singleton helper
_agnes_client: Optional[AgnesClient] = None


def get_agnes_client() -> AgnesClient:
    global _agnes_client
    if _agnes_client is None:
        _agnes_client = AgnesClient()
    return _agnes_client
