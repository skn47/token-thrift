from __future__ import annotations

import time

import httpx

from tokenthrift.generation.answer import Answer

ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 1024


class AnthropicClient:
    """Generation client for Anthropic's native Messages API — a genuinely
    different wire format from the OpenAI-compatible one (x-api-key auth,
    an anthropic-version header, and a content-block response shape), so
    it gets its own client rather than being forced into
    OpenAICompatibleClient. The API key is only ever held in memory for
    the lifetime of this object, matching every other client's contract."""

    def __init__(
        self, api_key: str, model: str,
        base_url: str = "https://api.anthropic.com",
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens

    def generate(self, prompt: str, temperature: float = 0.0) -> Answer:
        start = time.perf_counter()
        try:
            response = httpx.post(
                f"{self.base_url}/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": ANTHROPIC_VERSION,
                },
                json={
                    "model": self.model,
                    "max_tokens": self.max_tokens,
                    "temperature": temperature,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=60.0,
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPError as e:
            latency = time.perf_counter() - start
            return Answer(
                text="", latency_seconds=latency, input_tokens=None,
                output_tokens=None, model=self.model, error=str(e),
            )

        latency = time.perf_counter() - start
        blocks = payload.get("content") or []
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        usage = payload.get("usage") or {}
        return Answer(
            text=text,
            latency_seconds=latency,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            model=self.model,
            error=None,
        )
