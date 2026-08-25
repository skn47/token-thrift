from __future__ import annotations

import time

import httpx

from tokenthrift.generation.answer import Answer


class OpenAICompatibleClient:
    """Generation client for any OpenAI-wire-compatible chat completions
    endpoint (OpenAI, Groq, OpenRouter, self-hosted/local servers, or any
    other custom base URL) — one client, driven purely by `base_url` and
    `api_key`, so committing to BYOK doesn't mean committing to one SDK per
    provider. The API key is only ever held in memory for the lifetime of
    this object — callers (the Streamlit UI) are responsible for sourcing
    it from transient session state and never persisting or logging it."""

    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    def generate(self, prompt: str, temperature: float = 0.0) -> Answer:
        start = time.perf_counter()
        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
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
        choice = payload["choices"][0]
        usage = payload.get("usage") or {}
        return Answer(
            text=choice["message"]["content"] or "",
            latency_seconds=latency,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            model=self.model,
            error=None,
        )
