from __future__ import annotations

import time
from dataclasses import dataclass

import httpx

from tokenthrift.generation.anthropic_client import (
    ANTHROPIC_VERSION,
    DEFAULT_MAX_TOKENS,
)
from tokenthrift.generation.providers import WIRE_ANTHROPIC, WIRE_OPENAI
from tokenthrift.proxy.server import TOKENS_PRUNED_HEADER

HEALTH_TIMEOUT_SECONDS = 5.0
CALL_TIMEOUT_SECONDS = 60.0


@dataclass(frozen=True)
class ProxyHealth:
    reachable: bool
    upstream_configured: bool | None
    policy_preset: str | None
    error: str | None = None


def check_health(proxy_base_url: str) -> ProxyHealth:
    """Read-only status check — never starts, stops, or reconfigures the
    proxy, only reports what a running one says about itself."""
    try:
        response = httpx.get(
            f"{proxy_base_url.rstrip('/')}/healthz", timeout=HEALTH_TIMEOUT_SECONDS)
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPError as e:
        return ProxyHealth(
            reachable=False, upstream_configured=None, policy_preset=None, error=str(e))
    return ProxyHealth(
        reachable=True,
        upstream_configured=payload.get("upstream_configured"),
        policy_preset=payload.get("policy_preset"),
    )


@dataclass(frozen=True)
class ProxyCallResult:
    text: str
    tokens_pruned: int | None
    latency_seconds: float
    error: str | None = None


def _endpoint_base_url(proxy_base_url: str, wire_format: str) -> str:
    """The proxy exposes /v1/chat/completions and /v1/messages at its root
    — mirrors how OpenAICompatibleClient/AnthropicClient derive their
    request URL from base_url, so the same proxy_base_url the user types
    (its bare root, e.g. http://localhost:8787) works for either wire
    format without the user needing to know the suffix convention."""
    root = proxy_base_url.rstrip("/")
    return f"{root}/v1" if wire_format == WIRE_OPENAI else root


def call_through_proxy(
    proxy_base_url: str, wire_format: str, api_key: str, model: str, prompt: str,
) -> ProxyCallResult:
    """Sends one real request through a running TokenThrift proxy — the
    same request an agent pointed at this proxy would send — and reports
    what it actually pruned via the X-TokenThrift-Tokens-Pruned response
    header. Deliberately separate from generation/*_client.py: those return
    the shared Answer dataclass used by feedback/SGD/cost code elsewhere,
    and widening that contract with a proxy-only header field isn't worth
    it for this one panel."""
    base_url = _endpoint_base_url(proxy_base_url, wire_format)
    start = time.perf_counter()
    try:
        if wire_format == WIRE_ANTHROPIC:
            response = httpx.post(
                f"{base_url}/v1/messages",
                headers={"x-api-key": api_key, "anthropic-version": ANTHROPIC_VERSION},
                json={
                    "model": model,
                    "max_tokens": DEFAULT_MAX_TOKENS,
                    "temperature": 0.0,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=CALL_TIMEOUT_SECONDS,
            )
        else:
            response = httpx.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                },
                timeout=CALL_TIMEOUT_SECONDS,
            )
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPError as e:
        return ProxyCallResult(
            text="", tokens_pruned=None,
            latency_seconds=time.perf_counter() - start, error=str(e))

    latency = time.perf_counter() - start
    tokens_pruned_header = response.headers.get(TOKENS_PRUNED_HEADER)
    tokens_pruned = int(tokens_pruned_header) if tokens_pruned_header is not None else None

    if wire_format == WIRE_ANTHROPIC:
        blocks = payload.get("content") or []
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
    else:
        text = payload["choices"][0]["message"]["content"] or ""

    return ProxyCallResult(
        text=text, tokens_pruned=tokens_pruned, latency_seconds=latency)
