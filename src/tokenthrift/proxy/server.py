from __future__ import annotations

import os
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

from tokenthrift.core.types import Policy
from tokenthrift.proxy.chunking import strip_marked_blocks
from tokenthrift.proxy.pruning import prune_message_text
from tokenthrift.safety.policy import PRESETS, default_policy

UPSTREAM_ENV = "TOKENTHRIFT_UPSTREAM_BASE_URL"
POLICY_ENV = "TOKENTHRIFT_POLICY_PRESET"
TOKENS_PRUNED_HEADER = "X-TokenThrift-Tokens-Pruned"

REQUEST_TIMEOUT_SECONDS = 120.0

app = FastAPI(title="TokenThrift Proxy")


def _upstream_base_url() -> str:
    """No default on purpose: forwarding to the wrong upstream (or none)
    with a real API key attached would be a much worse failure than a
    proxy that refuses to start."""
    base_url = os.environ.get(UPSTREAM_ENV)
    if not base_url:
        raise RuntimeError(
            f"{UPSTREAM_ENV} must be set to the real provider's base URL, "
            "e.g. https://api.groq.com/openai/v1 or https://api.anthropic.com")
    return base_url.rstrip("/")


def _policy() -> Policy:
    preset_name = os.environ.get(POLICY_ENV, "conservative")
    return PRESETS.get(preset_name, default_policy())


def _forward_headers(request: Request, names: tuple[str, ...]) -> dict[str, str]:
    """Pass through only the auth/protocol headers the wire format needs —
    never the caller's Host/Content-Length, and never anything the proxy
    itself would need to inspect. The proxy holds no API key of its own and
    never logs the ones it forwards."""
    lowered = {k.lower(): v for k, v in request.headers.items()}
    return {name: lowered[name.lower()] for name in names if name.lower() in lowered}


def _respond(
    upstream_url: str, headers: dict[str, str], body: dict[str, Any],
    extra_response_headers: dict[str, str],
) -> Response | StreamingResponse:
    if not body.get("stream"):
        upstream_response = httpx.post(
            upstream_url, headers=headers, json=body, timeout=REQUEST_TIMEOUT_SECONDS)
        return Response(
            content=upstream_response.content,
            status_code=upstream_response.status_code,
            media_type=upstream_response.headers.get("content-type", "application/json"),
            headers=extra_response_headers,
        )

    stream_ctx = httpx.stream(
        "POST", upstream_url, headers=headers, json=body, timeout=REQUEST_TIMEOUT_SECONDS)
    upstream = stream_ctx.__enter__()

    def iter_bytes():
        try:
            yield from upstream.iter_bytes()
        finally:
            stream_ctx.__exit__(None, None, None)

    return StreamingResponse(
        iter_bytes(),
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "text/event-stream"),
        headers=extra_response_headers,
    )


def _extract_openai_query(messages: list[dict]) -> str:
    """The query for relevance scoring, not the query the caller sent to
    the model — those differ once a message bundles marked context with
    the actual question in one turn (see strip_marked_blocks)."""
    for message in reversed(messages):
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            return strip_marked_blocks(message["content"])
    return ""


def _prune_openai_messages(
    messages: list[dict], query: str, policy: Policy,
) -> tuple[list[dict], int]:
    pruned_messages = []
    tokens_pruned = 0
    for i, message in enumerate(messages):
        content = message.get("content")
        if isinstance(content, str):
            result = prune_message_text(content, query, policy, block_id_prefix=f"msg{i}")
            pruned_messages.append({**message, "content": result.text})
            tokens_pruned += result.tokens_pruned
        else:
            # Non-string content (e.g. multimodal parts) is passed through
            # untouched — pruning only ever acts on marked plain text.
            pruned_messages.append(message)
    return pruned_messages, tokens_pruned


@app.post("/v1/chat/completions", response_model=None)
def chat_completions(request: Request, body: dict[str, Any]) -> Response | StreamingResponse:
    """OpenAI-wire-compatible endpoint — serves Groq, OpenAI, OpenRouter, and
    any other OpenAI-compatible upstream the operator points this proxy at."""
    messages = body.get("messages", [])
    query = _extract_openai_query(messages)
    pruned_messages, tokens_pruned = _prune_openai_messages(messages, query, _policy())

    forward_body = {**body, "messages": pruned_messages}
    headers = _forward_headers(request, ("authorization",))
    return _respond(
        f"{_upstream_base_url()}/chat/completions", headers, forward_body,
        {TOKENS_PRUNED_HEADER: str(tokens_pruned)},
    )


def _extract_anthropic_query(messages: list[dict]) -> str:
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return strip_marked_blocks(content)
        if isinstance(content, list):
            return strip_marked_blocks("\n".join(
                block.get("text", "") for block in content
                if isinstance(block, dict) and block.get("type") == "text"))
    return ""


def _prune_anthropic_content(
    content: Any, query: str, policy: Policy, prefix: str,
) -> tuple[Any, int]:
    if isinstance(content, str):
        result = prune_message_text(content, query, policy, block_id_prefix=prefix)
        return result.text, result.tokens_pruned
    if isinstance(content, list):
        new_blocks = []
        tokens_pruned = 0
        for j, block in enumerate(content):
            if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
                result = prune_message_text(
                    block["text"], query, policy, block_id_prefix=f"{prefix}-b{j}")
                new_blocks.append({**block, "text": result.text})
                tokens_pruned += result.tokens_pruned
            else:
                new_blocks.append(block)
        return new_blocks, tokens_pruned
    return content, 0


@app.post("/v1/messages", response_model=None)
def messages_endpoint(request: Request, body: dict[str, Any]) -> Response | StreamingResponse:
    """Anthropic Messages-API-wire-compatible endpoint."""
    messages = body.get("messages", [])
    query = _extract_anthropic_query(messages)
    policy = _policy()

    pruned_messages = []
    tokens_pruned = 0
    for i, message in enumerate(messages):
        new_content, message_tokens_pruned = _prune_anthropic_content(
            message.get("content"), query, policy, prefix=f"msg{i}")
        pruned_messages.append({**message, "content": new_content})
        tokens_pruned += message_tokens_pruned

    forward_body = {**body, "messages": pruned_messages}
    if "system" in body:
        new_system, system_tokens_pruned = _prune_anthropic_content(
            body["system"], query, policy, prefix="system")
        forward_body["system"] = new_system
        tokens_pruned += system_tokens_pruned

    headers = _forward_headers(request, ("x-api-key", "anthropic-version"))
    return _respond(
        f"{_upstream_base_url()}/v1/messages", headers, forward_body,
        {TOKENS_PRUNED_HEADER: str(tokens_pruned)},
    )


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    """Liveness *and* config visibility for a caller that isn't the proxy's
    own request path — e.g. the UI's read-only status panel. Reports
    whether an upstream is configured, never the upstream URL itself: the
    proxy's "holds/reveals no secrets" contract extends to its own config,
    not just to forwarded API keys."""
    return {
        "status": "ok",
        "upstream_configured": bool(os.environ.get(UPSTREAM_ENV)),
        "policy_preset": _policy().preset_name,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8787")))
