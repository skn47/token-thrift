from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

from tokenthrift.core.types import Policy
from tokenthrift.proxy.chunking import strip_marked_blocks
from tokenthrift.proxy.pruning import prune_message_text
from tokenthrift.proxy.tool_results import wrap_as_marked
from tokenthrift.safety.policy import PRESETS, default_policy

UPSTREAM_ENV = "TOKENTHRIFT_UPSTREAM_BASE_URL"
POLICY_ENV = "TOKENTHRIFT_POLICY_PRESET"
AUTO_MARK_ENV = "TOKENTHRIFT_AUTO_MARK_TOOL_RESULTS"
TOKENS_PRUNED_HEADER = "X-TokenThrift-Tokens-Pruned"

REQUEST_TIMEOUT_SECONDS = 120.0

app = FastAPI(title="TokenThrift Proxy")


@dataclass
class RuntimeConfig:
    """The proxy's only runtime-mutable setting. Everything else
    (`_upstream_base_url`, `_policy`) is read fresh from the environment on
    every request; this one is instead flipped live via `POST /v1/config`
    (normally from the Streamlit sidebar's Proxy panel) so turning
    tool-result auto-marking on/off never requires restarting a proxy an
    agent is already pointed at."""
    auto_mark_tool_results: bool = False


_runtime_config = RuntimeConfig(
    auto_mark_tool_results=os.environ.get(AUTO_MARK_ENV, "") == "1")


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
    messages: list[dict], query: str, policy: Policy, auto_mark_tool_results: bool,
) -> tuple[list[dict], int]:
    pruned_messages = []
    tokens_pruned = 0
    for i, message in enumerate(messages):
        content = message.get("content")
        if isinstance(content, str):
            text = content
            if auto_mark_tool_results and message.get("role") == "tool":
                # OpenAI tool-result messages carry the returned data (file
                # reads, command output) as plain string content — the
                # structural "role" signal is enough to know it's prunable
                # without the caller adding markers itself.
                text = wrap_as_marked(text)
            result = prune_message_text(text, query, policy, block_id_prefix=f"msg{i}")
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
    pruned_messages, tokens_pruned = _prune_openai_messages(
        messages, query, _policy(), _runtime_config.auto_mark_tool_results)

    if tokens_pruned:
        # flush=True: stdout is block-buffered whenever it isn't a TTY (a
        # log redirect, `tee`, a process manager) — without it this line
        # can sit unflushed for as long as the process runs, defeating the
        # entire point of a live "here's proof it's working" signal.
        print(f"[TokenThrift] pruned {tokens_pruned} tokens from this request", flush=True)

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


def _prune_tool_result_block(
    block: dict, query: str, policy: Policy, prefix: str,
) -> tuple[dict, int]:
    """A `tool_result` block carries its payload in `content`, not `text`
    (that key is what a `text` block uses) — its own `content` can in turn
    be a string or a list of sub-blocks, mirroring the message-level content
    shape one level down. Only reached when auto-marking is on: the block's
    `type` is already the structural signal that it's prunable, so its text
    is force-wrapped rather than requiring a manual marker."""
    result_content = block.get("content")
    if isinstance(result_content, str):
        result = prune_message_text(
            wrap_as_marked(result_content), query, policy, block_id_prefix=prefix)
        return {**block, "content": result.text}, result.tokens_pruned
    if isinstance(result_content, list):
        new_sub_blocks = []
        tokens_pruned = 0
        for k, sub_block in enumerate(result_content):
            if isinstance(sub_block, dict) and sub_block.get("type") == "text" and isinstance(sub_block.get("text"), str):
                result = prune_message_text(
                    wrap_as_marked(sub_block["text"]), query, policy,
                    block_id_prefix=f"{prefix}-s{k}")
                new_sub_blocks.append({**sub_block, "text": result.text})
                tokens_pruned += result.tokens_pruned
            else:
                new_sub_blocks.append(sub_block)
        return {**block, "content": new_sub_blocks}, tokens_pruned
    return block, 0


def _prune_anthropic_content(
    content: Any, query: str, policy: Policy, prefix: str, auto_mark_tool_results: bool,
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
            elif auto_mark_tool_results and isinstance(block, dict) and block.get("type") == "tool_result":
                new_block, block_tokens_pruned = _prune_tool_result_block(
                    block, query, policy, prefix=f"{prefix}-b{j}")
                new_blocks.append(new_block)
                tokens_pruned += block_tokens_pruned
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
    auto_mark_tool_results = _runtime_config.auto_mark_tool_results

    pruned_messages = []
    tokens_pruned = 0
    for i, message in enumerate(messages):
        new_content, message_tokens_pruned = _prune_anthropic_content(
            message.get("content"), query, policy, prefix=f"msg{i}",
            auto_mark_tool_results=auto_mark_tool_results)
        pruned_messages.append({**message, "content": new_content})
        tokens_pruned += message_tokens_pruned

    forward_body = {**body, "messages": pruned_messages}
    if "system" in body:
        new_system, system_tokens_pruned = _prune_anthropic_content(
            body["system"], query, policy, prefix="system",
            auto_mark_tool_results=auto_mark_tool_results)
        forward_body["system"] = new_system
        tokens_pruned += system_tokens_pruned

    if tokens_pruned:
        # flush=True: stdout is block-buffered whenever it isn't a TTY (a
        # log redirect, `tee`, a process manager) — without it this line
        # can sit unflushed for as long as the process runs, defeating the
        # entire point of a live "here's proof it's working" signal.
        print(f"[TokenThrift] pruned {tokens_pruned} tokens from this request", flush=True)

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
        "auto_mark_tool_results": _runtime_config.auto_mark_tool_results,
    }


@app.get("/v1/config")
def get_config() -> dict[str, Any]:
    return {"auto_mark_tool_results": _runtime_config.auto_mark_tool_results}


@app.post("/v1/config", response_model=None)
def set_config(body: dict[str, Any]) -> dict[str, Any]:
    """The proxy's one runtime-mutable setting, changed in place — normally
    from the Streamlit sidebar's Proxy panel, so an operator doesn't have
    to set an env var and restart a proxy an agent is already using."""
    if "auto_mark_tool_results" in body:
        _runtime_config.auto_mark_tool_results = bool(body["auto_mark_tool_results"])
    return {"auto_mark_tool_results": _runtime_config.auto_mark_tool_results}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8787"))
    auto_mark_status = "on" if _runtime_config.auto_mark_tool_results else "off"
    print(
        f"[TokenThrift] proxy on :{port} -> {_upstream_base_url()}, "
        f"policy: {_policy().preset_name}, auto-mark: {auto_mark_status}",
        flush=True)

    uvicorn.run(app, host="0.0.0.0", port=port)
