import httpx

from tokenthrift.generation.providers import WIRE_ANTHROPIC, WIRE_OPENAI
from tokenthrift.proxy.server import TOKENS_PRUNED_HEADER
from tokenthrift.ui.proxy_client import call_through_proxy, check_health


def _response(json_body, headers=None, status_code=200):
    request = httpx.Request("GET", "http://proxy.example")
    return httpx.Response(status_code, request=request, json=json_body, headers=headers or {})


def test_check_health_reports_reachable_config(monkeypatch):
    monkeypatch.setattr(
        httpx, "get",
        lambda url, timeout=None: _response(
            {"status": "ok", "upstream_configured": True, "policy_preset": "balanced"}))

    health = check_health("http://localhost:8787")
    assert health.reachable is True
    assert health.upstream_configured is True
    assert health.policy_preset == "balanced"
    assert health.error is None


def test_check_health_reports_unreachable_without_raising(monkeypatch):
    def _raise(url, timeout=None):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "get", _raise)
    health = check_health("http://localhost:8787")
    assert health.reachable is False
    assert health.error is not None


def test_call_through_proxy_openai_wire_reads_tokens_pruned_header(monkeypatch):
    captured = {}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        return _response(
            {"choices": [{"message": {"content": "hi there"}}]},
            headers={TOKENS_PRUNED_HEADER: "42"})

    monkeypatch.setattr(httpx, "post", _fake_post)
    result = call_through_proxy(
        "http://localhost:8787", WIRE_OPENAI, "sk-test", "gpt-test", "prompt text")

    assert result.error is None
    assert result.text == "hi there"
    assert result.tokens_pruned == 42
    assert captured["url"] == "http://localhost:8787/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer sk-test"


def test_call_through_proxy_anthropic_wire_hits_v1_messages(monkeypatch):
    captured = {}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        return _response(
            {"content": [{"type": "text", "text": "hi"}]},
            headers={TOKENS_PRUNED_HEADER: "7"})

    monkeypatch.setattr(httpx, "post", _fake_post)
    result = call_through_proxy(
        "http://localhost:8787", WIRE_ANTHROPIC, "sk-ant-test", "claude-test", "prompt text")

    assert result.text == "hi"
    assert result.tokens_pruned == 7
    assert captured["url"] == "http://localhost:8787/v1/messages"
    assert captured["headers"]["x-api-key"] == "sk-ant-test"
    assert "Authorization" not in captured["headers"]


def test_call_through_proxy_missing_header_is_none_not_zero(monkeypatch):
    monkeypatch.setattr(
        httpx, "post",
        lambda url, headers=None, json=None, timeout=None: _response(
            {"choices": [{"message": {"content": "hi"}}]}))

    result = call_through_proxy(
        "http://localhost:8787", WIRE_OPENAI, "sk-test", "m", "prompt")
    assert result.tokens_pruned is None


def test_call_through_proxy_connection_error_is_reported_not_raised(monkeypatch):
    def _raise(url, headers=None, json=None, timeout=None):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "post", _raise)
    result = call_through_proxy(
        "http://localhost:8787", WIRE_OPENAI, "sk-test", "m", "prompt")
    assert result.error is not None
    assert result.text == ""
