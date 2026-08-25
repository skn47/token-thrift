import httpx

from tokenthrift.generation.anthropic_client import ANTHROPIC_VERSION, AnthropicClient


def test_generate_sends_x_api_key_and_anthropic_wire_shape(monkeypatch):
    captured = {}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        request = httpx.Request("POST", url)
        return httpx.Response(
            200, request=request,
            json={
                "content": [{"type": "text", "text": "hi there"}],
                "usage": {"input_tokens": 12, "output_tokens": 3},
            },
        )

    monkeypatch.setattr(httpx, "post", _fake_post)
    client = AnthropicClient(api_key="sk-ant-test", model="claude-test")

    answer = client.generate("hello", temperature=0.2)

    assert captured["url"] == "https://api.anthropic.com/v1/messages"
    assert captured["headers"]["x-api-key"] == "sk-ant-test"
    assert captured["headers"]["anthropic-version"] == ANTHROPIC_VERSION
    assert captured["json"]["model"] == "claude-test"
    assert captured["json"]["messages"] == [{"role": "user", "content": "hello"}]
    assert "max_tokens" in captured["json"]

    assert answer.succeeded
    assert answer.text == "hi there"
    assert answer.input_tokens == 12
    assert answer.output_tokens == 3


def test_generate_failure_surfaces_an_error_not_a_crash(monkeypatch):
    def _raise_connect_error(*args, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "post", _raise_connect_error)
    client = AnthropicClient(api_key="sk-ant-test", model="claude-test")

    answer = client.generate("hello")

    assert not answer.succeeded
    assert answer.error is not None
    assert answer.text == ""
