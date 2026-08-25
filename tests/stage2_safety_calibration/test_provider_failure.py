import httpx

from tokenthrift.generation.openai_compatible_client import OpenAICompatibleClient


def test_connection_failure_is_surfaced_not_silently_swallowed(monkeypatch):
    def _raise_connect_error(*args, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "post", _raise_connect_error)
    client = OpenAICompatibleClient(
        base_url="https://api.example.com/v1", api_key="fake-key", model="fake-model")

    answer = client.generate("hello")

    assert not answer.succeeded
    assert answer.error is not None
    assert answer.text == ""


def test_rate_limit_failure_is_distinguishable_from_a_successful_empty_answer(monkeypatch):
    request = httpx.Request("POST", "https://api.example.com/v1/chat/completions")
    response = httpx.Response(
        429, request=request, json={"error": {"message": "rate limited"}})

    monkeypatch.setattr(httpx, "post", lambda *a, **kw: response)
    client = OpenAICompatibleClient(
        base_url="https://api.example.com/v1", api_key="fake-key", model="fake-model")

    answer = client.generate("hello")

    assert not answer.succeeded
    assert answer.error is not None
    # a real successful-but-empty generation would have succeeded=True and
    # error=None — failure must never be conflated with that state
    assert answer.text == ""
