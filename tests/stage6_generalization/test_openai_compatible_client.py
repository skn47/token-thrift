import httpx

from tokenthrift.generation.openai_compatible_client import OpenAICompatibleClient


def test_generate_sends_bearer_auth_and_openai_wire_shape(monkeypatch):
    captured = {}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        request = httpx.Request("POST", url)
        return httpx.Response(
            200, request=request,
            json={
                "choices": [{"message": {"content": "hi there"}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 3},
            },
        )

    monkeypatch.setattr(httpx, "post", _fake_post)
    client = OpenAICompatibleClient(
        base_url="https://api.example.com/v1", api_key="sk-test", model="gpt-test")

    answer = client.generate("hello", temperature=0.2)

    assert captured["url"] == "https://api.example.com/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    assert captured["json"]["model"] == "gpt-test"
    assert captured["json"]["messages"] == [{"role": "user", "content": "hello"}]
    assert captured["json"]["temperature"] == 0.2

    assert answer.succeeded
    assert answer.text == "hi there"
    assert answer.input_tokens == 12
    assert answer.output_tokens == 3
    assert answer.error is None


def test_base_url_trailing_slash_does_not_produce_a_double_slash(monkeypatch):
    captured = {}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        request = httpx.Request("POST", url)
        return httpx.Response(
            200, request=request,
            json={"choices": [{"message": {"content": "ok"}}], "usage": {}})

    monkeypatch.setattr(httpx, "post", _fake_post)
    client = OpenAICompatibleClient(
        base_url="https://api.example.com/v1/", api_key="k", model="m")
    client.generate("hi")

    assert captured["url"] == "https://api.example.com/v1/chat/completions"
