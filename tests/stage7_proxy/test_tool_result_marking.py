import httpx
from fastapi.testclient import TestClient

from tokenthrift.proxy.server import TOKENS_PRUNED_HEADER, UPSTREAM_ENV, app

client = TestClient(app)


def _fake_post_capturing(captured, response_json):
    def _fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        request = httpx.Request("POST", url)
        return httpx.Response(200, request=request, json=response_json)
    return _fake_post


def _many_chunks_body():
    """Same shape as the existing regression test in test_server.py: one
    on-topic sentence plus enough off-topic filler that the conservative
    preset's min_context floor can't force everything to be kept."""
    on_topic = "How do I reset my password? Go to Settings > Security > Reset."
    off_topic_paragraphs = "\n\n".join(
        f"Paragraph {i} covers an unrelated topic about office snacks and "
        f"parking passes, item number {i}."
        for i in range(10))
    return f"{on_topic}\n\n{off_topic_paragraphs}"


def test_openai_tool_message_unpruned_when_auto_mark_off(monkeypatch):
    monkeypatch.setenv(UPSTREAM_ENV, "https://api.example.com/openai/v1")
    captured = {}
    monkeypatch.setattr(
        httpx, "post",
        _fake_post_capturing(captured, {"choices": [{"message": {"content": "hi"}}]}))

    body = {
        "model": "m",
        "messages": [
            {"role": "user", "content": "Question: how do I reset my password?"},
            {"role": "tool", "tool_call_id": "call_1", "content": _many_chunks_body()},
        ],
    }
    response = client.post(
        "/v1/chat/completions", json=body, headers={"Authorization": "Bearer sk-test"})

    assert response.status_code == 200
    assert response.headers[TOKENS_PRUNED_HEADER] == "0"
    assert captured["json"]["messages"][1]["content"] == body["messages"][1]["content"]


def test_openai_tool_message_pruned_when_auto_mark_on(monkeypatch):
    monkeypatch.setenv(UPSTREAM_ENV, "https://api.example.com/openai/v1")
    captured = {}
    monkeypatch.setattr(
        httpx, "post",
        _fake_post_capturing(captured, {"choices": [{"message": {"content": "hi"}}]}))

    assert client.post("/v1/config", json={"auto_mark_tool_results": True}).json() == {
        "auto_mark_tool_results": True}

    body = {
        "model": "m",
        "messages": [
            {"role": "user", "content": "Question: how do I reset my password?"},
            {"role": "tool", "tool_call_id": "call_1", "content": _many_chunks_body()},
        ],
    }
    response = client.post(
        "/v1/chat/completions", json=body, headers={"Authorization": "Bearer sk-test"})

    assert response.status_code == 200
    assert int(response.headers[TOKENS_PRUNED_HEADER]) > 0
    forwarded = captured["json"]["messages"][1]["content"]
    assert "Settings > Security > Reset" in forwarded
    assert forwarded.count("Paragraph") < 10


def test_openai_regular_messages_unaffected_by_toggle(monkeypatch):
    monkeypatch.setenv(UPSTREAM_ENV, "https://api.example.com/openai/v1")
    captured = {}
    monkeypatch.setattr(
        httpx, "post",
        _fake_post_capturing(captured, {"choices": [{"message": {"content": "hi"}}]}))
    client.post("/v1/config", json={"auto_mark_tool_results": True})

    body = {
        "model": "m",
        "messages": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "Let me think."},
        ],
    }
    response = client.post(
        "/v1/chat/completions", json=body, headers={"Authorization": "Bearer sk-test"})

    assert response.status_code == 200
    assert captured["json"]["messages"] == body["messages"]


def test_openai_already_marked_tool_message_not_double_wrapped(monkeypatch):
    monkeypatch.setenv(UPSTREAM_ENV, "https://api.example.com/openai/v1")
    marked_content = (
        "<tokenthrift:context>\n" + _many_chunks_body() + "\n</tokenthrift:context>")
    body = {
        "model": "m",
        "messages": [
            {"role": "user", "content": "Question: how do I reset my password?"},
            {"role": "tool", "tool_call_id": "call_1", "content": marked_content},
        ],
    }

    captured_off = {}
    monkeypatch.setattr(
        httpx, "post",
        _fake_post_capturing(captured_off, {"choices": [{"message": {"content": "hi"}}]}))
    client.post("/v1/chat/completions", json=body, headers={"Authorization": "Bearer sk-test"})

    client.post("/v1/config", json={"auto_mark_tool_results": True})
    captured_on = {}
    monkeypatch.setattr(
        httpx, "post",
        _fake_post_capturing(captured_on, {"choices": [{"message": {"content": "hi"}}]}))
    client.post("/v1/chat/completions", json=body, headers={"Authorization": "Bearer sk-test"})

    assert captured_on["json"]["messages"][1]["content"] == captured_off["json"]["messages"][1]["content"]


def test_anthropic_tool_result_string_content_pruned_when_enabled(monkeypatch):
    monkeypatch.setenv(UPSTREAM_ENV, "https://api.anthropic.com")
    captured = {}
    monkeypatch.setattr(
        httpx, "post",
        _fake_post_capturing(
            captured, {"content": [{"type": "text", "text": "hi"}], "usage": {}}))
    client.post("/v1/config", json={"auto_mark_tool_results": True})

    body = {
        "model": "claude-test", "max_tokens": 100,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": _many_chunks_body()},
                {"type": "text", "text": "Question: how do I reset my password?"},
            ],
        }],
    }
    response = client.post(
        "/v1/messages", json=body,
        headers={"x-api-key": "sk-ant-test", "anthropic-version": "2023-06-01"})

    assert response.status_code == 200
    assert int(response.headers[TOKENS_PRUNED_HEADER]) > 0
    forwarded_block = captured["json"]["messages"][0]["content"][0]
    assert forwarded_block["type"] == "tool_result"
    assert "Settings > Security > Reset" in forwarded_block["content"]
    assert forwarded_block["content"].count("Paragraph") < 10


def test_anthropic_tool_result_list_content_pruned_when_enabled(monkeypatch):
    monkeypatch.setenv(UPSTREAM_ENV, "https://api.anthropic.com")
    captured = {}
    monkeypatch.setattr(
        httpx, "post",
        _fake_post_capturing(
            captured, {"content": [{"type": "text", "text": "hi"}], "usage": {}}))
    client.post("/v1/config", json={"auto_mark_tool_results": True})

    body = {
        "model": "claude-test", "max_tokens": 100,
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": [{"type": "text", "text": _many_chunks_body()}],
                },
                {"type": "text", "text": "Question: how do I reset my password?"},
            ],
        }],
    }
    response = client.post(
        "/v1/messages", json=body,
        headers={"x-api-key": "sk-ant-test", "anthropic-version": "2023-06-01"})

    assert response.status_code == 200
    assert int(response.headers[TOKENS_PRUNED_HEADER]) > 0
    forwarded_sub_block = captured["json"]["messages"][0]["content"][0]["content"][0]
    assert "Settings > Security > Reset" in forwarded_sub_block["text"]
    assert forwarded_sub_block["text"].count("Paragraph") < 10


def test_anthropic_tool_result_untouched_when_auto_mark_off(monkeypatch):
    monkeypatch.setenv(UPSTREAM_ENV, "https://api.anthropic.com")
    captured = {}
    monkeypatch.setattr(
        httpx, "post",
        _fake_post_capturing(
            captured, {"content": [{"type": "text", "text": "hi"}], "usage": {}}))

    body = {
        "model": "claude-test", "max_tokens": 100,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": _many_chunks_body()},
                {"type": "text", "text": "Question: how do I reset my password?"},
            ],
        }],
    }
    response = client.post(
        "/v1/messages", json=body,
        headers={"x-api-key": "sk-ant-test", "anthropic-version": "2023-06-01"})

    assert response.status_code == 200
    assert response.headers[TOKENS_PRUNED_HEADER] == "0"
    assert captured["json"]["messages"][0]["content"][0]["content"] == _many_chunks_body()


def test_config_get_post_roundtrip_and_healthz_reflects_state():
    assert client.get("/v1/config").json() == {"auto_mark_tool_results": False}

    post_response = client.post("/v1/config", json={"auto_mark_tool_results": True})
    assert post_response.json() == {"auto_mark_tool_results": True}
    assert client.get("/v1/config").json() == {"auto_mark_tool_results": True}

    healthz_payload = client.get("/healthz").json()
    assert healthz_payload["auto_mark_tool_results"] is True

    client.post("/v1/config", json={"auto_mark_tool_results": False})
    assert client.get("/v1/config").json() == {"auto_mark_tool_results": False}
