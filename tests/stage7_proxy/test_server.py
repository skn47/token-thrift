import httpx
import pytest
from fastapi.testclient import TestClient

from tokenthrift.proxy.server import TOKENS_PRUNED_HEADER, UPSTREAM_ENV, app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _upstream(monkeypatch):
    monkeypatch.setenv(UPSTREAM_ENV, "https://api.example.com/openai/v1")


def _fake_post_capturing(captured, response_json):
    def _fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        request = httpx.Request("POST", url)
        return httpx.Response(200, request=request, json=response_json)
    return _fake_post


def test_unmarked_request_is_forwarded_completely_untouched(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        httpx, "post",
        _fake_post_capturing(captured, {"choices": [{"message": {"content": "hi"}}]}))

    body = {
        "model": "openai/gpt-oss-20b",
        "messages": [{"role": "user", "content": "What is 2+2?"}],
    }
    response = client.post(
        "/v1/chat/completions", json=body,
        headers={"Authorization": "Bearer sk-test"})

    assert response.status_code == 200
    assert captured["json"]["messages"] == body["messages"]
    assert captured["url"] == "https://api.example.com/openai/v1/chat/completions"
    assert captured["headers"]["authorization"] == "Bearer sk-test"
    assert response.headers[TOKENS_PRUNED_HEADER] == "0"


def test_marked_content_is_pruned_before_forwarding(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        httpx, "post",
        _fake_post_capturing(captured, {"choices": [{"message": {"content": "hi"}}]}))

    body = {
        "model": "openai/gpt-oss-20b",
        "messages": [{
            "role": "user",
            "content": (
                "<tokenthrift:context>\n"
                "How do I reset my password? Visit Settings and click "
                "Reset.\n\n"
                "Our cafeteria menu rotates weekly on Mondays.\n"
                "</tokenthrift:context>\n\n"
                "Question: how do I reset my password?"),
        }],
    }
    response = client.post(
        "/v1/chat/completions", json=body,
        headers={"Authorization": "Bearer sk-test"})

    assert response.status_code == 200
    forwarded_content = captured["json"]["messages"][0]["content"]
    assert "Question: how do I reset my password?" in forwarded_content
    assert forwarded_content != body["messages"][0]["content"]


def test_pruning_still_works_when_many_chunks_share_the_single_user_message(monkeypatch):
    """Regression test: with only one user message carrying both the marked
    context and the trailing question (the shape build_marked_prompt/the UI
    actually produce), the query for relevance scoring must be just the
    question — not the whole message, which at realistic chunk counts
    dilutes TF-IDF similarity until nothing scores low enough to prune."""
    captured = {}
    monkeypatch.setattr(
        httpx, "post",
        _fake_post_capturing(captured, {"choices": [{"message": {"content": "hi"}}]}))

    on_topic = "How do I reset my password? Go to Settings > Security > Reset."
    off_topic_paragraphs = "\n\n".join(
        f"Paragraph {i} covers an unrelated topic about office snacks and "
        f"parking passes, item number {i}."
        for i in range(10))
    body = {
        "model": "openai/gpt-oss-20b",
        "messages": [{
            "role": "user",
            "content": (
                f"<tokenthrift:context>\n{on_topic}\n\n{off_topic_paragraphs}\n"
                "</tokenthrift:context>\n\n"
                "Question: how do I reset my password?"),
        }],
    }
    response = client.post(
        "/v1/chat/completions", json=body,
        headers={"Authorization": "Bearer sk-test"})

    assert response.status_code == 200
    assert int(response.headers[TOKENS_PRUNED_HEADER]) > 0
    forwarded_content = captured["json"]["messages"][0]["content"]
    assert "Settings > Security > Reset" in forwarded_content
    # Some (not necessarily all — the conservative default's min_context
    # floor may keep a few) off-topic paragraphs must have been dropped.
    assert forwarded_content.count("Paragraph") < 10


def test_only_authorization_header_is_forwarded_for_openai_wire(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        httpx, "post",
        _fake_post_capturing(captured, {"choices": [{"message": {"content": "hi"}}]}))

    client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": "Bearer sk-test", "X-Custom-Header": "should-not-forward"})

    assert set(captured["headers"].keys()) == {"authorization"}


def test_anthropic_wire_forwards_x_api_key_and_anthropic_version(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        httpx, "post",
        _fake_post_capturing(
            captured, {"content": [{"type": "text", "text": "hi"}], "usage": {}}))
    monkeypatch.setenv(UPSTREAM_ENV, "https://api.anthropic.com")

    response = client.post(
        "/v1/messages",
        json={
            "model": "claude-test", "max_tokens": 100,
            "messages": [{"role": "user", "content": "hello"}],
        },
        headers={"x-api-key": "sk-ant-test", "anthropic-version": "2023-06-01"})

    assert response.status_code == 200
    assert captured["url"] == "https://api.anthropic.com/v1/messages"
    assert captured["headers"]["x-api-key"] == "sk-ant-test"
    assert captured["headers"]["anthropic-version"] == "2023-06-01"
    assert "authorization" not in captured["headers"]


def test_anthropic_content_block_list_is_pruned_per_block(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        httpx, "post",
        _fake_post_capturing(
            captured, {"content": [{"type": "text", "text": "hi"}], "usage": {}}))
    monkeypatch.setenv(UPSTREAM_ENV, "https://api.anthropic.com")

    body = {
        "model": "claude-test", "max_tokens": 100,
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "<tokenthrift:context>\n"
                        "How do I reset my password? Visit Settings.\n\n"
                        "Cafeteria menu rotates weekly.\n"
                        "</tokenthrift:context>"),
                },
            ],
        }],
    }
    client.post(
        "/v1/messages", json=body,
        headers={"x-api-key": "sk-ant-test", "anthropic-version": "2023-06-01"})

    forwarded = captured["json"]["messages"][0]["content"][0]["text"]
    assert forwarded != body["messages"][0]["content"][0]["text"]


def test_missing_upstream_config_fails_loudly_not_silently(monkeypatch):
    monkeypatch.delenv(UPSTREAM_ENV, raising=False)
    with pytest.raises(RuntimeError, match=UPSTREAM_ENV):
        client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer sk-test"})


def test_healthz_reports_config_without_leaking_the_upstream_url():
    response = client.get("/healthz")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["upstream_configured"] is True
    assert payload["policy_preset"] == "conservative"
    assert "api.example.com" not in response.text


def test_healthz_reports_unconfigured_upstream_without_raising(monkeypatch):
    monkeypatch.delenv(UPSTREAM_ENV, raising=False)
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["upstream_configured"] is False
