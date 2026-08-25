from streamlit.testing.v1 import AppTest

from tokenthrift.config import REPO_ROOT
from tokenthrift.generation.answer import Answer
from tokenthrift.generation.openai_compatible_client import OpenAICompatibleClient
from tokenthrift.ui.proxy_client import ProxyCallResult

APP_PATH = REPO_ROOT / "src" / "tokenthrift" / "ui" / "app.py"


def _fake_generate(self, prompt: str, temperature: float = 0.0) -> Answer:
    return Answer(
        text="This is a mocked answer.", latency_seconds=0.01,
        input_tokens=10, output_tokens=5, model=self.model, error=None,
    )


def test_proxy_panel_renders_with_toggle_off_by_default():
    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()
    assert not at.exception

    assert at.text_input(key="proxy_base_url_input").value == "http://localhost:8787"
    assert at.toggle(key="proxy_enabled_toggle").value is False
    assert at.button(key="proxy_health_button")


def test_enabling_proxy_toggle_renders_the_live_proxy_panel(monkeypatch):
    monkeypatch.setattr(OpenAICompatibleClient, "generate", _fake_generate)
    monkeypatch.setattr(
        "tokenthrift.ui.app.call_through_proxy",
        lambda *args, **kwargs: ProxyCallResult(
            text="proxied answer", tokens_pruned=12, latency_seconds=0.02))

    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()

    at.text_input(key="api_key_input").set_value("test-key").run()
    at.toggle(key="proxy_enabled_toggle").set_value(True).run()
    at.text_input(key="query_input").set_value("How do I reset my password?").run()
    at.button(key="compare_button").click().run()

    assert not at.exception
    subheaders = [s.value for s in at.subheader]
    assert "🔌 Live via TokenThrift Proxy" in subheaders


def test_proxy_disabled_never_shows_the_live_panel(monkeypatch):
    monkeypatch.setattr(OpenAICompatibleClient, "generate", _fake_generate)

    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()

    at.text_input(key="api_key_input").set_value("test-key").run()
    at.text_input(key="query_input").set_value("How do I reset my password?").run()
    at.button(key="compare_button").click().run()

    assert not at.exception
    subheaders = [s.value for s in at.subheader]
    assert "🔌 Live via TokenThrift Proxy" not in subheaders
