from streamlit.testing.v1 import AppTest

from tokenthrift.config import REPO_ROOT
from tokenthrift.generation.answer import Answer
from tokenthrift.generation.openai_compatible_client import OpenAICompatibleClient

APP_PATH = REPO_ROOT / "src" / "tokenthrift" / "ui" / "app.py"


def _fake_generate(self, prompt: str, temperature: float = 0.0) -> Answer:
    return Answer(
        text="Mocked answer.", latency_seconds=0.01,
        input_tokens=10, output_tokens=5, model=self.model, error=None,
    )


def test_theme_css_is_injected_without_crashing():
    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()
    assert not at.exception
    assert any("<style>" in m.value for m in at.markdown)


def test_every_prior_stage_widget_key_still_resolves():
    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()
    assert not at.exception

    assert at.selectbox(key="corpus_select") is not None
    assert at.selectbox(key="provider_select") is not None
    assert at.text_input(key="model_input") is not None
    assert at.text_input(key="api_key_input") is not None
    assert at.toggle(key="calibration_enabled") is not None
    assert at.text_input(key="query_input") is not None
    assert at.button(key="compare_button") is not None


def test_kept_and_pruned_panels_are_tabs(monkeypatch):
    monkeypatch.setattr(OpenAICompatibleClient, "generate", _fake_generate)

    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()
    at.text_input(key="api_key_input").set_value("test-key").run()
    at.text_input(key="query_input").set_value("How do I reset my password?").run()
    at.button(key="compare_button").click().run()
    assert not at.exception

    tab_labels = [t.label for t in at.tabs]
    assert any(label.startswith("Kept") for label in tab_labels)
    assert any(label.startswith("Pruned") for label in tab_labels)


def test_switching_provider_refills_the_default_model():
    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()
    assert at.text_input(key="model_input").value == "openai/gpt-oss-20b"

    at.selectbox(key="provider_select").set_value("openai").run()
    assert not at.exception
    assert at.text_input(key="model_input").value == "gpt-4o-mini"
