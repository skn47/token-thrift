from streamlit.testing.v1 import AppTest

from tokenthrift.config import REPO_ROOT
from tokenthrift.generation.answer import Answer
from tokenthrift.generation.openai_compatible_client import OpenAICompatibleClient

APP_PATH = REPO_ROOT / "src" / "tokenthrift" / "ui" / "app.py"


def _fake_generate(self, prompt: str, temperature: float = 0.0) -> Answer:
    return Answer(
        text="This is a mocked answer.", latency_seconds=0.01,
        input_tokens=10, output_tokens=5, model=self.model, error=None,
    )


def test_app_renders_initial_state_without_crashing():
    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()
    assert not at.exception


def test_app_renders_both_comparison_columns_with_mocked_client(monkeypatch):
    monkeypatch.setattr(OpenAICompatibleClient, "generate", _fake_generate)

    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()
    assert not at.exception

    at.text_input(key="api_key_input").set_value("test-key").run()
    at.text_input(key="query_input").set_value(
        "How do I reset my password?").run()
    at.button(key="compare_button").click().run()

    assert not at.exception

    subheaders = [s.value for s in at.subheader]
    assert "Without TokenThrift" in subheaders
    assert "With TokenThrift" in subheaders

    # 4 top-level metric tiles: tokens saved, cost saved, pruning latency,
    # generation latency — distinct fields for the two comparison columns.
    # (Later stages add further metric tiles elsewhere on the page, e.g.
    # the adaptation-history panel, so this checks presence, not a total.)
    metric_labels = {m.label for m in at.metric}
    assert {
        "Context tokens saved", "Estimated input cost saved",
        "Pruning latency", "Generation latency (this run)",
    } <= metric_labels
