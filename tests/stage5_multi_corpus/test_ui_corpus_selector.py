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


def test_corpus_selector_defaults_to_lighthouse():
    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()
    assert not at.exception
    assert at.selectbox(key="corpus_select").value == "lighthouse"
    assert "Lighthouse" in at.text_input(key="query_input").label


def test_switching_corpus_updates_query_label_and_retrieval_source(monkeypatch):
    monkeypatch.setattr(OpenAICompatibleClient, "generate", _fake_generate)

    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()
    at.selectbox(key="corpus_select").set_value("nimbus").run()
    assert not at.exception
    assert "Nimbus" in at.text_input(key="query_input").label

    at.text_input(key="api_key_input").set_value("test-key").run()
    at.text_input(key="query_input").set_value(
        "How do I pair a Wi-Fi device with Nimbus?").run()
    at.button(key="compare_button").click().run()
    assert not at.exception

    ranked_doc_ids = {
        rc.chunk.doc_id for rc in at.session_state["last_result"]["ranked"]
    }
    lighthouse_only_doc_ids = {
        "api-tasks", "api-webhooks", "guide-permissions", "guide-integrations",
    }
    assert not (ranked_doc_ids & lighthouse_only_doc_ids)


def test_custom_folder_option_reveals_controls_and_gates_compare(tmp_path):
    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()
    at.selectbox(key="corpus_select").set_value("custom").run()
    assert not at.exception

    at.text_input(key="custom_corpus_path")
    at.selectbox(key="model_source_select")
    assert any("No ground truth" in c.value for c in at.caption)

    at.text_input(key="api_key_input").set_value("test-key").run()
    at.text_input(key="query_input").set_value("anything").run()
    assert at.button(key="compare_button").proto.disabled is True

    (tmp_path / "notes.md").write_text(
        "# Topic\nSome unique custom content about zephyrs.\n")
    at.text_input(key="custom_corpus_path").set_value(str(tmp_path)).run()
    assert not at.exception
    assert at.button(key="compare_button").proto.disabled is False


def test_custom_folder_accepts_a_tilde_path(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "notes.md").write_text("# Topic\nSome content.\n")

    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()
    at.selectbox(key="corpus_select").set_value("custom").run()
    at.text_input(key="api_key_input").set_value("test-key").run()
    at.text_input(key="query_input").set_value("anything").run()

    at.text_input(key="custom_corpus_path").set_value("~/docs").run()
    assert not at.exception
    assert at.button(key="compare_button").proto.disabled is False


def test_browse_button_disabled_without_zenity(monkeypatch):
    monkeypatch.setattr("tokenthrift.ui.sidebar.shutil.which", lambda name: None)

    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()
    at.selectbox(key="corpus_select").set_value("custom").run()
    assert at.button(key="browse_folder_button").proto.disabled is True


def test_browse_button_fills_path_from_zenity(monkeypatch):
    monkeypatch.setattr(
        "tokenthrift.ui.sidebar.shutil.which", lambda name: "/usr/bin/zenity")

    class _FakeCompletedProcess:
        stdout = "/chosen/folder\n"

    monkeypatch.setattr(
        "tokenthrift.ui.sidebar.subprocess.run",
        lambda *a, **k: _FakeCompletedProcess())

    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()
    at.selectbox(key="corpus_select").set_value("custom").run()
    at.button(key="browse_folder_button").click().run()
    assert not at.exception
    assert at.text_input(key="custom_corpus_path").value == "/chosen/folder"


def test_custom_folder_compare_runs_unverified_pruning(tmp_path, monkeypatch):
    monkeypatch.setattr(OpenAICompatibleClient, "generate", _fake_generate)

    (tmp_path / "notes.md").write_text(
        "# Zephyr policy\nZephyrs are returned within thirty days.\n\n"
        "# Contact\nReach the zephyr team by email.\n")

    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()
    at.selectbox(key="corpus_select").set_value("custom").run()
    at.text_input(key="custom_corpus_path").set_value(str(tmp_path)).run()
    at.text_input(key="api_key_input").set_value("test-key").run()
    at.text_input(key="query_input").set_value(
        "How long until a zephyr is returned?").run()
    at.button(key="compare_button").click().run()
    assert not at.exception

    result = at.session_state["last_result"]["result"]
    assert result.pruning_enabled is True
    assert any("unverified" in c.value for c in at.caption)
