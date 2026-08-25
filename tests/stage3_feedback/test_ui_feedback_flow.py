from streamlit.testing.v1 import AppTest

from tokenthrift.config import REPO_ROOT
from tokenthrift.generation.answer import Answer
from tokenthrift.generation.openai_compatible_client import OpenAICompatibleClient

APP_PATH = REPO_ROOT / "src" / "tokenthrift" / "ui" / "app.py"


def _fake_generate(self, prompt: str, temperature: float = 0.0) -> Answer:
    return Answer(
        text="Click forgot password and follow the emailed reset link.",
        latency_seconds=0.01, input_tokens=10, output_tokens=5,
        model=self.model, error=None,
    )


def test_comparison_view_survives_reruns_triggered_by_feedback_buttons(monkeypatch):
    monkeypatch.setattr(OpenAICompatibleClient, "generate", _fake_generate)

    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()
    at.text_input(key="api_key_input").set_value("test-key").run()
    at.text_input(key="query_input").set_value("How do I reset my password?").run()
    at.button(key="compare_button").click().run()
    assert not at.exception

    at.button(key="fb_incorrect").click().run()
    assert not at.exception

    subheaders = [s.value for s in at.subheader]
    assert "Without TokenThrift" in subheaders
    assert "With TokenThrift" in subheaders


def test_incorrect_alone_does_not_move_the_calibrated_policy(monkeypatch):
    # Stage 3 corrects Stage 2's simplified stand-in: bare negative
    # feedback is ambiguous per the attribution table and must not shift
    # the policy on its own — only an improved full-context retry can.
    monkeypatch.setattr(OpenAICompatibleClient, "generate", _fake_generate)

    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()
    at.text_input(key="api_key_input").set_value("test-key").run()
    at.text_input(key="query_input").set_value("How do I reset my password?").run()
    at.toggle(key="calibration_enabled").set_value(True).run()
    at.button(key="compare_button").click().run()
    assert not at.exception

    before = at.session_state["tt_session"].policy

    for _ in range(3):
        at.button(key="fb_incorrect").click().run()
        assert not at.exception

    session = at.session_state["tt_session"]
    assert session.policy == before
    assert len(session.adaptation_history) == 3
    assert all("no update" in entry for entry in session.adaptation_history)


def test_accept_can_cautiously_move_the_policy_more_aggressive(monkeypatch):
    monkeypatch.setattr(OpenAICompatibleClient, "generate", _fake_generate)

    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()
    at.text_input(key="api_key_input").set_value("test-key").run()
    at.text_input(key="query_input").set_value("How do I reset my password?").run()
    at.toggle(key="calibration_enabled").set_value(True).run()
    at.button(key="compare_button").click().run()
    assert not at.exception

    at.button(key="fb_accept").click().run()
    assert not at.exception

    session = at.session_state["tt_session"]
    assert len(session.adaptation_history) == 1
    # a single accept only accrues toward the success streak; it must not
    # move the threshold by itself (matches SUCCESS_STREAK_FOR_RELAXATION),
    # but it must still be visibly acknowledged in the history
    assert "repeated success" not in session.adaptation_history[-1]
    assert "success recorded" in session.adaptation_history[-1]


def test_retry_with_full_context_records_a_definite_history_entry(monkeypatch):
    monkeypatch.setattr(OpenAICompatibleClient, "generate", _fake_generate)

    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()
    at.text_input(key="api_key_input").set_value("test-key").run()
    at.text_input(key="query_input").set_value("How do I reset my password?").run()
    at.toggle(key="calibration_enabled").set_value(True).run()
    at.button(key="compare_button").click().run()
    before_count = len(at.session_state["tt_session"].adaptation_history)

    at.button(key="fb_retry").click().run()
    assert not at.exception

    session = at.session_state["tt_session"]
    assert len(session.adaptation_history) > before_count
    last_entry = session.adaptation_history[-1]
    assert (
        "no update" in last_entry
        or "verified failure" in last_entry
        or "label recorded" in last_entry
    )


def test_mark_chunk_irrelevant_records_an_explicit_negative_label(monkeypatch):
    monkeypatch.setattr(OpenAICompatibleClient, "generate", _fake_generate)

    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()
    at.text_input(key="api_key_input").set_value("test-key").run()
    at.text_input(key="query_input").set_value("How do I reset my password?").run()
    at.button(key="compare_button").click().run()
    assert not at.exception

    at.selectbox(key="fb_mark_chunk_select")
    at.button(key="fb_mark_irrelevant").click().run()
    assert not at.exception

    session = at.session_state["tt_session"]
    assert len(session.labels) == 1
    assert session.labels[0].relevant is False
