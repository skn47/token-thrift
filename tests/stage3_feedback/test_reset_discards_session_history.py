from tokenthrift.feedback.attribution import attribute
from tokenthrift.feedback.events import MarkChunkIrrelevant, RetryFullContext
from tokenthrift.safety.policy import default_policy
from tokenthrift.session.calibration import CalibrationState
from tokenthrift.session.state import SessionState


def test_reset_discards_accrued_labels_and_the_cloned_model_slot():
    session = SessionState(
        calibration=CalibrationState(policy=default_policy()), model_version="v1")

    session.apply_adaptation(attribute(MarkChunkIrrelevant(chunk_id="c1")))
    session.apply_adaptation(attribute(RetryFullContext(
        improved=True, cited_restored_chunk_ids=frozenset({"c2"}))))
    assert len(session.labels) == 2
    assert session.policy != default_policy()

    session.cloned_model = object()  # stand-in for a Stage 4 clone
    session.reset()

    assert session.labels == []
    assert session.cloned_model is None
    assert session.policy == default_policy()
    assert session.adaptation_history[-1] == "reset to base policy"
