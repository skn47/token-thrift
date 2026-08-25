from tokenthrift.feedback.attribution import attribute
from tokenthrift.feedback.events import (
    AcceptAnswer,
    RegenerateSameContext,
    RetryFullContext,
    ThumbsDown,
)


def test_no_feedback_event_without_direct_evidence_ever_produces_a_label():
    events_without_direct_evidence = [
        RegenerateSameContext(),
        ThumbsDown(),
        RetryFullContext(improved=False, cited_restored_chunk_ids=frozenset()),
        AcceptAnswer(grounded=False),
        AcceptAnswer(grounded=True),  # acceptance is weak evidence for calibration only, never a chunk label
    ]
    for event in events_without_direct_evidence:
        decision = attribute(event)
        assert decision.labels == (), f"{event!r} unexpectedly produced labels"


def test_retry_labels_only_chunks_actually_cited_not_every_restored_chunk():
    decision = attribute(RetryFullContext(
        improved=True, cited_restored_chunk_ids=frozenset({"faq-account::c3"})))
    labeled_ids = {label.chunk_id for label in decision.labels}
    assert labeled_ids == {"faq-account::c3"}
