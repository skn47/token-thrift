from tokenthrift.feedback.attribution import (
    POLICY_SHIFT_MORE_AGGRESSIVE,
    POLICY_SHIFT_MORE_CONSERVATIVE,
    POLICY_SHIFT_NONE,
    PROVENANCE_COUNTERFACTUAL,
    attribute,
)
from tokenthrift.feedback.events import (
    AcceptAnswer,
    MarkChunkIrrelevant,
    RegenerateSameContext,
    RetryFullContext,
    ThumbsDown,
)


def test_regenerate_with_identical_context_causes_no_update():
    decision = attribute(RegenerateSameContext())
    assert decision.policy_shift == POLICY_SHIFT_NONE
    assert decision.labels == ()
    assert decision.had_effect is False


def test_thumbs_down_alone_is_ambiguous_no_weight_update():
    decision = attribute(ThumbsDown())
    assert decision.policy_shift == POLICY_SHIFT_NONE
    assert decision.labels == ()
    assert decision.had_effect is False


def test_unimproved_full_context_retry_causes_no_update():
    decision = attribute(
        RetryFullContext(improved=False, cited_restored_chunk_ids=frozenset()))
    assert decision.policy_shift == POLICY_SHIFT_NONE
    assert decision.labels == ()
    assert decision.had_effect is False


def test_improved_full_context_retry_shifts_conservative_and_labels_cited_chunks_only():
    decision = attribute(RetryFullContext(
        improved=True, cited_restored_chunk_ids=frozenset({"faq-billing::c4"})))

    assert decision.policy_shift == POLICY_SHIFT_MORE_CONSERVATIVE
    assert len(decision.labels) == 1
    label = decision.labels[0]
    assert label.chunk_id == "faq-billing::c4"
    assert label.relevant is True
    assert label.confidence == "high"
    assert label.provenance == PROVENANCE_COUNTERFACTUAL
    assert decision.had_effect is True


def test_improved_retry_with_multiple_cited_chunks_labels_all_of_them():
    decision = attribute(RetryFullContext(
        improved=True,
        cited_restored_chunk_ids=frozenset({"faq-billing::c3", "faq-billing::c4"})))
    labeled_ids = {label.chunk_id for label in decision.labels}
    assert labeled_ids == {"faq-billing::c3", "faq-billing::c4"}
    assert all(label.relevant for label in decision.labels)


def test_explicit_chunk_mark_creates_a_weighted_negative_label():
    decision = attribute(MarkChunkIrrelevant(chunk_id="pricing-table::c1"))
    assert decision.policy_shift == POLICY_SHIFT_NONE
    assert len(decision.labels) == 1
    label = decision.labels[0]
    assert label.chunk_id == "pricing-table::c1"
    assert label.relevant is False
    assert decision.had_effect is True


def test_accept_grounded_answer_causes_a_small_aggressive_shift():
    decision = attribute(AcceptAnswer(grounded=True))
    assert decision.policy_shift == POLICY_SHIFT_MORE_AGGRESSIVE
    assert decision.labels == ()


def test_accept_ungrounded_answer_causes_no_update():
    decision = attribute(AcceptAnswer(grounded=False))
    assert decision.policy_shift == POLICY_SHIFT_NONE
    assert decision.labels == ()
    assert decision.had_effect is False
