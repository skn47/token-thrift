from tokenthrift.feedback.attribution import (
    PROVENANCE_COUNTERFACTUAL,
    PROVENANCE_EXPLICIT,
    attribute,
)
from tokenthrift.feedback.events import MarkChunkIrrelevant, RetryFullContext


def test_explicit_mark_is_distinct_from_an_inferred_counterfactual_label():
    explicit_decision = attribute(MarkChunkIrrelevant(chunk_id="c1"))
    inferred_decision = attribute(
        RetryFullContext(improved=True, cited_restored_chunk_ids=frozenset({"c2"})))

    explicit_label = explicit_decision.labels[0]
    inferred_label = inferred_decision.labels[0]

    assert explicit_label.provenance == PROVENANCE_EXPLICIT
    assert inferred_label.provenance == PROVENANCE_COUNTERFACTUAL
    assert explicit_label.relevant is False
    assert inferred_label.relevant is True
    assert explicit_label.provenance != inferred_label.provenance


def test_explicit_mark_produces_exactly_one_negative_label_for_that_chunk():
    decision = attribute(MarkChunkIrrelevant(chunk_id="status-codes::c1"))
    assert len(decision.labels) == 1
    label = decision.labels[0]
    assert label.chunk_id == "status-codes::c1"
    assert label.relevant is False
    assert label.weight > 0
