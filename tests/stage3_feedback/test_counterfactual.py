from tokenthrift.core.types import Chunk, ChunkDecision, Policy, PruningResult
from tokenthrift.validation.counterfactual import compare_counterfactual


def _chunk(chunk_id, text, position=0):
    return Chunk(
        doc_id="d1", chunk_id=chunk_id, text=text, source_type="prose",
        doc_title="Doc", heading=None, position=position, doc_chunk_count=3,
    )


def _pruned_result(retained_chunks: list[Chunk]) -> PruningResult:
    retained = tuple(
        ChunkDecision(
            chunk=c, retrieval_rank=i, relevance_score=0.9, kept=True,
            reasons=("above_threshold",), safety_override=None,
        )
        for i, c in enumerate(retained_chunks)
    )
    policy = Policy(preset_name="custom", threshold=0.5, min_context=1, token_budget=2000)
    return PruningResult(
        query="q", retained=retained, pruned=(), policy=policy,
        pruning_enabled=True, disabled_reason=None,
        retained_tokens=10, baseline_tokens=20, model_version="v1",
    )


def test_restored_answer_citing_a_pruned_chunk_counts_as_improved():
    kept = _chunk("c0", "Grace period of 7 days applies after a failed payment.")
    pruned_away = _chunk("c1", "Data is never deleted, projects become read-only over the limit.")

    pruned_result = _pruned_result([kept])
    pruned_answer = "There is a 7 day grace period after a failed payment."
    restored_answer = (
        "There is a 7 day grace period after a failed payment. "
        "Data is never deleted, projects become read-only over the limit."
    )

    cf = compare_counterfactual(pruned_answer, pruned_result, restored_answer, [kept, pruned_away])

    assert cf.improved is True
    assert cf.cited_restored_chunk_ids == frozenset({"c1"})
    # both are fully grounded (ratio saturates at 1.0) — the restored
    # answer is longer and cites genuinely new evidence, which is what
    # actually signals improvement here, not a higher ratio
    assert cf.restored_grounded_ratio >= cf.pruned_grounded_ratio


def test_equally_grounded_restored_answer_is_not_counted_as_improved():
    kept = _chunk("c0", "Grace period of 7 days applies after a failed payment.")
    other = _chunk("c1", "Unrelated content about something else entirely.")

    pruned_result = _pruned_result([kept])
    same_answer = "There is a 7 day grace period after a failed payment."

    cf = compare_counterfactual(same_answer, pruned_result, same_answer, [kept, other])

    assert cf.improved is False
    assert cf.cited_restored_chunk_ids == frozenset()


def test_better_grounded_restored_answer_without_citing_new_evidence_is_not_improved():
    kept = _chunk("c0", "Grace period of 7 days applies after a failed payment.")
    pruned_result = _pruned_result([kept])

    pruned_answer = "Not sure about the grace period."
    # restored answer is better grounded, but only re-states what was
    # already available in the retained set — no previously-pruned chunk
    # is cited, so this is not evidence pruning caused the failure
    restored_answer = "There is a 7 day grace period after a failed payment."

    cf = compare_counterfactual(pruned_answer, pruned_result, restored_answer, [kept])

    assert cf.cited_restored_chunk_ids == frozenset()
    assert cf.improved is False
