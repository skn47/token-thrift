from tokenthrift.core.types import Chunk, Policy, RankedChunk
from tokenthrift.safety.rules import apply_safety_rules


def _chunk(chunk_id, text, position=0):
    return Chunk(
        doc_id="d1", chunk_id=chunk_id, text=text, source_type="prose",
        doc_title="Doc", heading=None, position=position, doc_chunk_count=3,
    )


def test_mandatory_chunk_exceeding_budget_is_kept_whole_and_flagged():
    c0 = _chunk("c0", "a " * 2000)  # alone far bigger than the token budget
    ranked = [RankedChunk(chunk=c0, retrieval_score=0.9, retrieval_rank=0)]
    policy = Policy(preset_name="custom", threshold=0.5, min_context=1, token_budget=1)
    scores = {"c0": 0.9}

    retained, _pruned, budget_conflict = apply_safety_rules(ranked, scores, policy)

    assert budget_conflict is True
    assert len(retained) == 1
    assert retained[0].chunk.chunk_id == "c0"
    assert retained[0].chunk.text == c0.text  # never truncated
    assert "budget_conflict" in retained[0].reasons


def test_non_mandatory_chunk_exceeding_budget_is_pruned_without_a_conflict_flag():
    c0 = _chunk("c0", "short", position=0)
    c1 = _chunk("c1", "b " * 2000, position=5)  # not adjacent to c0
    ranked = [
        RankedChunk(chunk=c0, retrieval_score=0.9, retrieval_rank=0),
        RankedChunk(chunk=c1, retrieval_score=0.6, retrieval_rank=1),
    ]
    policy = Policy(preset_name="custom", threshold=0.5, min_context=1, token_budget=50)
    scores = {"c0": 0.9, "c1": 0.6}

    _retained, pruned, budget_conflict = apply_safety_rules(ranked, scores, policy)

    assert budget_conflict is False
    pruned_c1 = next(d for d in pruned if d.chunk.chunk_id == "c1")
    assert pruned_c1.reasons == ("budget_exceeded",)
