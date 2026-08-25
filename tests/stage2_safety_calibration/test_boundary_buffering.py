from tokenthrift.core.types import Chunk, Policy, RankedChunk
from tokenthrift.safety.rules import BOUNDARY_BUFFER_OVERRIDE, apply_safety_rules


def _chunk(chunk_id, doc_id="d1", position=0, text="text"):
    return Chunk(
        doc_id=doc_id, chunk_id=chunk_id, text=text, source_type="prose",
        doc_title="Doc", heading=None, position=position, doc_chunk_count=3,
    )


def test_score_within_buffer_margin_is_retained_and_marked_borderline():
    c0 = _chunk("c0", position=0)
    c1 = _chunk("c1", position=1)
    c2 = _chunk("c2", position=2)
    ranked = [
        RankedChunk(chunk=c0, retrieval_score=0.9, retrieval_rank=0),
        RankedChunk(chunk=c1, retrieval_score=0.5, retrieval_rank=1),
        RankedChunk(chunk=c2, retrieval_score=0.1, retrieval_rank=2),
    ]
    policy = Policy(preset_name="custom", threshold=0.5, min_context=1, token_budget=10_000)
    # c1 is 0.03 below threshold — inside the 0.05 boundary buffer margin
    scores = {"c0": 0.9, "c1": 0.47, "c2": 0.1}

    retained, _pruned, _ = apply_safety_rules(ranked, scores, policy)
    retained_by_id = {d.chunk.chunk_id: d for d in retained}

    assert "c1" in retained_by_id
    assert retained_by_id["c1"].borderline is True
    assert retained_by_id["c1"].safety_override == BOUNDARY_BUFFER_OVERRIDE


def test_score_below_buffer_margin_is_still_pruned():
    # c1 lives in a different document from c0 so it is not also pulled in
    # by neighbor preservation — this isolates the boundary-buffer rule.
    c0 = _chunk("c0", doc_id="d1", position=0)
    c1 = _chunk("c1", doc_id="d2", position=0)
    ranked = [
        RankedChunk(chunk=c0, retrieval_score=0.9, retrieval_rank=0),
        RankedChunk(chunk=c1, retrieval_score=0.2, retrieval_rank=1),
    ]
    policy = Policy(preset_name="custom", threshold=0.5, min_context=1, token_budget=10_000)
    scores = {"c0": 0.9, "c1": 0.2}

    _retained, pruned, _ = apply_safety_rules(ranked, scores, policy)

    assert "c1" in {d.chunk.chunk_id for d in pruned}
