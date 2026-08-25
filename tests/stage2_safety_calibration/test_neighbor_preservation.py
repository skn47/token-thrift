from tokenthrift.core.types import Chunk, Policy, RankedChunk
from tokenthrift.safety.rules import NEIGHBOR_PRESERVATION_OVERRIDE, apply_safety_rules


def _chunk(chunk_id, doc_id, position, text="text"):
    return Chunk(
        doc_id=doc_id, chunk_id=chunk_id, text=text, source_type="prose",
        doc_title="FAQ", heading=None, position=position, doc_chunk_count=6,
    )


def test_neighbor_of_a_highly_relevant_chunk_is_retained_when_it_fits_budget():
    # Models the "relevant fact split across neighboring chunks" scenario:
    # the trigger chunk alone clears the threshold, but the answer's other
    # half lives in the adjacent chunk, which scores low on its own.
    trigger = _chunk("faq::c3", "faq", position=3)
    neighbor = _chunk("faq::c4", "faq", position=4)
    unrelated = _chunk("other::c0", "other", position=0)

    ranked = [
        RankedChunk(chunk=trigger, retrieval_score=0.8, retrieval_rank=0),
        RankedChunk(chunk=unrelated, retrieval_score=0.4, retrieval_rank=1),
        RankedChunk(chunk=neighbor, retrieval_score=0.05, retrieval_rank=2),
    ]
    policy = Policy(preset_name="custom", threshold=0.6, min_context=1, token_budget=10_000)
    scores = {"faq::c3": 0.8, "other::c0": 0.1, "faq::c4": 0.05}

    retained, pruned, _ = apply_safety_rules(ranked, scores, policy)
    retained_by_id = {d.chunk.chunk_id: d for d in retained}

    assert "faq::c4" in retained_by_id
    assert retained_by_id["faq::c4"].safety_override == NEIGHBOR_PRESERVATION_OVERRIDE
    assert "faq::c4" not in {d.chunk.chunk_id for d in pruned}


def test_neighbor_from_a_different_source_is_not_preserved():
    trigger = _chunk("faq::c3", "faq", position=3)
    other_doc_chunk = _chunk("guide::c0", "guide", position=4)  # same position, different doc

    ranked = [
        RankedChunk(chunk=trigger, retrieval_score=0.8, retrieval_rank=0),
        RankedChunk(chunk=other_doc_chunk, retrieval_score=0.05, retrieval_rank=1),
    ]
    policy = Policy(preset_name="custom", threshold=0.6, min_context=1, token_budget=10_000)
    scores = {"faq::c3": 0.8, "guide::c0": 0.05}

    _retained, pruned, _ = apply_safety_rules(ranked, scores, policy)

    assert "guide::c0" in {d.chunk.chunk_id for d in pruned}


def test_neighbor_preservation_still_respects_the_token_budget():
    trigger = _chunk("faq::c3", "faq", position=3)
    neighbor = _chunk("faq::c4", "faq", position=4)

    ranked = [
        RankedChunk(chunk=trigger, retrieval_score=0.8, retrieval_rank=0),
        RankedChunk(chunk=neighbor, retrieval_score=0.05, retrieval_rank=1),
    ]
    policy = Policy(preset_name="custom", threshold=0.6, min_context=1, token_budget=1)
    scores = {"faq::c3": 0.8, "faq::c4": 0.05}

    retained, pruned, _ = apply_safety_rules(ranked, scores, policy)
    retained_ids = {d.chunk.chunk_id for d in retained}
    pruned_ids = {d.chunk.chunk_id for d in pruned}

    assert "faq::c3" in retained_ids  # mandatory top-result, kept regardless of budget
    assert "faq::c4" in pruned_ids  # neighbor candidate must still respect the budget
