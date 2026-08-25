from __future__ import annotations

from dataclasses import dataclass

from tokenthrift.core.tokenizer import count_tokens
from tokenthrift.core.types import ChunkDecision, Policy, RankedChunk

TOP_RESULT_OVERRIDE = "top_ranked_result"
MIN_CONTEXT_OVERRIDE = "min_context"
BOUNDARY_BUFFER_OVERRIDE = "boundary_buffer"
NEIGHBOR_PRESERVATION_OVERRIDE = "neighbor_preservation"

BOUNDARY_BUFFER_MARGIN = 0.05


@dataclass(frozen=True)
class _Candidate:
    override: str | None
    reasons: tuple[str, ...]
    mandatory: bool
    borderline: bool = False


def apply_safety_rules(
    ranked_chunks: list[RankedChunk],
    scores: dict[str, float],
    policy: Policy,
) -> tuple[list[ChunkDecision], list[ChunkDecision], bool]:
    """Applies the full Stage 2 deterministic safety rules on top of
    classifier scores.

    Only the top-ranked result and the min-context fill are unconditionally
    mandatory (included even if they push the total over budget — that
    conflict is reported via the returned `budget_conflict` flag rather
    than silently truncating any chunk's text). Boundary buffering and
    neighbor preservation are soft upgrades from pruned to a retention
    candidate: they still have to fit the remaining budget like an
    ordinary above-threshold chunk.

    Returns (retained, pruned, budget_conflict).
    """
    ordered = sorted(ranked_chunks, key=lambda rc: rc.retrieval_rank)
    by_position = {(rc.chunk.doc_id, rc.chunk.position): rc for rc in ordered}

    wanted: dict[str, _Candidate] = {}

    if ordered:
        top_id = ordered[0].chunk.chunk_id
        wanted[top_id] = _Candidate(
            override=TOP_RESULT_OVERRIDE, reasons=(TOP_RESULT_OVERRIDE,),
            mandatory=True)

    highly_relevant_ids: list[str] = []
    for rc in ordered:
        chunk_id = rc.chunk.chunk_id
        if scores[chunk_id] >= policy.threshold:
            highly_relevant_ids.append(chunk_id)
            if chunk_id not in wanted:
                wanted[chunk_id] = _Candidate(
                    override=None, reasons=("above_threshold",), mandatory=False)

    for rc in ordered:
        chunk_id = rc.chunk.chunk_id
        if chunk_id in wanted:
            continue
        score = scores[chunk_id]
        if policy.threshold - BOUNDARY_BUFFER_MARGIN <= score < policy.threshold:
            wanted[chunk_id] = _Candidate(
                override=BOUNDARY_BUFFER_OVERRIDE,
                reasons=(BOUNDARY_BUFFER_OVERRIDE,),
                mandatory=False, borderline=True)

    for trigger_id in highly_relevant_ids:
        trigger_chunk = next(rc.chunk for rc in ordered if rc.chunk.chunk_id == trigger_id)
        for offset in (-1, 1):
            neighbor = by_position.get((trigger_chunk.doc_id, trigger_chunk.position + offset))
            if neighbor is None:
                continue
            neighbor_id = neighbor.chunk.chunk_id
            if neighbor_id not in wanted:
                wanted[neighbor_id] = _Candidate(
                    override=NEIGHBOR_PRESERVATION_OVERRIDE,
                    reasons=(NEIGHBOR_PRESERVATION_OVERRIDE,), mandatory=False)

    if len(wanted) < policy.min_context:
        for rc in ordered:
            if len(wanted) >= policy.min_context:
                break
            chunk_id = rc.chunk.chunk_id
            if chunk_id not in wanted:
                wanted[chunk_id] = _Candidate(
                    override=MIN_CONTEXT_OVERRIDE, reasons=(MIN_CONTEXT_OVERRIDE,),
                    mandatory=True)

    retained: list[ChunkDecision] = []
    pruned: list[ChunkDecision] = []
    used_tokens = 0
    budget_conflict = False

    for rc in ordered:
        chunk_id = rc.chunk.chunk_id
        score = scores[chunk_id]
        chunk_tokens = count_tokens(rc.chunk.text)

        if chunk_id not in wanted:
            pruned.append(ChunkDecision(
                chunk=rc.chunk, retrieval_rank=rc.retrieval_rank,
                relevance_score=score, kept=False,
                reasons=("below_threshold",), safety_override=None,
            ))
            continue

        candidate = wanted[chunk_id]
        fits_budget = used_tokens + chunk_tokens <= policy.token_budget
        if fits_budget or candidate.mandatory:
            reasons = candidate.reasons
            if candidate.mandatory and not fits_budget:
                budget_conflict = True
                reasons = (*reasons, "budget_conflict")
            used_tokens += chunk_tokens
            retained.append(ChunkDecision(
                chunk=rc.chunk, retrieval_rank=rc.retrieval_rank,
                relevance_score=score, kept=True, reasons=reasons,
                safety_override=candidate.override, borderline=candidate.borderline,
            ))
        else:
            pruned.append(ChunkDecision(
                chunk=rc.chunk, retrieval_rank=rc.retrieval_rank,
                relevance_score=score, kept=False,
                reasons=("budget_exceeded",), safety_override=None,
            ))

    return retained, pruned, budget_conflict
