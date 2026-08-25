from __future__ import annotations

from dataclasses import dataclass

from tokenthrift.core.types import Chunk, PruningResult
from tokenthrift.validation.grounding import ground_answer


@dataclass(frozen=True)
class CounterfactualResult:
    improved: bool
    pruned_grounded_ratio: float
    restored_grounded_ratio: float
    cited_restored_chunk_ids: frozenset[str]


def compare_counterfactual(
    pruned_answer_text: str,
    pruned_result: PruningResult,
    restored_answer_text: str,
    full_context_chunks: list[Chunk],
) -> CounterfactualResult:
    """Compares the original pruned-context answer against a full-context
    retry. The only evidence that pruning caused the original failure is
    both of: the restored answer is at least as well grounded, AND it
    actually cites a chunk that pruning had dropped — a differently-worded
    but equally (un)grounded answer that cites nothing new proves nothing
    about the pruning decision. Grounding ratio alone can saturate at 1.0
    for an already-fully-grounded pruned answer, so a strictly-greater
    comparison would miss the case where the retry adds genuinely new,
    cited evidence without changing the ratio — hence >=, not >."""
    retained_ids = {d.chunk.chunk_id for d in pruned_result.retained}
    retained_chunks = [d.chunk for d in pruned_result.retained]

    pruned_grounding = ground_answer(pruned_answer_text, retained_chunks)
    restored_grounding = ground_answer(restored_answer_text, full_context_chunks)

    cited_restored_chunk_ids = frozenset(
        chunk_id for chunk_id in restored_grounding.chunk_support
        if chunk_id not in retained_ids
    )

    improved = (
        restored_grounding.supported_ratio >= pruned_grounding.supported_ratio
        and bool(cited_restored_chunk_ids)
    )

    return CounterfactualResult(
        improved=improved,
        pruned_grounded_ratio=pruned_grounding.supported_ratio,
        restored_grounded_ratio=restored_grounding.supported_ratio,
        cited_restored_chunk_ids=cited_restored_chunk_ids,
    )
