from __future__ import annotations

from dataclasses import dataclass

from tokenthrift.core.text import content_tokens, token_overlap_ratio

COVERAGE_THRESHOLD = 0.3


@dataclass(frozen=True)
class CoverageResult:
    covered_ratio: float
    covered: bool


def check_coverage(query: str, answer_text: str) -> CoverageResult:
    """Heuristic estimate of whether the answer addresses the material
    parts of the query: the fraction of the query's non-stopword tokens
    that also appear somewhere in the answer. A paraphrased answer will
    score lower than a literal one — this is a coarse proxy, not a
    semantic entailment check."""
    query_tokens = set(content_tokens(query))
    answer_tokens = set(content_tokens(answer_text))
    ratio = token_overlap_ratio(query_tokens, answer_tokens)
    return CoverageResult(covered_ratio=ratio, covered=ratio >= COVERAGE_THRESHOLD)
