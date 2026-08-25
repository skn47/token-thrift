from __future__ import annotations

from dataclasses import dataclass

from tokenthrift.core.types import Chunk
from tokenthrift.validation.coverage import CoverageResult, check_coverage
from tokenthrift.validation.grounding import GroundingResult, ground_answer
from tokenthrift.validation.structured import StructuredCheckResult, check_structured


@dataclass(frozen=True)
class ValidationResult:
    structured: StructuredCheckResult
    coverage: CoverageResult
    grounding: GroundingResult


def run_layered_check(
    query: str, answer_text: str, context_chunks: list[Chunk],
) -> ValidationResult:
    """Runs the first three (fast, always-on) layers of the answer check.
    The fourth layer — counterfactual comparison — only runs after negative
    feedback triggers a full-context retry; see validation/counterfactual.py.
    """
    return ValidationResult(
        structured=check_structured(answer_text, context_chunks),
        coverage=check_coverage(query, answer_text),
        grounding=ground_answer(answer_text, context_chunks),
    )
