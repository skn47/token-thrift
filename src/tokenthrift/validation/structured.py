from __future__ import annotations

import re
from dataclasses import dataclass

from tokenthrift.core.types import Chunk

# Numbers (including bare HTTP-status-style 3-digit codes, percentages, and
# decimals) and Lighthouse-style identifiers (tsk_xxxx, prj_xxxx, usr_xxxx).
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?%?\b")
_IDENTIFIER_RE = re.compile(r"\b[a-z]{2,4}_[a-z0-9]{3,}\b", re.IGNORECASE)


@dataclass(frozen=True)
class StructuredCheckResult:
    applicable: bool
    passed: bool
    unsupported_numbers: tuple[str, ...]
    unsupported_identifiers: tuple[str, ...]


def check_structured(answer_text: str, context_chunks: list[Chunk]) -> StructuredCheckResult:
    """Deterministic check: any number or identifier-looking token the
    answer states must appear somewhere in the retained context. This
    cannot prove correctness, only catch outright fabricated identifiers
    or numbers — it applies only when the answer actually contains such
    tokens at all."""
    context_text = "\n".join(c.text for c in context_chunks)
    context_numbers = set(_NUMBER_RE.findall(context_text))
    context_identifiers = {m.lower() for m in _IDENTIFIER_RE.findall(context_text)}

    answer_numbers = set(_NUMBER_RE.findall(answer_text))
    answer_identifiers = {m.lower() for m in _IDENTIFIER_RE.findall(answer_text)}

    if not answer_numbers and not answer_identifiers:
        return StructuredCheckResult(
            applicable=False, passed=True,
            unsupported_numbers=(), unsupported_identifiers=())

    unsupported_numbers = tuple(sorted(answer_numbers - context_numbers))
    unsupported_identifiers = tuple(sorted(answer_identifiers - context_identifiers))

    return StructuredCheckResult(
        applicable=True,
        passed=not unsupported_numbers and not unsupported_identifiers,
        unsupported_numbers=unsupported_numbers,
        unsupported_identifiers=unsupported_identifiers,
    )
