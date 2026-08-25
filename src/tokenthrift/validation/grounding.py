from __future__ import annotations

from dataclasses import dataclass

from tokenthrift.core.text import content_tokens, split_sentences, token_overlap_ratio
from tokenthrift.core.types import Chunk

SENTENCE_SUPPORT_THRESHOLD = 0.3
ANSWER_GROUNDED_THRESHOLD = 0.5


@dataclass(frozen=True)
class GroundingResult:
    total_sentences: int
    supported_sentences: int
    supported_ratio: float
    chunk_support: dict[str, tuple[str, ...]]

    @property
    def is_grounded(self) -> bool:
        if self.total_sentences == 0:
            return False
        return self.supported_ratio >= ANSWER_GROUNDED_THRESHOLD


def ground_answer(answer_text: str, chunks: list[Chunk]) -> GroundingResult:
    """Maps each answer sentence to its best-supporting retained chunk by
    content-token overlap. This establishes the answer is grounded in
    retained text, not that any individual claim is factually correct."""
    sentences = split_sentences(answer_text)
    chunk_tokens = {c.chunk_id: set(content_tokens(c.text)) for c in chunks}

    supported = 0
    chunk_support: dict[str, list[str]] = {}
    for sentence in sentences:
        sentence_tokens = set(content_tokens(sentence))
        best_chunk_id: str | None = None
        best_ratio = 0.0
        for chunk_id, tokens in chunk_tokens.items():
            ratio = token_overlap_ratio(sentence_tokens, tokens)
            if ratio > best_ratio:
                best_ratio = ratio
                best_chunk_id = chunk_id
        if best_chunk_id is not None and best_ratio >= SENTENCE_SUPPORT_THRESHOLD:
            supported += 1
            chunk_support.setdefault(best_chunk_id, []).append(sentence)

    total = len(sentences)
    ratio = supported / total if total else 0.0
    return GroundingResult(
        total_sentences=total,
        supported_sentences=supported,
        supported_ratio=ratio,
        chunk_support={cid: tuple(s) for cid, s in chunk_support.items()},
    )
