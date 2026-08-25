from __future__ import annotations

from dataclasses import dataclass

SOURCE_TYPES = ("prose", "code", "table", "other")


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    chunk_id: str
    text: str
    source_type: str
    doc_title: str
    heading: str | None
    position: int
    doc_chunk_count: int


@dataclass(frozen=True)
class RankedChunk:
    chunk: Chunk
    retrieval_score: float
    retrieval_rank: int


@dataclass(frozen=True)
class FeatureVector:
    feature_version: str
    tfidf_similarity: float
    retrieval_rank: int
    score_margin: float
    query_token_overlap_ratio: float
    exact_phrase_overlap: float
    title_or_heading_overlap: float
    entity_overlap: float
    normalized_chunk_length: float
    source_type: str
    neighbor_relevance: float

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "tfidf_similarity": self.tfidf_similarity,
            "retrieval_rank": self.retrieval_rank,
            "score_margin": self.score_margin,
            "query_token_overlap_ratio": self.query_token_overlap_ratio,
            "exact_phrase_overlap": self.exact_phrase_overlap,
            "title_or_heading_overlap": self.title_or_heading_overlap,
            "entity_overlap": self.entity_overlap,
            "normalized_chunk_length": self.normalized_chunk_length,
            "source_type": self.source_type,
            "neighbor_relevance": self.neighbor_relevance,
        }


NUMERIC_FEATURE_COLUMNS = (
    "tfidf_similarity",
    "retrieval_rank",
    "score_margin",
    "query_token_overlap_ratio",
    "exact_phrase_overlap",
    "title_or_heading_overlap",
    "entity_overlap",
    "normalized_chunk_length",
    "neighbor_relevance",
)
CATEGORICAL_FEATURE_COLUMNS = ("source_type",)


@dataclass(frozen=True)
class Policy:
    preset_name: str
    threshold: float
    min_context: int
    token_budget: int


@dataclass(frozen=True)
class ChunkDecision:
    chunk: Chunk
    retrieval_rank: int
    relevance_score: float | None
    kept: bool
    reasons: tuple[str, ...]
    safety_override: str | None
    borderline: bool = False


@dataclass(frozen=True)
class PruningResult:
    query: str
    retained: tuple[ChunkDecision, ...]
    pruned: tuple[ChunkDecision, ...]
    policy: Policy
    pruning_enabled: bool
    disabled_reason: str | None
    retained_tokens: int
    baseline_tokens: int
    model_version: str | None
    budget_conflict: bool = False

    @property
    def tokens_saved(self) -> int:
        return max(self.baseline_tokens - self.retained_tokens, 0)

    @property
    def pct_tokens_saved(self) -> float:
        if self.baseline_tokens == 0:
            return 0.0
        return self.tokens_saved / self.baseline_tokens
