from __future__ import annotations

from dataclasses import dataclass, field

from tokenthrift.core.types import FeatureVector


@dataclass(frozen=True)
class RegenerateSameContext:
    """User clicked Regenerate: same retained context, a new generation
    call. Any difference in the answer is attributable to generation
    stochasticity, not to pruning."""


@dataclass(frozen=True)
class ThumbsDown:
    """User clicked Incorrect with no further action. The cause (bad
    generation vs. missing evidence) is ambiguous from this signal alone."""


@dataclass(frozen=True)
class RetryFullContext:
    """User clicked Retry (full context). `improved` and
    `cited_restored_chunk_ids` come from
    validation.counterfactual.compare_counterfactual — improvement is only
    counted when the retry both scores better AND actually cites evidence
    that pruning had dropped. `features_by_chunk_id` carries the feature
    vector actually observed for each cited chunk at this query/turn — the
    only form a Stage 4 partial_fit update can legitimately train on,
    since features are query-relative, not fixed per chunk."""

    improved: bool
    cited_restored_chunk_ids: frozenset[str]
    features_by_chunk_id: dict[str, FeatureVector] = field(default_factory=dict)


@dataclass(frozen=True)
class MarkChunkIrrelevant:
    """User explicitly marked one retained chunk as not relevant — a
    direct chunk-level judgment, independent of the answer's quality.
    `features` is the feature vector observed for this chunk at this
    query/turn, if available."""

    chunk_id: str
    features: FeatureVector | None = None


@dataclass(frozen=True)
class AcceptAnswer:
    """User clicked Accept. `grounded` comes from the layered answer
    check's grounding layer run against the pruned answer."""

    grounded: bool


FeedbackEvent = (
    RegenerateSameContext | ThumbsDown | RetryFullContext
    | MarkChunkIrrelevant | AcceptAnswer
)
