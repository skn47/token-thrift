from __future__ import annotations

from dataclasses import dataclass

from tokenthrift.core.types import FeatureVector
from tokenthrift.feedback.events import (
    AcceptAnswer,
    FeedbackEvent,
    MarkChunkIrrelevant,
    RegenerateSameContext,
    RetryFullContext,
    ThumbsDown,
)

POLICY_SHIFT_NONE = "none"
POLICY_SHIFT_MORE_CONSERVATIVE = "more_conservative"
POLICY_SHIFT_MORE_AGGRESSIVE = "more_aggressive"

PROVENANCE_EXPLICIT = "explicit_user_mark"
PROVENANCE_COUNTERFACTUAL = "counterfactual_restore"


@dataclass(frozen=True)
class ChunkLabel:
    chunk_id: str
    relevant: bool
    weight: float
    confidence: str  # "high" | "low"
    provenance: str
    features: FeatureVector | None = None


@dataclass(frozen=True)
class AdaptationDecision:
    policy_shift: str
    labels: tuple[ChunkLabel, ...]
    reason: str

    @property
    def had_effect(self) -> bool:
        return self.policy_shift != POLICY_SHIFT_NONE or bool(self.labels)


def attribute(event: FeedbackEvent) -> AdaptationDecision:
    """Implements the feedback-attribution table from the reference doc as
    explicit branching logic — the one place adaptation decisions get made,
    so nothing downstream has to (re-)infer intent from raw UI events.
    Absence of feedback is never represented here at all: every branch that
    reaches this function corresponds to an explicit user action."""
    if isinstance(event, RegenerateSameContext):
        return AdaptationDecision(
            policy_shift=POLICY_SHIFT_NONE, labels=(),
            reason="regenerated with identical context — the generator may "
                   "have been stochastic or unsatisfactory, not a pruning "
                   "signal")

    if isinstance(event, ThumbsDown):
        return AdaptationDecision(
            policy_shift=POLICY_SHIFT_NONE, labels=(),
            reason="incorrect feedback alone is ambiguous — the cause "
                   "could be generation or pruning; recorded without a "
                   "weight update")

    if isinstance(event, RetryFullContext):
        if not event.improved:
            return AdaptationDecision(
                policy_shift=POLICY_SHIFT_NONE, labels=(),
                reason="full-context retry did not improve the answer — "
                       "pruning was probably not the cause")
        labels = tuple(
            ChunkLabel(
                chunk_id=chunk_id, relevant=True, weight=1.0,
                confidence="high", provenance=PROVENANCE_COUNTERFACTUAL,
                features=event.features_by_chunk_id.get(chunk_id))
            for chunk_id in sorted(event.cited_restored_chunk_ids)
        )
        return AdaptationDecision(
            policy_shift=POLICY_SHIFT_MORE_CONSERVATIVE, labels=labels,
            reason="full-context retry improved the answer using restored "
                   "evidence — one or more required chunks were wrongly "
                   "pruned")

    if isinstance(event, MarkChunkIrrelevant):
        label = ChunkLabel(
            chunk_id=event.chunk_id, relevant=False, weight=1.0,
            confidence="high", provenance=PROVENANCE_EXPLICIT,
            features=event.features)
        return AdaptationDecision(
            policy_shift=POLICY_SHIFT_NONE, labels=(label,),
            reason=f"user explicitly marked chunk {event.chunk_id!r} irrelevant")

    if isinstance(event, AcceptAnswer):
        if event.grounded:
            return AdaptationDecision(
                policy_shift=POLICY_SHIFT_MORE_AGGRESSIVE, labels=(),
                reason="user accepted a grounded answer — weak evidence "
                       "the policy worked; at most a small calibration "
                       "nudge")
        return AdaptationDecision(
            policy_shift=POLICY_SHIFT_NONE, labels=(),
            reason="user accepted an answer the layered check could not "
                   "confirm was grounded — no adaptation")

    raise TypeError(f"unhandled feedback event type: {type(event)!r}")
