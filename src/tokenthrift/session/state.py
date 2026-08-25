from __future__ import annotations

from dataclasses import dataclass, field

from tokenthrift.core.types import Policy
from tokenthrift.feedback.attribution import AdaptationDecision, ChunkLabel
from tokenthrift.pruner.model import PrunerModel
from tokenthrift.session.calibration import (
    SUCCESS_STREAK_FOR_RELAXATION,
    CalibrationState,
)
from tokenthrift.session.reset import reset_to_base
from tokenthrift.session.sgd_adapter import (
    CanaryExample,
    clone_base_model,
    propose_update,
)


@dataclass
class SessionState:
    """Stage 2 added bounded threshold/min-context/budget calibration on
    top of the Stage 1 static policy. Stage 3 added a label store (fed by
    feedback attribution) and an inert cloned-model slot. Stage 4 turns
    the slot real: labels with an attached feature vector now drive
    guarded partial_fit updates via session.sgd_adapter, bounded and
    canary-checked, never touching the shared base model."""

    calibration: CalibrationState
    model_version: str
    adaptation_history: list[str] = field(default_factory=list)
    labels: list[ChunkLabel] = field(default_factory=list)
    cloned_model: PrunerModel | None = None
    sgd_active: bool = False
    accepted_updates: int = 0
    rejected_updates: int = 0

    @property
    def policy(self) -> Policy:
        return self.calibration.policy

    def record_verified_failure(self) -> None:
        before = self.calibration.policy
        self.calibration = self.calibration.record_verified_failure()
        after = self.calibration.policy
        self.adaptation_history.append(
            "verified failure: more conservative — "
            f"threshold {before.threshold:.2f}->{after.threshold:.2f}, "
            f"min_context {before.min_context}->{after.min_context}, "
            f"token_budget {before.token_budget}->{after.token_budget}")

    def record_success(self) -> None:
        before = self.calibration.policy
        self.calibration = self.calibration.record_success()
        after = self.calibration.policy
        if before != after:
            self.adaptation_history.append(
                "repeated success: cautiously more aggressive — "
                f"threshold {before.threshold:.2f}->{after.threshold:.2f}")
        else:
            # Always log something — an Accept click that only accrues
            # toward the relaxation streak (without yet moving the
            # threshold) must still be visibly acknowledged, or a user
            # sees nothing happen and assumes the click was lost.
            self.adaptation_history.append(
                f"success recorded ({self.calibration.consecutive_successes}/"
                f"{SUCCESS_STREAK_FOR_RELAXATION} toward cautious relaxation)")

    def apply_adaptation(
        self,
        decision: AdaptationDecision,
        base_model: PrunerModel | None = None,
        canary: list[CanaryExample] | None = None,
    ) -> None:
        """Consumes an AdaptationDecision from feedback.attribution.attribute():
        applies any bounded policy shift via the existing calibration
        machinery, records any derived chunk labels to the session-local
        label store, and always leaves a human-readable trace in the
        adaptation history — including when nothing changed, so "why didn't
        this do anything" is answerable from the log.

        `base_model`/`canary` are optional: when omitted (Stage 1-3 call
        sites, most tests), no guarded SGD update is attempted — labels
        still accumulate in the store, matching the Stage 3 scaffolding
        behavior exactly. Passing both activates the real Stage 4 path.
        """
        if decision.policy_shift == "more_conservative":
            self.record_verified_failure()
        elif decision.policy_shift == "more_aggressive":
            self.record_success()

        for label in decision.labels:
            self.labels.append(label)
            self.adaptation_history.append(
                f"label recorded: {label.chunk_id} relevant={label.relevant} "
                f"(confidence={label.confidence}, provenance={label.provenance})")

        if not decision.had_effect:
            self.adaptation_history.append(f"no update: {decision.reason}")

        if base_model is not None and canary is not None:
            self._maybe_update_sgd(decision.labels, base_model, canary)

    def _maybe_update_sgd(
        self, labels: tuple[ChunkLabel, ...], base_model: PrunerModel,
        canary: list[CanaryExample],
    ) -> None:
        usable = [l for l in labels if l.features is not None]
        if not usable:
            return
        if self.cloned_model is None:
            self.cloned_model = clone_base_model(base_model)

        updated_model, outcome = propose_update(
            self.cloned_model, base_model, usable, canary)
        self.cloned_model = updated_model

        if outcome.accepted:
            self.sgd_active = True
            self.accepted_updates += 1
            self.adaptation_history.append(
                "session-local SGD update accepted (coefficient distance "
                f"{outcome.coef_distance:.2f}, canary recall "
                f"{outcome.canary_recall_before:.2f}->{outcome.canary_recall_after:.2f})")
        else:
            self.rejected_updates += 1
            self.adaptation_history.append(
                f"session-local SGD update rejected: {outcome.reason}")

    def reset(self) -> None:
        self.calibration = reset_to_base()
        self.labels = []
        self.cloned_model = None
        self.sgd_active = False
        self.accepted_updates = 0
        self.rejected_updates = 0
        self.adaptation_history.append("reset to base policy")
