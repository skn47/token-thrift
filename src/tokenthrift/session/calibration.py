from __future__ import annotations

from dataclasses import dataclass, replace

from tokenthrift.core.types import Policy

# Bounds a calibrated policy may never cross, regardless of how long a
# feedback stream runs — this is what "repeated bad feedback reaching but
# never exceeding adaptation bounds" means structurally.
MIN_THRESHOLD = 0.15
MAX_THRESHOLD = 0.85
MIN_TOKEN_BUDGET = 500
MAX_TOKEN_BUDGET = 6000
MAX_MIN_CONTEXT = 6

THRESHOLD_STEP_DOWN = 0.05
THRESHOLD_STEP_UP = 0.02
MIN_CONTEXT_STEP_UP = 1
TOKEN_BUDGET_STEP_UP = 200

SUCCESS_STREAK_FOR_RELAXATION = 3


@dataclass(frozen=True)
class CalibrationState:
    policy: Policy
    consecutive_successes: int = 0

    def record_verified_failure(self) -> CalibrationState:
        """A verified pruning failure immediately makes the policy more
        conservative: lower threshold (more chunks clear it), a higher
        min-context floor, and more budget room — and resets the success
        streak, since a failure means the current policy has not proven
        itself yet."""
        new_policy = replace(
            self.policy,
            preset_name="custom",
            threshold=max(MIN_THRESHOLD, self.policy.threshold - THRESHOLD_STEP_DOWN),
            min_context=min(MAX_MIN_CONTEXT, self.policy.min_context + MIN_CONTEXT_STEP_UP),
            token_budget=min(MAX_TOKEN_BUDGET, self.policy.token_budget + TOKEN_BUDGET_STEP_UP),
        )
        return CalibrationState(policy=new_policy, consecutive_successes=0)

    def record_success(self) -> CalibrationState:
        """A single success only accrues toward a streak; only repeated
        successes cautiously permit more pruning (raise the threshold)."""
        streak = self.consecutive_successes + 1
        if streak < SUCCESS_STREAK_FOR_RELAXATION:
            return CalibrationState(policy=self.policy, consecutive_successes=streak)
        new_policy = replace(
            self.policy,
            preset_name="custom",
            threshold=min(MAX_THRESHOLD, self.policy.threshold + THRESHOLD_STEP_UP),
        )
        return CalibrationState(policy=new_policy, consecutive_successes=0)
