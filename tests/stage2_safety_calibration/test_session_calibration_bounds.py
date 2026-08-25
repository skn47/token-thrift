from tokenthrift.safety.policy import default_policy
from tokenthrift.session.calibration import (
    MAX_MIN_CONTEXT,
    MAX_THRESHOLD,
    MAX_TOKEN_BUDGET,
    MIN_THRESHOLD,
    SUCCESS_STREAK_FOR_RELAXATION,
    CalibrationState,
)


def test_repeated_verified_failures_move_toward_but_never_past_conservative_bounds():
    state = CalibrationState(policy=default_policy())

    for _ in range(200):
        state = state.record_verified_failure()
        assert state.policy.threshold >= MIN_THRESHOLD
        assert state.policy.min_context <= MAX_MIN_CONTEXT
        assert state.policy.token_budget <= MAX_TOKEN_BUDGET

    assert state.policy.threshold == MIN_THRESHOLD
    assert state.policy.min_context == MAX_MIN_CONTEXT
    assert state.policy.token_budget == MAX_TOKEN_BUDGET


def test_repeated_successes_move_toward_but_never_past_aggressive_threshold_bound():
    state = CalibrationState(policy=default_policy())

    for _ in range(200):
        state = state.record_success()
        assert state.policy.threshold <= MAX_THRESHOLD

    assert state.policy.threshold == MAX_THRESHOLD


def test_a_single_success_does_not_move_the_policy_before_the_streak_threshold():
    state = CalibrationState(policy=default_policy())
    before = state.policy

    for _ in range(SUCCESS_STREAK_FOR_RELAXATION - 1):
        state = state.record_success()
        assert state.policy == before


def test_a_verified_failure_resets_an_in_progress_success_streak():
    state = CalibrationState(policy=default_policy())
    state = state.record_success()
    assert state.consecutive_successes == 1

    state = state.record_verified_failure()
    assert state.consecutive_successes == 0
