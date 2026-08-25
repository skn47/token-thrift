import pytest

from tokenthrift.core.types import Policy
from tokenthrift.eval.metrics import (
    expected_calibration_error,
    mean_recovery_trials,
    policy_movement,
    recovery_trial_counts,
)


def test_expected_calibration_error_is_zero_for_perfectly_calibrated_scores():
    y_true = [1, 1, 0, 0, 1, 0]
    y_scores = [1.0, 1.0, 0.0, 0.0, 1.0, 0.0]
    assert expected_calibration_error(y_true, y_scores) == 0.0


def test_expected_calibration_error_is_positive_for_overconfident_scores():
    y_true = [0, 0, 0, 0]
    y_scores = [0.9, 0.9, 0.9, 0.9]
    assert expected_calibration_error(y_true, y_scores) > 0.5


def test_policy_movement_reports_signed_deltas():
    start = Policy(preset_name="balanced", threshold=0.5, min_context=2, token_budget=2000)
    end = Policy(preset_name="custom", threshold=0.35, min_context=3, token_budget=2200)

    movement = policy_movement(start, end)

    assert movement["threshold_delta"] == pytest.approx(-0.15)
    assert movement["min_context_delta"] == 1.0
    assert movement["token_budget_delta"] == 200.0


def test_recovery_trial_counts_measures_trials_to_return_to_full_recall():
    recalls = [1.0, 1.0, 0.5, 0.5, 1.0, 0.8, 1.0, 1.0]
    counts = recovery_trial_counts(recalls)
    # first failure streak: indices 2-3 (0.5, 0.5), recovers at index 4 -> 2 trials
    # second failure streak: index 5 (0.8), recovers at index 6 -> 1 trial
    assert counts == [2, 1]
    assert mean_recovery_trials(recalls) == 1.5


def test_recovery_trial_counts_excludes_a_failure_that_never_recovers():
    recalls = [1.0, 0.5, 0.5]
    assert recovery_trial_counts(recalls) == []
    assert mean_recovery_trials(recalls) is None


def test_mean_recovery_trials_is_none_when_there_are_no_failures():
    assert mean_recovery_trials([1.0, 1.0, 1.0]) is None
