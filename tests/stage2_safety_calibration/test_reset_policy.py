from tokenthrift.safety.policy import default_policy
from tokenthrift.session.calibration import CalibrationState
from tokenthrift.session.reset import reset_to_base
from tokenthrift.session.state import SessionState


def test_reset_restores_the_exact_default_policy_after_drift():
    state = CalibrationState(policy=default_policy())
    for _ in range(20):
        state = state.record_verified_failure()
    assert state.policy != default_policy()

    reset_state = reset_to_base()
    assert reset_state.policy == default_policy()
    assert reset_state.consecutive_successes == 0


def test_session_reset_clears_adaptation_history_and_calibration_drift():
    session = SessionState(calibration=CalibrationState(policy=default_policy()), model_version="v1")
    session.record_verified_failure()
    session.record_verified_failure()
    assert session.policy != default_policy()
    assert session.adaptation_history

    session.reset()

    assert session.policy == default_policy()
    assert session.adaptation_history[-1] == "reset to base policy"
