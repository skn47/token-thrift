from __future__ import annotations

from tokenthrift.safety.policy import default_policy
from tokenthrift.session.calibration import CalibrationState


def reset_to_base() -> CalibrationState:
    """Restores the exact validated default policy, discarding any
    accrued calibration movement and success streak."""
    return CalibrationState(policy=default_policy())
