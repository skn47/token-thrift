import numpy as np

from tokenthrift.feedback.attribution import attribute
from tokenthrift.feedback.events import MarkChunkIrrelevant, RetryFullContext
from tokenthrift.safety.policy import default_policy
from tokenthrift.session.calibration import CalibrationState
from tokenthrift.session.sgd_adapter import build_canary_set, clone_base_model
from tokenthrift.session.state import SessionState

from ._helpers import base_model, real_labeled_examples


def _fresh_session() -> SessionState:
    return SessionState(
        calibration=CalibrationState(policy=default_policy()), model_version="v1")


def test_improved_retry_produces_a_real_bounded_weight_update():
    base = base_model()
    canary = build_canary_set()
    session = _fresh_session()

    positives = real_labeled_examples(relevant=True, n=4)
    features_by_chunk_id = {l.chunk_id: l.features for l in positives}
    decision = attribute(RetryFullContext(
        improved=True,
        cited_restored_chunk_ids=frozenset(features_by_chunk_id.keys()),
        features_by_chunk_id=features_by_chunk_id))

    session.apply_adaptation(decision, base_model=base, canary=canary)

    assert session.cloned_model is not None
    assert session.sgd_active is True
    assert session.accepted_updates == 1
    cloned_clf = session.cloned_model.pipeline.named_steps["classifier"]
    base_clf = base.pipeline.named_steps["classifier"]
    assert not np.array_equal(cloned_clf.coef_, base_clf.coef_)
    assert any(
        "SGD update accepted" in entry for entry in session.adaptation_history)


def test_apply_adaptation_without_base_or_canary_never_attempts_sgd():
    session = _fresh_session()
    positives = real_labeled_examples(relevant=True, n=2)
    features_by_chunk_id = {l.chunk_id: l.features for l in positives}
    decision = attribute(RetryFullContext(
        improved=True, cited_restored_chunk_ids=frozenset(features_by_chunk_id.keys()),
        features_by_chunk_id=features_by_chunk_id))

    session.apply_adaptation(decision)  # no base_model/canary — Stage 1-3 call shape

    assert session.cloned_model is None
    assert session.sgd_active is False
    assert len(session.labels) == 2  # labels still recorded either way


def test_adversarial_mark_irrelevant_events_get_rejected_via_the_full_session_path():
    base = base_model()
    canary = build_canary_set()
    session = _fresh_session()

    adversarial = real_labeled_examples(relevant=False, n=6)
    for label in adversarial:
        decision = attribute(
            MarkChunkIrrelevant(chunk_id=label.chunk_id, features=label.features))
        session.apply_adaptation(decision, base_model=base, canary=canary)

    assert session.rejected_updates >= 1
    assert any(
        "SGD update rejected" in entry for entry in session.adaptation_history)


def test_reset_discards_sgd_weights_exactly_matching_base_not_merely_close():
    base = base_model()
    canary = build_canary_set()
    session = _fresh_session()

    positives = real_labeled_examples(relevant=True, n=4)
    features_by_chunk_id = {l.chunk_id: l.features for l in positives}
    decision = attribute(RetryFullContext(
        improved=True, cited_restored_chunk_ids=frozenset(features_by_chunk_id.keys()),
        features_by_chunk_id=features_by_chunk_id))
    session.apply_adaptation(decision, base_model=base, canary=canary)
    assert session.cloned_model is not None

    session.reset()

    assert session.cloned_model is None
    assert session.sgd_active is False
    assert session.accepted_updates == 0
    assert session.rejected_updates == 0

    # a fresh clone taken after reset must be byte-identical to base
    fresh = clone_base_model(base)
    fresh_clf = fresh.pipeline.named_steps["classifier"]
    base_clf = base.pipeline.named_steps["classifier"]
    assert np.array_equal(fresh_clf.coef_, base_clf.coef_)
    assert np.array_equal(fresh_clf.intercept_, base_clf.intercept_)
