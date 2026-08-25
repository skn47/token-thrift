import dataclasses

import numpy as np

from tokenthrift.session.sgd_adapter import (
    MAX_COEF_DISTANCE,
    build_canary_set,
    clone_base_model,
    coef_distance,
    propose_update,
)

from ._helpers import base_model, real_labeled_examples


def test_update_with_no_usable_labels_is_rejected_and_model_is_unchanged():
    base = base_model()
    session_model = clone_base_model(base)
    canary = build_canary_set()
    unusable = real_labeled_examples(relevant=True, n=1)
    stripped = [dataclasses.replace(unusable[0], features=None)]

    updated, outcome = propose_update(session_model, base, stripped, canary)

    assert outcome.accepted is False
    assert updated is session_model


def test_a_plausible_correctly_labeled_update_is_accepted_within_bound():
    base = base_model()
    session_model = clone_base_model(base)
    canary = build_canary_set()
    labels = real_labeled_examples(relevant=True, n=4)

    updated, outcome = propose_update(session_model, base, labels, canary)

    assert outcome.accepted is True
    assert 0.0 <= outcome.coef_distance <= MAX_COEF_DISTANCE
    assert updated is not session_model
    # accepting genuinely-correct evidence must not regress the canary
    assert outcome.canary_recall_after >= outcome.canary_recall_before


def test_adversarial_mislabeled_evidence_is_rejected_and_rolled_back():
    # Six genuinely-relevant chunks relabeled irrelevant — sustained,
    # unanimous, wrong-direction pressure against known-good evidence.
    base = base_model()
    session_model = clone_base_model(base)
    canary = build_canary_set()
    adversarial_labels = real_labeled_examples(relevant=False, n=6)

    updated, outcome = propose_update(session_model, base, adversarial_labels, canary)

    assert outcome.accepted is False
    assert "coefficient distance" in outcome.reason or "canary" in outcome.reason
    # rollback means the model in use afterward is byte-identical to the
    # pre-update session model, not merely "close"
    updated_clf = updated.pipeline.named_steps["classifier"]
    session_clf = session_model.pipeline.named_steps["classifier"]
    assert np.array_equal(updated_clf.coef_, session_clf.coef_)
    assert np.array_equal(updated_clf.intercept_, session_clf.intercept_)


def test_repeated_adversarial_updates_never_push_coefficients_past_the_bound():
    base = base_model()
    canary = build_canary_set()
    adversarial_labels = real_labeled_examples(relevant=False, n=6)

    current = clone_base_model(base)
    for _ in range(25):
        current, _outcome = propose_update(current, base, adversarial_labels, canary)
        assert coef_distance(current, base) <= MAX_COEF_DISTANCE


def test_repeated_genuine_updates_also_never_exceed_the_bound():
    base = base_model()
    canary = build_canary_set()
    genuine_labels = real_labeled_examples(relevant=True, n=4)

    current = clone_base_model(base)
    for _ in range(50):
        current, _outcome = propose_update(current, base, genuine_labels, canary)
        assert coef_distance(current, base) <= MAX_COEF_DISTANCE
