import numpy as np

from tokenthrift.session.sgd_adapter import clone_base_model

from ._helpers import base_model


def test_cloned_model_starts_identical_to_the_immutable_base():
    base = base_model()
    cloned = clone_base_model(base)

    base_clf = base.pipeline.named_steps["classifier"]
    cloned_clf = cloned.pipeline.named_steps["classifier"]

    assert np.array_equal(base_clf.coef_, cloned_clf.coef_)
    assert np.array_equal(base_clf.intercept_, cloned_clf.intercept_)
    assert list(base_clf.classes_) == list(cloned_clf.classes_)
    assert set(cloned_clf.classes_) == {0, 1}  # both classes already initialized from batch training


def test_clone_is_independent_of_the_base_pipeline_object():
    base = base_model()
    cloned = clone_base_model(base)

    assert cloned.pipeline is not base.pipeline
    assert cloned.pipeline.named_steps["classifier"] is not base.pipeline.named_steps["classifier"]
    assert cloned.pipeline.named_steps["preprocessing"] is not base.pipeline.named_steps["preprocessing"]
