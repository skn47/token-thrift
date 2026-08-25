import dataclasses

import pytest

from tokenthrift.config import ARTIFACTS_DIR
from tokenthrift.core.types import Policy
from tokenthrift.corpus.documents import load_all_chunks
from tokenthrift.pruner.model import PrunerModel, resolve_latest_version
from tokenthrift.retrieval.tfidf_retriever import TfidfRetriever
from tokenthrift.safety.fallback import (
    ModelLoadError,
    SafePruner,
    load_model_or_raise,
)


def test_missing_artifact_raises_model_load_error():
    with pytest.raises(ModelLoadError):
        load_model_or_raise(None)


def test_nonexistent_artifact_dir_raises_model_load_error(tmp_path):
    with pytest.raises(ModelLoadError):
        load_model_or_raise(tmp_path / "does-not-exist")


def test_incompatible_feature_version_raises_model_load_error(tmp_path):
    version_dir = resolve_latest_version(ARTIFACTS_DIR)
    model = PrunerModel.load(version_dir)
    stale_metadata = dataclasses.replace(model.metadata, feature_version="v0-stale")
    stale_model = PrunerModel(pipeline=model.pipeline, metadata=stale_metadata)
    stale_dir = tmp_path / "v_stale"
    stale_model.save(stale_dir)

    with pytest.raises(ModelLoadError):
        load_model_or_raise(stale_dir)


def test_safe_pruner_falls_back_to_unpruned_when_model_is_none():
    chunks = load_all_chunks()
    retriever = TfidfRetriever(chunks)
    safe_pruner = SafePruner(retriever=retriever, model=None, model_version="none")
    policy = Policy(preset_name="balanced", threshold=0.5, min_context=2, token_budget=2000)

    result = safe_pruner.prune("How do I reset my password?", policy, k=5)

    assert result.pruning_enabled is False
    assert result.disabled_reason is not None
    assert len(result.retained) == 5
    assert len(result.pruned) == 0


def test_safe_pruner_falls_back_when_scoring_raises(monkeypatch):
    chunks = load_all_chunks()
    retriever = TfidfRetriever(chunks)
    version_dir = resolve_latest_version(ARTIFACTS_DIR)
    model = PrunerModel.load(version_dir)

    def _broken_predict_proba(self, vectors):
        raise RuntimeError("simulated feature-pipeline failure")

    monkeypatch.setattr(PrunerModel, "predict_proba", _broken_predict_proba)

    safe_pruner = SafePruner(retriever=retriever, model=model, model_version="v1")
    policy = Policy(preset_name="balanced", threshold=0.5, min_context=2, token_budget=2000)
    result = safe_pruner.prune("How do I reset my password?", policy, k=5)

    assert result.pruning_enabled is False
    assert "simulated feature-pipeline failure" in result.disabled_reason
    assert len(result.retained) == 5
