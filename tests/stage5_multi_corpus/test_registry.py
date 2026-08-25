from pathlib import Path

import pytest

from tokenthrift.corpus.registry import (
    CUSTOM_CORPUS_SENTINEL,
    UnknownCorpusError,
    ad_hoc_corpus,
    list_bundled_corpora,
    resolve_corpus,
)


def test_both_bundled_corpora_are_discovered_and_labeled():
    corpora = {c.corpus_id: c for c in list_bundled_corpora()}
    assert "lighthouse" in corpora
    assert "nimbus" in corpora
    assert corpora["lighthouse"].labeled is True
    assert corpora["nimbus"].labeled is True
    assert corpora["lighthouse"].display_name
    assert corpora["nimbus"].display_name


def test_resolve_corpus_paths_exist_on_disk():
    for corpus_id in ("lighthouse", "nimbus"):
        spec = resolve_corpus(corpus_id)
        assert spec.corpus_dir.is_dir()
        assert spec.labels_path.exists()
        assert spec.splits_dir.is_dir()
        assert spec.artifacts_dir.name == corpus_id


def test_unknown_corpus_id_raises_clearly():
    with pytest.raises(UnknownCorpusError):
        resolve_corpus("does-not-exist")


def test_ad_hoc_corpus_is_unlabeled_and_borrows_a_chosen_models_artifacts(tmp_path):
    spec = ad_hoc_corpus(tmp_path, model_source_id="nimbus")
    assert spec.corpus_id == CUSTOM_CORPUS_SENTINEL
    assert spec.labeled is False
    assert spec.labels_path is None
    assert spec.splits_dir is None
    assert spec.artifacts_dir == resolve_corpus("nimbus").artifacts_dir


def test_ad_hoc_corpus_requires_a_valid_model_source():
    with pytest.raises(UnknownCorpusError):
        ad_hoc_corpus(Path("."), model_source_id="nope")
