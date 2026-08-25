import pytest

from tokenthrift.corpus.ingest import chunk_local_folder, resolve_folder_path
from tokenthrift.corpus.registry import resolve_corpus
from tokenthrift.pruner.interface import Pruner
from tokenthrift.pruner.model import PrunerModel, resolve_latest_version
from tokenthrift.retrieval.tfidf_retriever import TfidfRetriever
from tokenthrift.safety.policy import default_policy


def test_chunk_local_folder_produces_valid_chunks(tmp_path):
    (tmp_path / "doc1.md").write_text(
        "# Intro\nSome prose about widgets.\n\n"
        "# Setup\nHow to set widgets up.\n")
    (tmp_path / "doc2.txt").write_text(
        "Paragraph one about gadgets.\n\nParagraph two about more gadgets.\n")

    chunks = chunk_local_folder(tmp_path)
    assert chunks
    for c in chunks:
        assert c.text.strip()
        assert c.chunk_id.startswith(f"{c.doc_id}::")
        assert c.source_type in ("prose", "code", "table")


def test_chunk_local_folder_rejects_empty_or_missing_folder(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError):
        chunk_local_folder(empty)
    with pytest.raises(ValueError):
        chunk_local_folder(tmp_path / "does-not-exist")


def test_resolve_folder_path_expands_tilde(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "docs").mkdir()

    resolved = resolve_folder_path("~/docs")
    assert resolved == (tmp_path / "docs").resolve()
    assert resolved.is_dir()


def test_resolve_folder_path_strips_surrounding_whitespace(tmp_path):
    assert resolve_folder_path(f"  {tmp_path}  ") == tmp_path.resolve()


def test_ad_hoc_chunks_prune_successfully_with_an_existing_labeled_model(tmp_path):
    (tmp_path / "notes.md").write_text(
        "# Refunds\nRefunds are issued within five business days.\n\n"
        "# Support\nContact support by email for help.\n")
    chunks = chunk_local_folder(tmp_path)

    spec = resolve_corpus("lighthouse")
    model = PrunerModel.load(resolve_latest_version(spec.artifacts_dir))
    retriever = TfidfRetriever(chunks)
    pruner = Pruner(retriever=retriever, model=model, model_version="ad_hoc")

    result = pruner.prune("How long do refunds take?", default_policy(), k=5)
    assert result.pruning_enabled is True
