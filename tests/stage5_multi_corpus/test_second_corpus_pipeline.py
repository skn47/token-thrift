import pytest

from tokenthrift.corpus.documents import load_all_chunks
from tokenthrift.corpus.labels import load_question_labels
from tokenthrift.corpus.registry import resolve_corpus
from tokenthrift.corpus.splits import load_persisted_splits
from tokenthrift.pruner.model import PrunerModel, resolve_latest_version
from tokenthrift.pruner.training import build_examples, chunks_for_docs


@pytest.mark.parametrize("corpus_id", ["lighthouse", "nimbus"])
def test_train_examples_are_built_only_from_train_split_documents(corpus_id):
    spec = resolve_corpus(corpus_id)
    all_chunks = load_all_chunks(spec.corpus_dir)
    all_questions = {
        q.question_id: q for q in load_question_labels(spec.labels_path)
    }
    splits = load_persisted_splits(spec.splits_dir)

    train_doc_ids = set(splits.doc_ids_for("train"))
    val_doc_ids = set(splits.doc_ids_for("val"))
    test_doc_ids = set(splits.doc_ids_for("test"))
    assert train_doc_ids.isdisjoint(val_doc_ids)
    assert train_doc_ids.isdisjoint(test_doc_ids)

    train_chunks = chunks_for_docs(all_chunks, train_doc_ids)
    assert all(c.doc_id in train_doc_ids for c in train_chunks)

    train_questions = [all_questions[qid] for qid in splits.question_ids_for("train")]
    features, labels = build_examples(train_questions, train_chunks)

    assert features
    assert set(labels) == {0, 1}


@pytest.mark.parametrize("corpus_id", ["lighthouse", "nimbus"])
def test_persisted_artifact_has_non_degenerate_held_out_metrics(corpus_id):
    spec = resolve_corpus(corpus_id)
    version_dir = resolve_latest_version(spec.artifacts_dir)
    assert version_dir is not None

    model = PrunerModel.load(version_dir)
    assert model.metadata.corpus_id == corpus_id
    assert 0.0 <= model.metadata.threshold <= 1.0
    assert model.metadata.train_examples > 0
    assert model.metadata.val_examples > 0
    assert model.metadata.test_examples > 0
    assert model.metadata.test_metrics["recall"] > 0.0
