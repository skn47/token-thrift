from tokenthrift.config import ARTIFACTS_DIR
from tokenthrift.corpus.documents import load_all_chunks
from tokenthrift.corpus.labels import load_question_labels
from tokenthrift.corpus.splits import load_persisted_splits
from tokenthrift.pruner.model import PrunerModel, resolve_latest_version
from tokenthrift.pruner.training import build_examples, chunks_for_docs


def test_train_examples_are_built_only_from_train_split_documents():
    all_chunks = load_all_chunks()
    all_questions = {q.question_id: q for q in load_question_labels()}
    splits = load_persisted_splits()

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


def test_persisted_artifact_has_threshold_from_validation_and_test_metrics_once():
    version_dir = resolve_latest_version(ARTIFACTS_DIR)
    assert version_dir is not None

    model = PrunerModel.load(version_dir)

    assert 0.0 <= model.metadata.threshold <= 1.0
    assert model.metadata.train_examples > 0
    assert model.metadata.val_examples > 0
    assert model.metadata.test_examples > 0
    assert "recall" in model.metadata.test_metrics
    assert model.metadata.feature_version
