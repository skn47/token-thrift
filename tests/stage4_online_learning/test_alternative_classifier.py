from tokenthrift.corpus.documents import load_all_chunks
from tokenthrift.corpus.labels import load_question_labels
from tokenthrift.corpus.splits import load_persisted_splits
from tokenthrift.pruner.baselines import (
    build_alternative_classifier_pipeline,
    train_alternative_classifier,
)
from tokenthrift.pruner.model import build_pipeline, features_to_frame
from tokenthrift.pruner.training import build_examples, chunks_for_docs


def _train_features_and_labels():
    all_chunks = load_all_chunks()
    splits = load_persisted_splits()
    all_questions = {q.question_id: q for q in load_question_labels()}
    train_chunks = chunks_for_docs(all_chunks, set(splits.doc_ids_for("train")))
    train_questions = [all_questions[qid] for qid in splits.question_ids_for("train")]
    return build_examples(train_questions, train_chunks)


def test_alternative_classifier_trains_and_predicts_on_the_same_features():
    features, labels = _train_features_and_labels()
    assert set(labels) == {0, 1}

    pipeline = train_alternative_classifier(features, labels)
    scores = pipeline.predict_proba(features_to_frame(features))[:, 1]

    assert len(scores) == len(features)
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_alternative_classifier_shares_preprocessing_with_the_primary_pruner():
    alt = build_alternative_classifier_pipeline()
    primary = build_pipeline()

    alt_numeric_cols = alt.named_steps["preprocessing"].transformers[0][2]
    primary_numeric_cols = primary.named_steps["preprocessing"].transformers[0][2]
    assert alt_numeric_cols == primary_numeric_cols
