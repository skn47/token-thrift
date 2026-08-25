from __future__ import annotations

import datetime as dt

from sklearn.utils.class_weight import compute_sample_weight

from tokenthrift.config import DEFAULT_CORPUS_ID, FEATURE_VERSION, RETRIEVAL_TOP_K
from tokenthrift.core.types import Chunk, FeatureVector
from tokenthrift.corpus.documents import load_all_chunks
from tokenthrift.corpus.labels import QuestionLabels, load_question_labels
from tokenthrift.corpus.registry import resolve_corpus
from tokenthrift.corpus.splits import load_persisted_splits
from tokenthrift.eval.metrics import classification_metrics
from tokenthrift.features.extractor import extract_features
from tokenthrift.pruner.model import (
    EvalMetadata,
    PrunerModel,
    allocate_next_version,
    build_pipeline,
    features_to_frame,
)
from tokenthrift.retrieval.tfidf_retriever import TfidfRetriever


def chunks_for_docs(chunks: list[Chunk], doc_ids: set[str]) -> list[Chunk]:
    return [c for c in chunks if c.doc_id in doc_ids]


def build_examples(
    questions: list[QuestionLabels],
    chunks: list[Chunk],
    k: int = RETRIEVAL_TOP_K,
) -> tuple[list[FeatureVector], list[int]]:
    """Builds (feature, label) training examples for one split.

    `chunks` must already be restricted to that split's own documents, so
    negative examples are only ever drawn from the same split's chunks —
    never from validation or held-out test documents.
    """
    if not chunks:
        return [], []
    retriever = TfidfRetriever(chunks)
    features: list[FeatureVector] = []
    labels: list[int] = []
    for q in questions:
        ranked = retriever.retrieve(q.question_text, k=k)
        for rc in ranked:
            label = 1 if rc.chunk.chunk_id in q.relevant_chunk_ids else 0
            features.append(extract_features(q.question_text, rc, ranked))
            labels.append(label)
    return features, labels


def select_threshold(y_true: list[int], y_scores: list[float]) -> float:
    """Sweeps candidate thresholds and picks the one maximizing F1 on the
    validation set, breaking ties toward the lower (higher-recall, more
    conservative) threshold since ties are checked in ascending order."""
    best_threshold = 0.5
    best_f1 = -1.0
    for step in range(5, 95, 5):
        threshold = step / 100
        y_pred = [1 if s >= threshold else 0 for s in y_scores]
        metrics = classification_metrics(y_true, y_pred)
        if metrics.f1 > best_f1:
            best_f1 = metrics.f1
            best_threshold = threshold
    return best_threshold


def train_and_evaluate(seed: int = 0, corpus_id: str = DEFAULT_CORPUS_ID) -> PrunerModel:
    spec = resolve_corpus(corpus_id)
    all_chunks = load_all_chunks(spec.corpus_dir)
    all_questions = {q.question_id: q for q in load_question_labels(spec.labels_path)}
    splits = load_persisted_splits(spec.splits_dir)

    def questions_for(split: str) -> list[QuestionLabels]:
        return [all_questions[qid] for qid in splits.question_ids_for(split)]

    train_chunks = chunks_for_docs(all_chunks, set(splits.doc_ids_for("train")))
    val_chunks = chunks_for_docs(all_chunks, set(splits.doc_ids_for("val")))
    test_chunks = chunks_for_docs(all_chunks, set(splits.doc_ids_for("test")))

    train_features, train_labels = build_examples(questions_for("train"), train_chunks)
    val_features, val_labels = build_examples(questions_for("val"), val_chunks)
    test_features, test_labels = build_examples(questions_for("test"), test_chunks)

    if len(set(train_labels)) < 2:
        raise ValueError(
            "training split must contain both relevant and irrelevant "
            "examples; check the corpus/labels/split configuration")

    # class_weight="balanced" isn't available on the classifier (see
    # pruner/model.py) since it breaks Stage 4's single-class partial_fit
    # batches, so class balance is applied here instead via sample_weight,
    # which has the same rebalancing effect at batch-training time.
    sample_weight = compute_sample_weight("balanced", train_labels)
    pipeline = build_pipeline(random_state=seed)
    pipeline.fit(
        features_to_frame(train_features), train_labels,
        classifier__sample_weight=sample_weight)

    val_scores = (
        list(pipeline.predict_proba(features_to_frame(val_features))[:, 1])
        if val_features else []
    )
    threshold = select_threshold(val_labels, val_scores) if val_features else 0.5
    val_pred = [1 if s >= threshold else 0 for s in val_scores]
    val_metrics = (
        classification_metrics(val_labels, val_pred).to_dict()
        if val_features else {}
    )

    test_scores = (
        list(pipeline.predict_proba(features_to_frame(test_features))[:, 1])
        if test_features else []
    )
    test_pred = [1 if s >= threshold else 0 for s in test_scores]
    test_metrics = (
        classification_metrics(test_labels, test_pred).to_dict()
        if test_features else {}
    )

    metadata = EvalMetadata(
        corpus_id=spec.corpus_id,
        trained_at=dt.datetime.now(dt.UTC).isoformat(),
        feature_version=FEATURE_VERSION,
        train_examples=len(train_features),
        val_examples=len(val_features),
        test_examples=len(test_features),
        threshold=threshold,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
    )
    model = PrunerModel(pipeline=pipeline, metadata=metadata)
    version_dir = allocate_next_version(spec.artifacts_dir)
    model.save(version_dir)
    return model
