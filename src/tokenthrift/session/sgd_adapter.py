from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from tokenthrift.config import DEFAULT_CORPUS_ID
from tokenthrift.core.types import FeatureVector
from tokenthrift.corpus.documents import load_all_chunks
from tokenthrift.corpus.labels import load_question_labels
from tokenthrift.corpus.registry import resolve_corpus
from tokenthrift.corpus.splits import load_persisted_splits
from tokenthrift.features.extractor import extract_features
from tokenthrift.feedback.attribution import ChunkLabel
from tokenthrift.pruner.model import PrunerModel, features_to_frame
from tokenthrift.pruner.training import chunks_for_docs
from tokenthrift.retrieval.tfidf_retriever import TfidfRetriever

# Bounded distance (L2, over concatenated coef_ + intercept_) a session
# classifier may ever drift from the immutable base classifier — an
# absolute bound measured against the fixed base, not a per-step delta, so
# it cannot be exceeded no matter how long a feedback stream runs.
MAX_COEF_DISTANCE = 3.0

# The canary set must retain at least this much evidence recall, and must
# not regress by more than this much relative to its own pre-update score,
# or the proposed update is rejected.
CANARY_MIN_RECALL = 0.5
CANARY_MAX_RECALL_DROP = 0.15

# Explicit user judgments are trusted more than labels derived from an
# imperfect grounding heuristic.
WEIGHT_BY_PROVENANCE = {
    "explicit_user_mark": 1.0,
    "counterfactual_restore": 0.5,
}
DEFAULT_LABEL_WEIGHT = 0.25


@dataclass(frozen=True)
class CanaryExample:
    features: FeatureVector
    label: int


@dataclass(frozen=True)
class UpdateOutcome:
    accepted: bool
    reason: str
    coef_distance: float
    canary_recall_before: float
    canary_recall_after: float


def clone_base_model(base: PrunerModel) -> PrunerModel:
    """Deep-copies the immutable base model's fitted pipeline. A session
    never mutates `base` itself — every update operates on this copy."""
    return PrunerModel(pipeline=copy.deepcopy(base.pipeline), metadata=base.metadata)


def build_canary_set(
    max_size: int = 20, k: int = 12, corpus_id: str = DEFAULT_CORPUS_ID,
) -> list[CanaryExample]:
    """Builds a small evidence-recall canary from the validation split's
    known-positive (query, chunk) pairs: examples the classifier must keep
    recognizing as relevant after any proposed update, or the update is
    rejected. Restricted to the val split so the canary is independent of
    both the training data and any online-eval stream built from test/all
    questions."""
    spec = resolve_corpus(corpus_id)
    all_chunks = load_all_chunks(spec.corpus_dir)
    splits = load_persisted_splits(spec.splits_dir)
    val_doc_ids = set(splits.doc_ids_for("val"))
    val_chunks = chunks_for_docs(all_chunks, val_doc_ids)
    if not val_chunks:
        return []

    all_questions = {q.question_id: q for q in load_question_labels(spec.labels_path)}
    val_questions = [all_questions[qid] for qid in splits.question_ids_for("val")]
    retriever = TfidfRetriever(val_chunks)

    canary: list[CanaryExample] = []
    for q in val_questions:
        ranked = retriever.retrieve(q.question_text, k=k)
        for rc in ranked:
            if rc.chunk.chunk_id in q.relevant_chunk_ids:
                features = extract_features(q.question_text, rc, ranked)
                canary.append(CanaryExample(features=features, label=1))
                if len(canary) >= max_size:
                    return canary
    return canary


def coef_vector(model: PrunerModel) -> np.ndarray:
    clf = model.pipeline.named_steps["classifier"]
    return np.concatenate([clf.coef_.ravel(), clf.intercept_.ravel()])


def coef_distance(session_model: PrunerModel, base_model: PrunerModel) -> float:
    return float(np.linalg.norm(coef_vector(session_model) - coef_vector(base_model)))


def canary_recall(model: PrunerModel, canary: list[CanaryExample]) -> float:
    positives = [c for c in canary if c.label == 1]
    if not positives:
        return 1.0
    scores = model.predict_proba([c.features for c in positives])
    correct = sum(1 for s in scores if s >= 0.5)
    return correct / len(positives)


def propose_update(
    session_model: PrunerModel,
    base_model: PrunerModel,
    labels: list[ChunkLabel],
    canary: list[CanaryExample],
) -> tuple[PrunerModel, UpdateOutcome]:
    """Attempts one guarded partial_fit update on a *copy* of the session
    model — the safeguards from the reference doc, in order:

    1. operates on a clone, never the live session_model or the base;
    2. preprocessing (already fixed on the base) is only ever transformed,
       never refit;
    3. explicit labels are weighted above inferred ones;
    4. the candidate is rejected if it would drift the coefficients more
       than MAX_COEF_DISTANCE from the immutable base;
    5. the candidate is rejected if it regresses the evidence-recall
       canary below CANARY_MIN_RECALL or by more than
       CANARY_MAX_RECALL_DROP relative to its pre-update score.

    Returns (model_to_use_going_forward, outcome) — the returned model is
    session_model unchanged when the update is rejected.
    """
    usable = [l for l in labels if l.features is not None]
    canary_recall_before = canary_recall(session_model, canary)
    if not usable:
        return session_model, UpdateOutcome(
            accepted=False, reason="no usable labels (missing feature vectors)",
            coef_distance=0.0, canary_recall_before=canary_recall_before,
            canary_recall_after=canary_recall_before)

    candidate_pipeline = copy.deepcopy(session_model.pipeline)
    preprocessing = candidate_pipeline.named_steps["preprocessing"]
    classifier = candidate_pipeline.named_steps["classifier"]

    X = features_to_frame([l.features for l in usable])
    y = [1 if l.relevant else 0 for l in usable]
    sample_weight = [
        l.weight * WEIGHT_BY_PROVENANCE.get(l.provenance, DEFAULT_LABEL_WEIGHT)
        for l in usable
    ]

    X_transformed = preprocessing.transform(X)
    classifier.partial_fit(X_transformed, y, sample_weight=sample_weight)
    candidate_model = PrunerModel(pipeline=candidate_pipeline, metadata=session_model.metadata)

    distance = coef_distance(candidate_model, base_model)
    canary_recall_after = canary_recall(candidate_model, canary)

    if distance > MAX_COEF_DISTANCE:
        return session_model, UpdateOutcome(
            accepted=False,
            reason=f"coefficient distance {distance:.2f} exceeds bound {MAX_COEF_DISTANCE}",
            coef_distance=distance, canary_recall_before=canary_recall_before,
            canary_recall_after=canary_recall_after)

    recall_drop = canary_recall_before - canary_recall_after
    if canary_recall_after < CANARY_MIN_RECALL or recall_drop > CANARY_MAX_RECALL_DROP:
        return session_model, UpdateOutcome(
            accepted=False,
            reason=(f"canary recall regressed {canary_recall_before:.2f} -> "
                    f"{canary_recall_after:.2f}"),
            coef_distance=distance, canary_recall_before=canary_recall_before,
            canary_recall_after=canary_recall_after)

    return candidate_model, UpdateOutcome(
        accepted=True, reason="update accepted", coef_distance=distance,
        canary_recall_before=canary_recall_before, canary_recall_after=canary_recall_after)
