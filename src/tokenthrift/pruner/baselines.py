from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from tokenthrift.core.types import FeatureVector, RankedChunk
from tokenthrift.pruner.model import build_preprocessing, features_to_frame


class RetrievalOnlyThresholdBaseline:
    """Keeps every chunk whose raw retrieval (TF-IDF cosine) score clears a
    fixed threshold, with no learned scoring at all."""

    def __init__(self, threshold: float):
        self.threshold = threshold

    def select(self, ranked_chunks: list[RankedChunk]) -> list[RankedChunk]:
        return [rc for rc in ranked_chunks if rc.retrieval_score >= self.threshold]


class TopKBaseline:
    """Keeps the top K chunks by retrieval rank, ignoring their scores."""

    def __init__(self, k: int):
        self.k = k

    def select(self, ranked_chunks: list[RankedChunk]) -> list[RankedChunk]:
        return ranked_chunks[: self.k]


def build_alternative_classifier_pipeline(random_state: int = 0) -> Pipeline:
    """Stretch goal: a structurally different (non-linear, non-SGD)
    classifier sharing the primary pruner's exact feature preprocessing,
    so it can be benchmarked against logistic regression on the same
    features rather than assumed superior/inferior — per the reference
    doc's "Why Logistic Regression" section, which explicitly requires
    this kind of comparison rather than asserting logistic regression
    wins by default."""
    preprocessing = build_preprocessing()
    classifier = RandomForestClassifier(
        n_estimators=200, max_depth=6, random_state=random_state)
    return Pipeline([("preprocessing", preprocessing), ("classifier", classifier)])


def train_alternative_classifier(
    train_features: list[FeatureVector], train_labels: list[int], random_state: int = 0,
) -> Pipeline:
    sample_weight = compute_sample_weight("balanced", train_labels)
    pipeline = build_alternative_classifier_pipeline(random_state=random_state)
    pipeline.fit(
        features_to_frame(train_features), train_labels,
        classifier__sample_weight=sample_weight)
    return pipeline
