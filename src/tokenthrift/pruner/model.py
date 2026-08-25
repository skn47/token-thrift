from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from tokenthrift.config import ARTIFACTS_DIR
from tokenthrift.core.types import (
    CATEGORICAL_FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
    SOURCE_TYPES,
    FeatureVector,
)

_VERSION_RE = re.compile(r"^v(\d+)$")


def build_preprocessing() -> ColumnTransformer:
    """Shared feature preprocessing: every pruner classifier (primary SGD
    pipeline, and any alternative benchmarked against it in
    pruner/baselines.py) uses this exact same transformer, so a
    classifier-choice comparison isolates the classifier, not the
    features."""
    return ColumnTransformer([
        ("numeric", StandardScaler(), list(NUMERIC_FEATURE_COLUMNS)),
        (
            "categorical",
            OneHotEncoder(
                categories=[list(SOURCE_TYPES)], handle_unknown="ignore"),
            list(CATEGORICAL_FEATURE_COLUMNS),
        ),
    ])


def build_pipeline(random_state: int = 0) -> Pipeline:
    preprocessing = build_preprocessing()
    # class_weight="balanced" cannot be used here: it requires every class
    # in classes_ to be present in *each* partial_fit call's y, which a
    # single Stage 4 feedback event's label batch will routinely violate
    # (e.g. one "mark irrelevant" click is an all-negative batch). Class
    # balance is instead handled via explicit sample_weight at both batch
    # training time (pruner/training.py) and SGD-update time
    # (session/sgd_adapter.py), which works with single-class batches.
    classifier = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        random_state=random_state,
    )
    return Pipeline([("preprocessing", preprocessing), ("classifier", classifier)])


def features_to_frame(vectors: list[FeatureVector]) -> pd.DataFrame:
    columns = list(NUMERIC_FEATURE_COLUMNS) + list(CATEGORICAL_FEATURE_COLUMNS)
    return pd.DataFrame([v.to_dict() for v in vectors], columns=columns)


@dataclass
class EvalMetadata:
    corpus_id: str
    trained_at: str
    feature_version: str
    train_examples: int
    val_examples: int
    test_examples: int
    threshold: float
    val_metrics: dict
    test_metrics: dict


class PrunerModel:
    def __init__(self, pipeline: Pipeline, metadata: EvalMetadata):
        self.pipeline = pipeline
        self.metadata = metadata

    def predict_proba(self, vectors: list[FeatureVector]) -> list[float]:
        if not vectors:
            return []
        X = features_to_frame(vectors)
        return list(self.pipeline.predict_proba(X)[:, 1])

    def save(self, version_dir: Path) -> None:
        version_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, version_dir / "model.joblib")
        (version_dir / "eval_metadata.json").write_text(
            json.dumps(asdict(self.metadata), indent=2) + "\n")

    @classmethod
    def load(cls, version_dir: Path) -> PrunerModel:
        pipeline = joblib.load(version_dir / "model.joblib")
        metadata = EvalMetadata(**json.loads(
            (version_dir / "eval_metadata.json").read_text()))
        return cls(pipeline=pipeline, metadata=metadata)


def resolve_latest_version(artifacts_dir: Path = ARTIFACTS_DIR) -> Path | None:
    if not artifacts_dir.exists():
        return None
    versions = []
    for child in artifacts_dir.iterdir():
        m = _VERSION_RE.match(child.name)
        if m and child.is_dir():
            versions.append((int(m.group(1)), child))
    if not versions:
        return None
    return max(versions, key=lambda vp: vp[0])[1]


def allocate_next_version(artifacts_dir: Path = ARTIFACTS_DIR) -> Path:
    latest = resolve_latest_version(artifacts_dir)
    next_n = 1
    if latest is not None:
        next_n = int(_VERSION_RE.match(latest.name).group(1)) + 1
    return artifacts_dir / f"v{next_n}"
