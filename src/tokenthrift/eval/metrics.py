from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass

from tokenthrift.core.types import Policy


@dataclass(frozen=True)
class ClassificationMetrics:
    recall: float
    precision: float
    false_negative_rate: float
    f1: float

    def to_dict(self) -> dict:
        return asdict(self)


def classification_metrics(y_true: list[int], y_pred: list[int]) -> ClassificationMetrics:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    false_negative_rate = fn / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return ClassificationMetrics(
        recall=recall,
        precision=precision,
        false_negative_rate=false_negative_rate,
        f1=f1,
    )


def token_reduction_pct(baseline_tokens: int, retained_tokens: int) -> float:
    if baseline_tokens <= 0:
        return 0.0
    return max(baseline_tokens - retained_tokens, 0) / baseline_tokens


def latency_percentile(samples_seconds: list[float], pct: float) -> float:
    if not samples_seconds:
        return 0.0
    ordered = sorted(samples_seconds)
    idx = min(len(ordered) - 1, max(0, round(pct * (len(ordered) - 1))))
    return ordered[idx]


def median_latency(samples_seconds: list[float]) -> float:
    return statistics.median(samples_seconds) if samples_seconds else 0.0


def expected_calibration_error(
    y_true: list[int], y_scores: list[float], n_bins: int = 5,
) -> float:
    """Bucket predictions by score and compare each bucket's average
    predicted score against its actual positive rate — a standard
    (simplified) expected calibration error. Labels the UI value a
    "relevance score" rather than a literal probability when this is
    high, per the reference doc's calibration guidance."""
    if not y_true:
        return 0.0
    bins: list[list[tuple[int, float]]] = [[] for _ in range(n_bins)]
    for t, s in zip(y_true, y_scores):
        idx = min(n_bins - 1, max(0, int(s * n_bins)))
        bins[idx].append((t, s))
    total = len(y_true)
    error = 0.0
    for bucket in bins:
        if not bucket:
            continue
        avg_score = sum(s for _, s in bucket) / len(bucket)
        avg_true = sum(t for t, _ in bucket) / len(bucket)
        error += (len(bucket) / total) * abs(avg_score - avg_true)
    return error


def policy_movement(start: Policy, end: Policy) -> dict[str, float]:
    return {
        "threshold_delta": end.threshold - start.threshold,
        "min_context_delta": float(end.min_context - start.min_context),
        "token_budget_delta": float(end.token_budget - start.token_budget),
    }


def recovery_trial_counts(recalls: list[float], full_recall: float = 1.0) -> list[int]:
    """For each contiguous run of trials with recall below `full_recall`,
    counts how many trials it took to return to full recall. A run that
    never recovers within the stream is excluded (its recovery time is
    unbounded, not measurable), rather than silently treated as zero."""
    counts: list[int] = []
    i = 0
    n = len(recalls)
    while i < n:
        if recalls[i] < full_recall:
            j = i + 1
            while j < n and recalls[j] < full_recall:
                j += 1
            if j < n:
                counts.append(j - i)
            i = j
        else:
            i += 1
    return counts


def mean_recovery_trials(recalls: list[float], full_recall: float = 1.0) -> float | None:
    counts = recovery_trial_counts(recalls, full_recall)
    return statistics.mean(counts) if counts else None


@dataclass(frozen=True)
class AdaptationSafetyMetrics:
    cumulative_false_pruning_rate: float
    accepted_updates: int
    rejected_updates: int
    mean_recovery_trials: float | None

    def to_dict(self) -> dict:
        return asdict(self)
