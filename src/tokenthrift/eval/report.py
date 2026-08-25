from __future__ import annotations

from tokenthrift.eval.harness import HeldOutReport
from tokenthrift.pruner.model import EvalMetadata

TARGET_STATEMENT = (
    "TokenThrift targets 40-70% token reduction on suitable workloads. "
    "This is a benchmark target, not a guarantee."
)
QUALITY_STATEMENT = (
    "TokenThrift preserves measured answer quality at conservative "
    "settings; it does not claim absolute correctness."
)
LATENCY_STATEMENT = (
    "Pruning latency is low-overhead and measured separately from "
    "generation latency below."
)


def render_report(report: HeldOutReport, metadata: EvalMetadata) -> str:
    lines = [
        "TokenThrift held-out evaluation report",
        f"Corpus: {metadata.corpus_id}",
        f"Trained: {metadata.trained_at}  Feature version: {metadata.feature_version}",
        f"Held-out questions evaluated: {report.num_questions}",
        "",
        "Measured results (held-out test set, single run):",
        f"  Evidence recall (pruner):        {report.pruner_recall:.0%}",
        f"  Precision (pruner):               {report.pruner_precision:.0%}",
        f"  Token reduction vs baseline:      {report.pruner_token_reduction_pct:.0%}",
        "",
        "Baselines (for comparison, not a claim of TokenThrift superiority):",
        f"  Retrieval-only threshold recall:  {report.baseline_threshold_recall:.0%}",
        f"  Top-K recall:                     {report.baseline_topk_recall:.0%}",
        "",
        TARGET_STATEMENT,
        QUALITY_STATEMENT,
        LATENCY_STATEMENT,
    ]
    return "\n".join(lines)
