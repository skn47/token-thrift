from tokenthrift.config import REPO_ROOT
from tokenthrift.core.claim_language import find_banned_claim
from tokenthrift.eval.harness import HeldOutReport
from tokenthrift.eval.report import render_report
from tokenthrift.pruner.model import EvalMetadata

_UI_AND_REPORT_SOURCE_FILES = [
    REPO_ROOT / "src" / "tokenthrift" / "eval" / "report.py",
    REPO_ROOT / "src" / "tokenthrift" / "ui" / "app.py",
    REPO_ROOT / "src" / "tokenthrift" / "ui" / "sidebar.py",
    REPO_ROOT / "src" / "tokenthrift" / "ui" / "comparison.py",
    REPO_ROOT / "src" / "tokenthrift" / "ui" / "metrics_panel.py",
]


def test_no_banned_absolute_claims_in_user_facing_source_files():
    for path in _UI_AND_REPORT_SOURCE_FILES:
        content = path.read_text()
        claim = find_banned_claim(content)
        assert claim is None, f"banned phrase {claim!r} found in {path}"


def test_eval_report_uses_target_language_not_guarantees():
    report = HeldOutReport(
        num_questions=5, pruner_recall=0.7, pruner_precision=0.5,
        pruner_token_reduction_pct=0.5, baseline_threshold_recall=0.3,
        baseline_topk_recall=0.6,
    )
    metadata = EvalMetadata(
        corpus_id="lighthouse",
        trained_at="now", feature_version="v1", train_examples=1,
        val_examples=1, test_examples=1, threshold=0.5,
        val_metrics={}, test_metrics={},
    )

    text = render_report(report, metadata)

    assert "targets" in text.lower()
    assert find_banned_claim(text) is None
