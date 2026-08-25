from tokenthrift.corpus.labels import load_question_labels
from tokenthrift.eval.online_protocol import POLICY_NAMES, run_online_protocol

_QUESTIONS = sorted(load_question_labels(), key=lambda q: q.question_id)


def test_all_five_policies_run_over_the_same_stream_and_report_cumulative_curves():
    questions = _QUESTIONS[:20]
    runs = run_online_protocol(seed=1, questions=questions)

    assert set(runs.keys()) == set(POLICY_NAMES)
    for run in runs.values():
        assert len(run.trials) == len(questions)
        recall_curve = run.cumulative_recall_curve()
        false_pruning_curve = run.cumulative_false_pruning_curve()
        assert len(recall_curve) == len(questions)
        assert all(0.0 <= r <= 1.0 for r in recall_curve)
        assert all(
            abs((r + f) - 1.0) < 1e-9
            for r, f in zip(recall_curve, false_pruning_curve)
        )


def test_static_base_never_attempts_any_update():
    runs = run_online_protocol(seed=2, questions=_QUESTIONS[:15])
    static_run = runs["static_base"]
    assert static_run.accepted_updates == 0
    assert static_run.rejected_updates == 0


def test_bounded_calibration_never_attempts_sgd_updates():
    runs = run_online_protocol(seed=2, questions=_QUESTIONS[:15])
    calibration_run = runs["bounded_calibration"]
    assert calibration_run.accepted_updates == 0
    assert calibration_run.rejected_updates == 0


def test_noisy_policy_is_rejected_at_least_as_often_as_the_oracle_across_seeds():
    # A weak but meaningful statistical check: the guard should reject
    # updates at least as often under adversarial-quality feedback as
    # under a perfect labeler, since noisy feedback introduces spurious
    # or mislabeled updates the oracle never produces.
    questions = _QUESTIONS[:20]
    noisy_rejections = 0
    oracle_rejections = 0
    for seed in range(3):
        runs = run_online_protocol(seed=seed, questions=questions)
        noisy_rejections += runs["sgd_noisy"].rejected_updates
        oracle_rejections += runs["sgd_oracle"].rejected_updates
    assert noisy_rejections >= oracle_rejections


def test_summary_reports_cumulative_metrics_not_only_the_final_model():
    runs = run_online_protocol(seed=4, questions=_QUESTIONS[:12])
    for run in runs.values():
        summary = run.summary()
        assert summary["num_trials"] == 12
        assert 0.0 <= summary["final_cumulative_recall"] <= 1.0
        assert 0.0 <= summary["final_cumulative_false_pruning_rate"] <= 1.0
        assert "accepted_updates" in summary
        assert "rejected_updates" in summary
