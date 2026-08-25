from tokenthrift.corpus.labels import load_question_labels
from tokenthrift.eval.online_protocol import run_online_protocol


def test_early_trial_predictions_are_unaffected_by_later_trials_in_the_stream():
    questions = sorted(load_question_labels(), key=lambda q: q.question_id)
    short_stream = questions[:10]
    long_stream = questions[:20]

    short_runs = run_online_protocol(seed=0, questions=short_stream)
    long_runs = run_online_protocol(seed=0, questions=long_stream)

    for name in short_runs:
        short_recalls = short_runs[name].per_trial_recalls()
        long_recalls_prefix = long_runs[name].per_trial_recalls()[:10]
        assert short_recalls == long_recalls_prefix, (
            f"{name}: predictions in the first 10 trials changed when "
            f"later trials were appended to the stream — feedback leaked "
            f"backward in time")


def test_cumulative_curves_are_prefix_consistent_across_stream_lengths():
    questions = sorted(load_question_labels(), key=lambda q: q.question_id)
    short_runs = run_online_protocol(seed=3, questions=questions[:8])
    long_runs = run_online_protocol(seed=3, questions=questions[:16])

    for name in short_runs:
        short_curve = short_runs[name].cumulative_recall_curve()
        long_curve_prefix = long_runs[name].cumulative_recall_curve()[:8]
        assert short_curve == long_curve_prefix
