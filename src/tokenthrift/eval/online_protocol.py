"""Online evaluation protocol: predict, reveal, update — strictly
chronologically, per the reference doc's Online Evaluation Protocol.

Five policies share the same retriever and the same chronological question
stream, each with independent adaptation state:
  - static_base: never adapts.
  - bounded_calibration: only threshold/min-context/budget calibration.
  - guarded_sgd: full guarded partial_fit driven by realistically-imperfect
    simulated feedback (missed evidence is detected and cited with
    probability REALISTIC_DETECTION_RATE, independently per chunk).
  - sgd_oracle: same guarded partial_fit machinery, fed idealized
    (perfectly accurate) feedback.
  - sgd_noisy: same machinery, fed feedback corrupted with probability
    NOISY_CORRUPTION_RATE (false or missing citations) to test whether
    the bound/canary guards hold up under adversarial-quality feedback.

This module demonstrates the adaptation MECHANICS (chronological
no-leakage, bounded drift, canary-guarded rollback, cumulative-recall
reporting) under controlled feedback-quality variation. It does not run
real generation or the grounding heuristic per trial, so it is not a
substitute for eval.harness's honest held-out quality/token/cost report —
see TokenThrift's Responsible Claim Language section.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from tokenthrift.config import DEFAULT_CORPUS_ID, RETRIEVAL_TOP_K
from tokenthrift.corpus.documents import load_all_chunks
from tokenthrift.corpus.labels import QuestionLabels, load_question_labels
from tokenthrift.corpus.registry import resolve_corpus
from tokenthrift.eval.metrics import mean_recovery_trials
from tokenthrift.features.extractor import extract_features
from tokenthrift.feedback.attribution import attribute
from tokenthrift.feedback.events import AcceptAnswer, RetryFullContext
from tokenthrift.pruner.interface import Pruner
from tokenthrift.pruner.model import PrunerModel, resolve_latest_version
from tokenthrift.retrieval.tfidf_retriever import TfidfRetriever
from tokenthrift.safety.policy import default_policy
from tokenthrift.session.calibration import CalibrationState
from tokenthrift.session.sgd_adapter import CanaryExample, build_canary_set
from tokenthrift.session.state import SessionState

STATIC_BASE = "static_base"
BOUNDED_CALIBRATION = "bounded_calibration"
GUARDED_SGD = "guarded_sgd"
SGD_ORACLE = "sgd_oracle"
SGD_NOISY = "sgd_noisy"
POLICY_NAMES = (STATIC_BASE, BOUNDED_CALIBRATION, GUARDED_SGD, SGD_ORACLE, SGD_NOISY)

# Feedback-quality regimes simulated per adaptive policy. This protocol
# simulates feedback QUALITY directly (rather than generating real text and
# running the grounding heuristic on it) so the full corpus can be swept
# chronologically without an LLM call per trial — see the module docstring
# for what this does and does not demonstrate.
REALISTIC_DETECTION_RATE = 0.8
NOISY_CORRUPTION_RATE = 0.4


@dataclass
class TrialResult:
    question_id: str
    recall: float
    relevant_total: int
    relevant_retained: int


@dataclass
class PolicyRun:
    name: str
    trials: list[TrialResult] = field(default_factory=list)
    accepted_updates: int = 0
    rejected_updates: int = 0

    def cumulative_recall_curve(self) -> list[float]:
        curve = []
        total_relevant = 0
        total_retained = 0
        for t in self.trials:
            total_relevant += t.relevant_total
            total_retained += t.relevant_retained
            curve.append(total_retained / total_relevant if total_relevant else 1.0)
        return curve

    def cumulative_false_pruning_curve(self) -> list[float]:
        return [1.0 - r for r in self.cumulative_recall_curve()]

    def per_trial_recalls(self) -> list[float]:
        return [t.recall for t in self.trials]

    def summary(self) -> dict:
        recall_curve = self.cumulative_recall_curve()
        return {
            "name": self.name,
            "num_trials": len(self.trials),
            "final_cumulative_recall": recall_curve[-1] if recall_curve else 1.0,
            "final_cumulative_false_pruning_rate": (
                1.0 - recall_curve[-1] if recall_curve else 0.0),
            "accepted_updates": self.accepted_updates,
            "rejected_updates": self.rejected_updates,
            "mean_recovery_trials": mean_recovery_trials(self.per_trial_recalls()),
        }


def _ordered_stream(
    questions: list[QuestionLabels] | None, labels_path,
) -> list[QuestionLabels]:
    source = questions if questions is not None else load_question_labels(labels_path)
    return sorted(source, key=lambda q: q.question_id)


def _simulate_cited_ids(
    policy_name: str, missed_ids: set[str], available_ids: set[str],
    other_ids: set[str], rng: random.Random,
) -> set[str]:
    """Simulates which missed-relevant chunk_ids get correctly cited in a
    Retry event, for chunks that were at least present in the broad-recall
    result (a chunk never retrieved at all cannot be cited by any real
    user or grounding check)."""
    citable = missed_ids & available_ids
    if policy_name == SGD_ORACLE:
        return set(citable)
    if policy_name == GUARDED_SGD:
        return {cid for cid in citable if rng.random() < REALISTIC_DETECTION_RATE}
    if policy_name == SGD_NOISY:
        cited = set()
        for cid in citable:
            if rng.random() < NOISY_CORRUPTION_RATE:
                continue  # fails to cite real evidence this time
            cited.add(cid)
        if other_ids and rng.random() < NOISY_CORRUPTION_RATE:
            cited.add(rng.choice(sorted(other_ids)))  # false citation
        return cited
    return set()


def run_online_protocol(
    seed: int = 0,
    questions: list[QuestionLabels] | None = None,
    k: int = RETRIEVAL_TOP_K,
    base_model: PrunerModel | None = None,
    corpus_id: str = DEFAULT_CORPUS_ID,
) -> dict[str, PolicyRun]:
    spec = resolve_corpus(corpus_id)
    if base_model is None:
        base_model = PrunerModel.load(resolve_latest_version(spec.artifacts_dir))

    stream = _ordered_stream(questions, spec.labels_path)
    retriever = TfidfRetriever(load_all_chunks(spec.corpus_dir))
    canary: list[CanaryExample] = build_canary_set(corpus_id=corpus_id)
    rng = random.Random(seed)

    sessions: dict[str, SessionState] = {
        name: SessionState(
            calibration=CalibrationState(policy=default_policy()),
            model_version=name,
        )
        for name in POLICY_NAMES if name != STATIC_BASE
    }
    runs = {name: PolicyRun(name=name) for name in POLICY_NAMES}

    for q in stream:
        ranked = retriever.retrieve(q.question_text, k=k)
        ranked_ids = {rc.chunk.chunk_id for rc in ranked}
        features_by_chunk_id = {
            rc.chunk.chunk_id: extract_features(q.question_text, rc, ranked)
            for rc in ranked
        }

        for name in POLICY_NAMES:
            if name == STATIC_BASE:
                model, policy_obj = base_model, default_policy()
            else:
                session = sessions[name]
                model = session.cloned_model if session.cloned_model is not None else base_model
                policy_obj = session.policy

            result = Pruner(retriever, model, name).prune(q.question_text, policy_obj, k=k)
            retained_ids = {d.chunk.chunk_id for d in result.retained}
            relevant_ids = set(q.relevant_chunk_ids)
            relevant_retained = len(relevant_ids & retained_ids)
            recall = relevant_retained / len(relevant_ids) if relevant_ids else 1.0

            runs[name].trials.append(TrialResult(
                question_id=q.question_id, recall=recall,
                relevant_total=len(relevant_ids), relevant_retained=relevant_retained))

            if name == STATIC_BASE:
                continue

            session = sessions[name]
            missed_ids = relevant_ids - retained_ids
            if missed_ids:
                other_ids = ranked_ids - relevant_ids - retained_ids
                cited = _simulate_cited_ids(name, missed_ids, ranked_ids, other_ids, rng)
                improved = bool(cited)
                event = RetryFullContext(
                    improved=improved,
                    cited_restored_chunk_ids=frozenset(cited),
                    features_by_chunk_id={
                        cid: features_by_chunk_id[cid] for cid in cited
                        if cid in features_by_chunk_id
                    },
                )
            else:
                event = AcceptAnswer(grounded=True)

            decision = attribute(event)
            if name == BOUNDED_CALIBRATION:
                session.apply_adaptation(decision)  # calibration only, no SGD
            else:
                before_accepted, before_rejected = session.accepted_updates, session.rejected_updates
                session.apply_adaptation(decision, base_model=base_model, canary=canary)
                runs[name].accepted_updates += session.accepted_updates - before_accepted
                runs[name].rejected_updates += session.rejected_updates - before_rejected

    return runs
