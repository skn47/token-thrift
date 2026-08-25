from __future__ import annotations

from dataclasses import dataclass

from tokenthrift.config import DEFAULT_CORPUS_ID, RETRIEVAL_TOP_K
from tokenthrift.corpus.documents import load_all_chunks
from tokenthrift.corpus.labels import load_question_labels
from tokenthrift.corpus.registry import resolve_corpus
from tokenthrift.corpus.splits import load_persisted_splits
from tokenthrift.eval.metrics import classification_metrics, token_reduction_pct
from tokenthrift.pruner.baselines import RetrievalOnlyThresholdBaseline, TopKBaseline
from tokenthrift.pruner.interface import Pruner
from tokenthrift.pruner.model import PrunerModel
from tokenthrift.retrieval.tfidf_retriever import TfidfRetriever
from tokenthrift.safety.policy import PRESETS


@dataclass(frozen=True)
class HeldOutReport:
    num_questions: int
    pruner_recall: float
    pruner_precision: float
    pruner_token_reduction_pct: float
    baseline_threshold_recall: float
    baseline_topk_recall: float


def run_held_out_eval(
    model: PrunerModel,
    policy_name: str = "balanced",
    k: int = RETRIEVAL_TOP_K,
    corpus_id: str = DEFAULT_CORPUS_ID,
) -> HeldOutReport:
    """Runs the pruner and two non-learned baselines over the held-out test
    split, reporting evidence recall/precision and token reduction once.
    The retriever here is built only from test-split chunks, matching how
    training built per-split retrievers, so no train/val document text
    influences this reported evaluation."""
    spec = resolve_corpus(corpus_id)
    all_chunks = load_all_chunks(spec.corpus_dir)
    all_questions = {q.question_id: q for q in load_question_labels(spec.labels_path)}
    splits = load_persisted_splits(spec.splits_dir)

    test_doc_ids = set(splits.doc_ids_for("test"))
    test_chunks = [c for c in all_chunks if c.doc_id in test_doc_ids]
    test_questions = [all_questions[qid] for qid in splits.question_ids_for("test")]

    if not test_chunks or not test_questions:
        raise ValueError(
            "test split is empty; run scripts/build_splits.py first")

    retriever = TfidfRetriever(test_chunks)
    policy = PRESETS[policy_name]
    pruner = Pruner(retriever=retriever, model=model, model_version="held_out_eval")

    threshold_baseline = RetrievalOnlyThresholdBaseline(threshold=policy.threshold)
    topk_baseline = TopKBaseline(k=policy.min_context)

    y_true: list[int] = []
    pruner_y_pred: list[int] = []
    threshold_y_pred: list[int] = []
    topk_y_pred: list[int] = []
    baseline_tokens_total = 0
    pruner_retained_tokens_total = 0

    for q in test_questions:
        result = pruner.prune(q.question_text, policy, k=k)
        baseline_tokens_total += result.baseline_tokens
        pruner_retained_tokens_total += result.retained_tokens

        ranked = retriever.retrieve(q.question_text, k=k)
        retained_ids = {d.chunk.chunk_id for d in result.retained}
        threshold_ids = {rc.chunk.chunk_id for rc in threshold_baseline.select(ranked)}
        topk_ids = {rc.chunk.chunk_id for rc in topk_baseline.select(ranked)}

        for rc in ranked:
            y_true.append(1 if rc.chunk.chunk_id in q.relevant_chunk_ids else 0)
            pruner_y_pred.append(1 if rc.chunk.chunk_id in retained_ids else 0)
            threshold_y_pred.append(1 if rc.chunk.chunk_id in threshold_ids else 0)
            topk_y_pred.append(1 if rc.chunk.chunk_id in topk_ids else 0)

    pruner_metrics = classification_metrics(y_true, pruner_y_pred)
    threshold_metrics = classification_metrics(y_true, threshold_y_pred)
    topk_metrics = classification_metrics(y_true, topk_y_pred)

    return HeldOutReport(
        num_questions=len(test_questions),
        pruner_recall=pruner_metrics.recall,
        pruner_precision=pruner_metrics.precision,
        pruner_token_reduction_pct=token_reduction_pct(
            baseline_tokens_total, pruner_retained_tokens_total),
        baseline_threshold_recall=threshold_metrics.recall,
        baseline_topk_recall=topk_metrics.recall,
    )
