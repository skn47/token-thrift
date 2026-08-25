"""Shared fixtures for Stage 4 tests. Not collected by pytest (no test_
prefix) — imported directly by the test modules that need them."""

from __future__ import annotations

from tokenthrift.config import ARTIFACTS_DIR, RETRIEVAL_TOP_K
from tokenthrift.corpus.documents import load_all_chunks
from tokenthrift.corpus.labels import QuestionLabels, load_question_labels
from tokenthrift.corpus.splits import load_persisted_splits
from tokenthrift.features.extractor import extract_features
from tokenthrift.feedback.attribution import ChunkLabel
from tokenthrift.pruner.model import PrunerModel, resolve_latest_version
from tokenthrift.pruner.training import chunks_for_docs
from tokenthrift.retrieval.tfidf_retriever import TfidfRetriever


def base_model() -> PrunerModel:
    return PrunerModel.load(resolve_latest_version(ARTIFACTS_DIR))


def _train_retriever_and_questions() -> tuple[TfidfRetriever, list[QuestionLabels]]:
    all_chunks = load_all_chunks()
    splits = load_persisted_splits()
    train_chunks = chunks_for_docs(all_chunks, set(splits.doc_ids_for("train")))
    retriever = TfidfRetriever(train_chunks)
    all_questions = {q.question_id: q for q in load_question_labels()}
    train_questions = [all_questions[qid] for qid in splits.question_ids_for("train")]
    return retriever, train_questions


def real_labeled_examples(
    relevant: bool, n: int = 6, provenance: str = "explicit_user_mark",
) -> list[ChunkLabel]:
    """Real (question, chunk) pairs from the train split with real feature
    vectors attached. Pass relevant=False against genuinely-relevant
    evidence to build deliberately mislabeled (adversarial) examples —
    the real corpus has no natural "wrong" examples to draw from."""
    retriever, questions = _train_retriever_and_questions()
    labels: list[ChunkLabel] = []
    for q in questions:
        ranked = retriever.retrieve(q.question_text, k=RETRIEVAL_TOP_K)
        for rc in ranked:
            if rc.chunk.chunk_id in q.relevant_chunk_ids:
                features = extract_features(q.question_text, rc, ranked)
                labels.append(ChunkLabel(
                    chunk_id=rc.chunk.chunk_id, relevant=relevant, weight=1.0,
                    confidence="high", provenance=provenance, features=features))
                if len(labels) >= n:
                    return labels
    return labels
