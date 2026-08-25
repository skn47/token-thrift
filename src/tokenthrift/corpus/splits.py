from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from tokenthrift.config import CORPUS_DIR, LABELS_PATH, SPLITS_DIR
from tokenthrift.corpus.documents import load_documents
from tokenthrift.corpus.labels import QuestionLabels, load_question_labels

SPLIT_NAMES = ("train", "val", "test")
DEFAULT_RATIOS = (0.6, 0.2, 0.2)


@dataclass(frozen=True)
class CorpusSplits:
    doc_split: dict[str, str]
    question_split: dict[str, str]
    excluded_question_ids: frozenset[str]

    def doc_ids_for(self, split: str) -> list[str]:
        return sorted(d for d, s in self.doc_split.items() if s == split)

    def question_ids_for(self, split: str) -> list[str]:
        return sorted(q for q, s in self.question_split.items() if s == split)


def split_by_document(
    documents: dict[str, dict],
    seed: int = 0,
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
) -> dict[str, str]:
    """Assigns whole documents to train/val/test, stratified by doc_type.

    Stratifying (rather than a single global shuffle) keeps every document
    type represented in every split even with only a handful of documents
    per type.
    """
    by_type: dict[str, list[str]] = {}
    for doc_id, doc in documents.items():
        by_type.setdefault(doc["doc_type"], []).append(doc_id)

    doc_split: dict[str, str] = {}
    for doc_type, doc_ids in by_type.items():
        rng = random.Random(f"{seed}:{doc_type}")
        ordered = sorted(doc_ids)
        rng.shuffle(ordered)
        n = len(ordered)
        n_train = round(n * ratios[0])
        n_val = round(n * ratios[1])
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)
        n_test = n - n_train - n_val
        assert n_test >= 0
        for doc_id in ordered[:n_train]:
            doc_split[doc_id] = "train"
        for doc_id in ordered[n_train:n_train + n_val]:
            doc_split[doc_id] = "val"
        for doc_id in ordered[n_train + n_val:]:
            doc_split[doc_id] = "test"
    return doc_split


def assign_question_splits(
    questions: list[QuestionLabels],
    doc_split: dict[str, str],
) -> tuple[dict[str, str], frozenset[str]]:
    """Assigns each question to the split owning its evidence.

    A question whose evidence chunks all live in documents of one split is
    assigned to that split. A question whose evidence spans documents in
    more than one split is assigned to whichever split holds a strict
    majority of its evidence chunks; if there is no strict majority the
    question is excluded from train/val/test entirely (explicit exclusion,
    not silent inclusion in an arbitrary split) and reserved for
    qualitative review.
    """
    question_split: dict[str, str] = {}
    excluded: set[str] = set()

    for q in questions:
        votes: dict[str, int] = {}
        for chunk_id in q.relevant_chunk_ids:
            doc_id = chunk_id.split("::", 1)[0]
            split = doc_split[doc_id]
            votes[split] = votes.get(split, 0) + 1

        if len(votes) == 1:
            question_split[q.question_id] = next(iter(votes))
            continue

        ranked = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)
        top_split, top_count = ranked[0]
        runner_up_count = ranked[1][1] if len(ranked) > 1 else 0
        if top_count > runner_up_count:
            question_split[q.question_id] = top_split
        else:
            excluded.add(q.question_id)

    return question_split, frozenset(excluded)


def compute_splits(
    seed: int = 0,
    corpus_dir: Path = CORPUS_DIR,
    labels_path: Path = LABELS_PATH,
) -> CorpusSplits:
    documents = load_documents(corpus_dir)
    questions = load_question_labels(labels_path)
    doc_split = split_by_document(documents, seed=seed)
    question_split, excluded = assign_question_splits(questions, doc_split)
    return CorpusSplits(
        doc_split=doc_split,
        question_split=question_split,
        excluded_question_ids=excluded,
    )


def persist_splits(splits: CorpusSplits, splits_dir: Path = SPLITS_DIR) -> None:
    splits_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLIT_NAMES:
        payload = {
            "doc_ids": splits.doc_ids_for(split),
            "question_ids": splits.question_ids_for(split),
        }
        (splits_dir / f"{split}.json").write_text(
            json.dumps(payload, indent=2) + "\n")
    (splits_dir / "excluded_questions.json").write_text(
        json.dumps(sorted(splits.excluded_question_ids), indent=2) + "\n")


def load_persisted_splits(splits_dir: Path = SPLITS_DIR) -> CorpusSplits:
    doc_split: dict[str, str] = {}
    question_split: dict[str, str] = {}
    for split in SPLIT_NAMES:
        payload = json.loads((splits_dir / f"{split}.json").read_text())
        for doc_id in payload["doc_ids"]:
            doc_split[doc_id] = split
        for question_id in payload["question_ids"]:
            question_split[question_id] = split
    excluded = frozenset(
        json.loads((splits_dir / "excluded_questions.json").read_text()))
    return CorpusSplits(
        doc_split=doc_split,
        question_split=question_split,
        excluded_question_ids=excluded,
    )
