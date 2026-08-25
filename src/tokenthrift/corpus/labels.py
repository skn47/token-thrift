from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from tokenthrift.config import LABELS_PATH


@dataclass(frozen=True)
class QuestionLabels:
    question_id: str
    question_text: str
    relevant_chunk_ids: frozenset[str]
    relevant_doc_ids: frozenset[str]


def load_question_labels(labels_path: Path = LABELS_PATH) -> list[QuestionLabels]:
    by_question: dict[str, dict] = {}
    with labels_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not row.get("relevant", False):
                continue
            entry = by_question.setdefault(row["question_id"], {
                "question_text": row["question_text"],
                "chunk_ids": set(),
                "doc_ids": set(),
            })
            entry["chunk_ids"].add(row["chunk_id"])
            entry["doc_ids"].add(row["doc_id"])

    return [
        QuestionLabels(
            question_id=qid,
            question_text=entry["question_text"],
            relevant_chunk_ids=frozenset(entry["chunk_ids"]),
            relevant_doc_ids=frozenset(entry["doc_ids"]),
        )
        for qid, entry in by_question.items()
    ]
