"""Shared writer for bundled corpus content under data/corpora/{corpus_id}/.

Each per-corpus generator script (generate_corpus.py for "lighthouse",
generate_corpus_nimbus.py for "nimbus") defines its own DOCUMENTS/QUESTIONS
literals and calls write_corpus() below — this module owns the on-disk
schema (manifest.json, documents/*.json, labels.jsonl) so every bundled
corpus is written identically.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPORA_DIR = REPO_ROOT / "data" / "corpora"


def _build_document_json(doc: dict) -> dict:
    doc_id = doc["doc_id"]
    chunks_out = []
    n = len(doc["chunks"])
    for position, entry in enumerate(doc["chunks"]):
        suffix, heading, text = entry[0], entry[1], entry[2]
        source_type = entry[3] if len(entry) > 3 else doc["doc_type"]
        chunks_out.append({
            "chunk_id": f"{doc_id}::{suffix}",
            "position": position,
            "doc_chunk_count": n,
            "heading": heading,
            "source_type": source_type,
            "text": text,
        })
    return {
        "doc_id": doc_id,
        "title": doc["title"],
        "doc_type": doc["doc_type"],
        "chunks": chunks_out,
    }


def write_corpus(
    corpus_id: str,
    display_name: str,
    description: str,
    documents: list[dict],
    questions: list[tuple[str, str, list[tuple[str, str]]]],
    corpora_dir: Path = CORPORA_DIR,
) -> None:
    """Writes manifest.json, documents/*.json, and labels.jsonl for one
    bundled, hand-labeled corpus. `documents` and `questions` use the same
    literal shapes as the original single-corpus generate_corpus.py."""
    corpus_root = corpora_dir / corpus_id
    documents_dir = corpus_root / "documents"
    documents_dir.mkdir(parents=True, exist_ok=True)

    (corpus_root / "manifest.json").write_text(json.dumps({
        "display_name": display_name,
        "description": description,
    }, indent=2) + "\n")

    doc_ids_seen: set[str] = set()
    for doc in documents:
        if doc["doc_id"] in doc_ids_seen:
            raise ValueError(f"duplicate doc_id {doc['doc_id']}")
        doc_ids_seen.add(doc["doc_id"])
        payload = _build_document_json(doc)
        (documents_dir / f"{doc['doc_id']}.json").write_text(
            json.dumps(payload, indent=2) + "\n")

    total_chunks = sum(len(d["chunks"]) for d in documents)

    question_ids_seen: set[str] = set()
    with (corpus_root / "labels.jsonl").open("w") as f:
        for question_id, question_text, evidence in questions:
            if question_id in question_ids_seen:
                raise ValueError(f"duplicate question_id {question_id}")
            question_ids_seen.add(question_id)
            for doc_id, chunk_suffix in evidence:
                if doc_id not in doc_ids_seen:
                    raise ValueError(
                        f"{question_id} references unknown doc {doc_id}")
                row = {
                    "question_id": question_id,
                    "question_text": question_text,
                    "doc_id": doc_id,
                    "chunk_id": f"{doc_id}::{chunk_suffix}",
                    "relevant": True,
                }
                f.write(json.dumps(row) + "\n")

    print(f"Wrote {len(documents)} documents, {total_chunks} chunks, "
          f"{len(questions)} questions to {corpus_root}")
