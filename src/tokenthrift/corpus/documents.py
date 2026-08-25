from __future__ import annotations

import json
from pathlib import Path

from tokenthrift.config import CORPUS_DIR
from tokenthrift.core.types import Chunk


def load_documents(corpus_dir: Path = CORPUS_DIR) -> dict[str, dict]:
    documents: dict[str, dict] = {}
    for path in sorted(corpus_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        documents[payload["doc_id"]] = payload
    return documents


def load_all_chunks(corpus_dir: Path = CORPUS_DIR) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in load_documents(corpus_dir).values():
        for raw in doc["chunks"]:
            chunks.append(Chunk(
                doc_id=doc["doc_id"],
                chunk_id=raw["chunk_id"],
                text=raw["text"],
                source_type=raw["source_type"],
                doc_title=doc["title"],
                heading=raw["heading"],
                position=raw["position"],
                doc_chunk_count=raw["doc_chunk_count"],
            ))
    return chunks


def chunks_by_id(chunks: list[Chunk]) -> dict[str, Chunk]:
    return {c.chunk_id: c for c in chunks}
