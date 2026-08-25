from __future__ import annotations

import re
from dataclasses import dataclass

from tokenthrift.core.types import Chunk
from tokenthrift.corpus.ingest import infer_source_type, split_into_sections

MARKER_OPEN = "<tokenthrift:context>"
MARKER_CLOSE = "</tokenthrift:context>"

_MARKED_SPAN_RE = re.compile(
    re.escape(MARKER_OPEN) + r"(.*?)" + re.escape(MARKER_CLOSE), re.DOTALL)


@dataclass(frozen=True)
class MarkedBlock:
    """One `<tokenthrift:context>...</tokenthrift:context>` span found in a
    message, split into chunks the safety-rule engine can score and decide
    on. `start`/`end` are byte offsets of the *full* marked span (markers
    included) in the original message text, so a caller can splice the
    pruned replacement back in without re-scanning."""
    start: int
    end: int
    chunks: list[Chunk]


def find_marked_blocks(text: str, block_id_prefix: str) -> list[MarkedBlock]:
    """Only text explicitly wrapped in the TokenThrift marker is ever
    touched — a deliberate, opt-in boundary (not a size/role heuristic) so
    the proxy never prunes a system prompt, a user's own question, or
    anything else the caller didn't mark, consistent with this project's
    "never silently touch what you're not sure about" rule."""
    blocks: list[MarkedBlock] = []
    for i, match in enumerate(_MARKED_SPAN_RE.finditer(text)):
        inner = match.group(1)
        doc_id = f"{block_id_prefix}-{i}"
        sections = split_into_sections(inner)
        if not sections and inner.strip():
            sections = [(None, inner.strip())]
        chunk_count = len(sections)
        chunks = [
            Chunk(
                doc_id=doc_id,
                chunk_id=f"{doc_id}::c{position}",
                text=body,
                source_type=infer_source_type(body),
                doc_title=doc_id,
                heading=heading,
                position=position,
                doc_chunk_count=chunk_count,
            )
            for position, (heading, body) in enumerate(sections)
        ]
        blocks.append(MarkedBlock(start=match.start(), end=match.end(), chunks=chunks))
    return blocks


def strip_marked_blocks(text: str) -> str:
    """Removes every marked span, leaving only what the caller wrote
    outside the prunable context. This is what "the query" should mean for
    relevance scoring: at realistic scale (many chunks in one marked
    block), scoring a chunk against a query built by combining every
    chunk's own text together dilutes the signal until nothing looks
    irrelevant enough to prune — the actual ask is whatever's outside the
    markers, same boundary find_marked_blocks already treats as sacred."""
    return _MARKED_SPAN_RE.sub(" ", text).strip()
