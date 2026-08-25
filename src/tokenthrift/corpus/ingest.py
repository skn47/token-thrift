from __future__ import annotations

import re
from pathlib import Path

from tokenthrift.core.types import Chunk

_HEADING_RE = re.compile(r"^#{1,3}\s+(\S.*)$")
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_INGESTIBLE_SUFFIXES = (".md", ".markdown", ".txt")


def infer_source_type(text: str) -> str:
    lines = text.strip().splitlines() or [""]
    table_lines = sum(1 for ln in lines if _TABLE_ROW_RE.match(ln))
    if table_lines / len(lines) > 0.4:
        return "table"
    indented_lines = sum(1 for ln in lines if ln.startswith(("    ", "\t")))
    if "```" in text or indented_lines / len(lines) > 0.4:
        return "code"
    return "prose"


def split_into_sections(text: str) -> list[tuple[str | None, str]]:
    """Splits on top-level markdown headings; falls back to blank-line
    paragraph breaks for files with no headings at all (e.g. plain .txt)."""
    lines = text.splitlines()
    has_heading = any(_HEADING_RE.match(ln) for ln in lines)

    sections: list[tuple[str | None, list[str]]] = []
    if has_heading:
        heading: str | None = None
        body: list[str] = []
        for line in lines:
            m = _HEADING_RE.match(line)
            if m:
                if body:
                    sections.append((heading, body))
                heading, body = m.group(1).strip(), []
            else:
                body.append(line)
        if body:
            sections.append((heading, body))
    else:
        paragraph: list[str] = []
        for line in lines:
            if line.strip() == "":
                if paragraph:
                    sections.append((None, paragraph))
                    paragraph = []
            else:
                paragraph.append(line)
        if paragraph:
            sections.append((None, paragraph))

    return [
        (heading, "\n".join(body).strip())
        for heading, body in sections
        if "\n".join(body).strip()
    ]


def resolve_folder_path(raw: str) -> Path:
    """Expands ~ and normalizes relative segments so a path the user can
    see is valid (e.g. via `ls ~/docs`) is recognized as valid here too —
    Path(...).is_dir() alone does not expand ~."""
    return Path(raw.strip()).expanduser().resolve()


def chunk_local_folder(folder: Path) -> list[Chunk]:
    """Generic chunker for an arbitrary folder of .md/.txt files, used only
    for unlabeled ad hoc corpora — no relevance labels are produced or
    assumed, so this feeds live pruning only, never training/evaluation."""
    if not folder.is_dir():
        raise ValueError(f"not a directory: {folder}")

    paths = sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in _INGESTIBLE_SUFFIXES
    )
    if not paths:
        raise ValueError(
            f"no .md/.markdown/.txt files found under {folder}")

    chunks: list[Chunk] = []
    for doc_index, path in enumerate(paths):
        text = path.read_text(errors="ignore")
        doc_id = f"adhoc-{doc_index:03d}-{path.stem}"
        sections = split_into_sections(text)
        n = len(sections)
        for position, (heading, body) in enumerate(sections):
            chunks.append(Chunk(
                doc_id=doc_id,
                chunk_id=f"{doc_id}::c{position}",
                text=body,
                source_type=infer_source_type(body),
                doc_title=path.stem,
                heading=heading,
                position=position,
                doc_chunk_count=n,
            ))
    return chunks
