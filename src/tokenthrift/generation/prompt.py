from __future__ import annotations

from tokenthrift.core.types import Chunk

SYSTEM_INSTRUCTION = (
    "You are a helpful assistant answering questions using only the "
    "provided context. If the context does not contain the answer, say "
    "you don't know rather than guessing."
)


def build_prompt(query: str, context_chunks: list[Chunk]) -> str:
    """Builds the generation prompt from a query and a list of chunks. Used
    by both the baseline and pruned paths — only the chunk list passed in
    differs between them — so any prompt-format change applies identically
    to both comparison columns."""
    if not context_chunks:
        context_block = "(no context retrieved)"
    else:
        context_block = "\n\n".join(
            f"[Source: {c.doc_title} - {c.heading or c.chunk_id}]\n{c.text}"
            for c in context_chunks
        )
    return (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )


def build_marked_prompt(query: str, context_chunks: list[Chunk]) -> str:
    """Same prompt as build_prompt, but with the context block wrapped in
    <tokenthrift:context> markers — the opt-in boundary the proxy
    (proxy/chunking.py) requires before it will touch any content. The
    system instruction and question stay outside the markers so they're
    never eligible for pruning, mirroring build_prompt's structure exactly
    so a side-by-side diff is just the added tags."""
    if not context_chunks:
        context_block = "(no context retrieved)"
    else:
        context_block = "\n\n".join(
            f"[Source: {c.doc_title} - {c.heading or c.chunk_id}]\n{c.text}"
            for c in context_chunks
        )
    return (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"Context:\n<tokenthrift:context>\n{context_block}\n</tokenthrift:context>\n\n"
        f"Question: {query}\n"
        "Answer:"
    )
