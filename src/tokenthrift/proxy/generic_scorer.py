from __future__ import annotations

from tokenthrift.core.types import Chunk, RankedChunk
from tokenthrift.retrieval.tfidf_retriever import TfidfRetriever


def rank_and_score(chunks: list[Chunk], query: str) -> tuple[list[RankedChunk], dict[str, float]]:
    """Relevance scoring for arbitrary, ad hoc context with no trained,
    corpus-specific classifier available — there's no pre-labeled training
    set for a request the proxy has never seen before. Falls back to the
    same TF-IDF-cosine-similarity signal `TfidfRetriever` already uses for
    broad-recall retrieval, fit fresh over just this request's chunks. This
    trades the trained pruner's precision for a signal that needs no
    training data, and is fed into the same `apply_safety_rules` retention
    floors the trained path uses.

    Never raises and never drops a chunk it can't score: if every chunk's
    text is empty/all-stopwords (a degenerate TF-IDF vocabulary), every
    chunk gets a score of 1.0 so the safety rules keep it rather than the
    proxy silently discarding unscoreable content.
    """
    if not chunks:
        return [], {}
    try:
        retriever = TfidfRetriever(chunks)
        ranked = retriever.retrieve(query, k=len(chunks))
    except ValueError:
        ranked = [
            RankedChunk(chunk=c, retrieval_score=1.0, retrieval_rank=i)
            for i, c in enumerate(chunks)
        ]
    scores = {rc.chunk.chunk_id: rc.retrieval_score for rc in ranked}
    return ranked, scores
