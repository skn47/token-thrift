from __future__ import annotations

import re
from itertools import pairwise

from tokenthrift.config import FEATURE_VERSION
from tokenthrift.core.text import content_tokens, tokenize
from tokenthrift.core.types import FeatureVector, RankedChunk

_ENTITY_RE = re.compile(r"[A-Za-z][A-Za-z0-9_./:-]*")


def _is_entity_like(raw_token: str) -> bool:
    if any(ch.isdigit() for ch in raw_token):
        return True
    if "_" in raw_token or "-" in raw_token or "." in raw_token or "/" in raw_token:
        return True
    if raw_token.isupper() and len(raw_token) > 1:
        return True
    return raw_token[:1].isupper() and not raw_token.isupper()


def _entity_tokens(text: str) -> set[str]:
    return {
        m.group(0).lower()
        for m in _ENTITY_RE.finditer(text)
        if _is_entity_like(m.group(0))
    }


def _query_bigrams(query_content_tokens: list[str]) -> list[str]:
    return [f"{a} {b}" for a, b in pairwise(query_content_tokens)]


def extract_features(
    query: str,
    target: RankedChunk,
    ranked_chunks: list[RankedChunk],
) -> FeatureVector:
    """Builds the feature vector for one (query, chunk) pair.

    `ranked_chunks` is the full broad-recall retrieval result for this
    query (target included) — several features are only meaningful relative
    to the rest of that result set (score_margin, normalized_chunk_length,
    neighbor_relevance), not the corpus as a whole.
    """
    chunk = target.chunk
    query_tokens = tokenize(query)
    query_token_set = set(query_tokens)
    chunk_tokens = tokenize(chunk.text)
    chunk_token_set = set(chunk_tokens)

    top_score = max((rc.retrieval_score for rc in ranked_chunks), default=0.0)
    score_margin = top_score - target.retrieval_score

    if query_token_set:
        overlap = len(query_token_set & chunk_token_set) / len(query_token_set)
    else:
        overlap = 0.0

    query_content_tokens = content_tokens(query)
    bigrams = _query_bigrams(query_content_tokens)
    chunk_text_lower = chunk.text.lower()
    if bigrams:
        matched = sum(1 for bg in bigrams if bg in chunk_text_lower)
        exact_phrase_overlap = matched / len(bigrams)
    else:
        exact_phrase_overlap = 0.0

    heading_text = f"{chunk.doc_title} {chunk.heading or ''}"
    heading_tokens = set(tokenize(heading_text))
    if query_token_set:
        title_or_heading_overlap = (
            len(query_token_set & heading_tokens) / len(query_token_set)
        )
    else:
        title_or_heading_overlap = 0.0

    query_entities = _entity_tokens(query)
    chunk_entities = _entity_tokens(chunk.text)
    if query_entities:
        entity_overlap = len(query_entities & chunk_entities) / len(query_entities)
    else:
        entity_overlap = 0.0

    max_len = max((len(tokenize(rc.chunk.text)) for rc in ranked_chunks), default=1)
    normalized_chunk_length = len(chunk_tokens) / max_len if max_len else 0.0

    neighbor_scores = [
        rc.retrieval_score
        for rc in ranked_chunks
        if rc.chunk.doc_id == chunk.doc_id
        and rc.chunk.chunk_id != chunk.chunk_id
        and abs(rc.chunk.position - chunk.position) == 1
    ]
    neighbor_relevance = (
        sum(neighbor_scores) / len(neighbor_scores) if neighbor_scores else 0.0
    )

    return FeatureVector(
        feature_version=FEATURE_VERSION,
        tfidf_similarity=target.retrieval_score,
        retrieval_rank=target.retrieval_rank,
        score_margin=score_margin,
        query_token_overlap_ratio=overlap,
        exact_phrase_overlap=exact_phrase_overlap,
        title_or_heading_overlap=title_or_heading_overlap,
        entity_overlap=entity_overlap,
        normalized_chunk_length=normalized_chunk_length,
        source_type=chunk.source_type,
        neighbor_relevance=neighbor_relevance,
    )
