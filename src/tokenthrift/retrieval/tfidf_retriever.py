from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from tokenthrift.core.types import Chunk, RankedChunk


class TfidfRetriever:
    """Broad-recall retrieval over the full corpus via TF-IDF cosine similarity.

    The vectorizer is fit once over all chunk text at construction time, so
    repeated identical queries always transform to the same vector and
    produce identical, deterministically ordered results.
    """

    def __init__(self, chunks: list[Chunk]):
        if not chunks:
            raise ValueError("TfidfRetriever requires at least one chunk")
        self._chunks = list(chunks)
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self._vectorizer.fit_transform(
            [c.text for c in self._chunks])

    def retrieve(self, query: str, k: int) -> list[RankedChunk]:
        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix)[0]
        order = sorted(
            range(len(scores)),
            key=lambda i: (-scores[i], self._chunks[i].chunk_id),
        )
        top = order[:k]
        return [
            RankedChunk(
                chunk=self._chunks[i],
                retrieval_score=float(scores[i]),
                retrieval_rank=rank,
            )
            for rank, i in enumerate(top)
        ]
