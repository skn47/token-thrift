from __future__ import annotations

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from tokenthrift.core.types import Chunk, RankedChunk

DEFAULT_N_COMPONENTS = 50


class EmbeddingRetriever:
    """Stretch goal: a lightweight dense-embedding retriever conforming to
    the same Retriever protocol as TfidfRetriever, built via TruncatedSVD
    (latent-semantic) over the TF-IDF matrix — no external embedding model
    or network dependency, so it stays usable offline at this demo-scale
    corpus size. Purely additive: nothing in the app switches to this
    automatically, matching the explicit-retriever/provider-switch
    constraint."""

    def __init__(self, chunks: list[Chunk], n_components: int = DEFAULT_N_COMPONENTS):
        if not chunks:
            raise ValueError("EmbeddingRetriever requires at least one chunk")
        self._chunks = list(chunks)

        self._vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = self._vectorizer.fit_transform([c.text for c in self._chunks])

        n_features = tfidf_matrix.shape[1]
        effective_components = max(1, min(n_components, n_features - 1, len(self._chunks) - 1))
        self._svd = TruncatedSVD(n_components=effective_components, random_state=0)
        self._embeddings = self._svd.fit_transform(tfidf_matrix)

    def retrieve(self, query: str, k: int) -> list[RankedChunk]:
        query_tfidf = self._vectorizer.transform([query])
        query_embedding = self._svd.transform(query_tfidf)
        scores = cosine_similarity(query_embedding, self._embeddings)[0]
        order = sorted(
            range(len(scores)),
            key=lambda i: (-scores[i], self._chunks[i].chunk_id),
        )
        top = order[:k]
        return [
            RankedChunk(
                chunk=self._chunks[i], retrieval_score=float(scores[i]),
                retrieval_rank=rank,
            )
            for rank, i in enumerate(top)
        ]
