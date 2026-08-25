from tokenthrift.corpus.documents import load_all_chunks
from tokenthrift.retrieval.tfidf_retriever import TfidfRetriever


def test_retrieval_is_deterministic_across_repeated_identical_queries():
    chunks = load_all_chunks()
    retriever = TfidfRetriever(chunks)
    query = "How do I reset my password?"

    first = retriever.retrieve(query, k=5)
    second = retriever.retrieve(query, k=5)

    assert [rc.chunk.chunk_id for rc in first] == [rc.chunk.chunk_id for rc in second]
    assert [rc.retrieval_score for rc in first] == [rc.retrieval_score for rc in second]


def test_retrieval_rank_is_ordered_by_descending_score():
    chunks = load_all_chunks()
    retriever = TfidfRetriever(chunks)
    ranked = retriever.retrieve("How do I reset my password?", k=8)

    scores = [rc.retrieval_score for rc in ranked]
    assert scores == sorted(scores, reverse=True)
    assert [rc.retrieval_rank for rc in ranked] == list(range(len(ranked)))
