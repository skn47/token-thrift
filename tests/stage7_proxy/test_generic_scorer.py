from tokenthrift.core.types import Chunk
from tokenthrift.proxy.generic_scorer import rank_and_score


def _chunk(chunk_id: str, text: str, position: int = 0) -> Chunk:
    return Chunk(
        doc_id="doc", chunk_id=chunk_id, text=text, source_type="prose",
        doc_title="doc", heading=None, position=position, doc_chunk_count=1,
    )


def test_no_chunks_returns_empty():
    ranked, scores = rank_and_score([], "some query")
    assert ranked == []
    assert scores == {}


def test_a_chunk_matching_the_query_scores_higher_than_an_unrelated_one():
    relevant = _chunk("c1", "The password reset flow requires a verified email address.")
    unrelated = _chunk("c2", "Our office is closed on national holidays.", position=1)
    ranked, scores = rank_and_score([relevant, unrelated], "how do I reset my password")

    assert scores["c1"] > scores["c2"]
    assert {rc.chunk.chunk_id for rc in ranked} == {"c1", "c2"}


def test_degenerate_all_stopword_chunks_fall_back_to_keeping_everything():
    chunks = [_chunk("c1", "the a an"), _chunk("c2", "is of to", position=1)]
    ranked, scores = rank_and_score(chunks, "the a an")
    assert scores == {"c1": 1.0, "c2": 1.0}
    assert len(ranked) == 2
