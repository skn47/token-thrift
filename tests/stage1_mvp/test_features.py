from tokenthrift.config import FEATURE_VERSION
from tokenthrift.core.types import Chunk, RankedChunk
from tokenthrift.features.extractor import extract_features


def _chunk(chunk_id, text, source_type, heading=None, doc_title="Doc"):
    return Chunk(
        doc_id="d1", chunk_id=chunk_id, text=text, source_type=source_type,
        doc_title=doc_title, heading=heading, position=0, doc_chunk_count=1,
    )


def test_feature_vector_has_stable_version_and_all_documented_fields():
    chunk = _chunk("c1", "To reset your password, click the forgot password link.", "prose")
    ranked = [RankedChunk(chunk=chunk, retrieval_score=0.5, retrieval_rank=0)]

    fv = extract_features("How do I reset my password?", ranked[0], ranked)

    assert fv.feature_version == FEATURE_VERSION
    fields = fv.to_dict()
    for key in (
        "tfidf_similarity", "retrieval_rank", "score_margin",
        "query_token_overlap_ratio", "exact_phrase_overlap",
        "title_or_heading_overlap", "entity_overlap",
        "normalized_chunk_length", "source_type", "neighbor_relevance",
    ):
        assert key in fields


def test_short_numeric_table_chunk_gets_no_blanket_relevance_boost():
    irrelevant_numeric = _chunk(
        "num1", "| Code | Meaning |\n|---|---|\n| 500 | Server error |", "table")
    relevant_prose = _chunk(
        "prose1",
        "To reset your password, click the forgot password link and "
        "follow the emailed reset link.", "prose")
    ranked = [
        RankedChunk(chunk=relevant_prose, retrieval_score=0.6, retrieval_rank=0),
        RankedChunk(chunk=irrelevant_numeric, retrieval_score=0.02, retrieval_rank=1),
    ]
    query = "How do I reset my password?"

    fv_numeric = extract_features(query, ranked[1], ranked)
    fv_prose = extract_features(query, ranked[0], ranked)

    assert fv_numeric.query_token_overlap_ratio == 0.0
    assert fv_numeric.exact_phrase_overlap == 0.0
    assert fv_numeric.tfidf_similarity < fv_prose.tfidf_similarity


def test_code_source_type_is_preserved_as_a_structural_feature():
    code_chunk = _chunk(
        "code1",
        'curl -X PATCH https://api.lighthouse.example/v1/tasks/tsk_9f2a '
        '-d status=done',
        "code", heading="PATCH /v1/tasks/{task_id}", doc_title="Tasks API Reference",
    )
    ranked = [RankedChunk(chunk=code_chunk, retrieval_score=0.3, retrieval_rank=0)]

    fv = extract_features(
        "How do I mark a task as done via the API?", ranked[0], ranked)

    assert fv.source_type == "code"
