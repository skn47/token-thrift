from tokenthrift.core.types import Chunk
from tokenthrift.validation.coverage import check_coverage
from tokenthrift.validation.grounding import ground_answer
from tokenthrift.validation.pipeline import run_layered_check
from tokenthrift.validation.structured import check_structured


def _chunk(chunk_id, text, position=0):
    return Chunk(
        doc_id="d1", chunk_id=chunk_id, text=text, source_type="prose",
        doc_title="Doc", heading=None, position=position, doc_chunk_count=2,
    )


def test_structured_check_flags_a_number_not_present_in_context():
    context = [_chunk("c0", "The rate limit for the Team plan is 600 requests per minute.")]
    result = check_structured("The rate limit is 9999 requests per minute.", context)

    assert result.applicable is True
    assert result.passed is False
    assert "9999" in result.unsupported_numbers


def test_structured_check_is_not_applicable_when_answer_has_no_numbers_or_ids():
    context = [_chunk("c0", "Click forgot password and follow the link.")]
    result = check_structured("Click forgot password and follow the emailed link.", context)

    assert result.applicable is False
    assert result.passed is True


def test_coverage_distinguishes_addressed_from_unaddressed_queries():
    covered = check_coverage(
        "How do I reset my password?",
        "Click forgot password and follow the reset link to reset your password.")
    uncovered = check_coverage(
        "How do I reset my password?",
        "Lighthouse has a mobile app for iOS and Android.")

    assert covered.covered is True
    assert uncovered.covered is False


def test_grounding_maps_supported_sentences_to_their_source_chunks():
    chunks = [
        _chunk("c0", "Click forgot password on the login screen.", position=0),
        _chunk("c1", "Follow the emailed link to set a new password.", position=1),
    ]
    answer = (
        "Click forgot password on the login screen. "
        "Follow the emailed link to set a new password."
    )
    result = ground_answer(answer, chunks)

    assert result.total_sentences == 2
    assert result.supported_sentences == 2
    assert result.is_grounded is True
    assert "c0" in result.chunk_support
    assert "c1" in result.chunk_support


def test_grounding_flags_a_claim_unsupported_by_any_retained_chunk():
    chunks = [_chunk("c0", "Click forgot password on the login screen.")]
    result = ground_answer("Lighthouse was founded in 1999 by aliens.", chunks)

    assert result.supported_sentences == 0
    assert result.is_grounded is False


def test_pipeline_runs_the_three_always_on_layers_and_excludes_counterfactual():
    chunks = [_chunk("c0", "Click forgot password on the login screen.")]
    result = run_layered_check(
        "How do I reset my password?",
        "Click forgot password on the login screen.", chunks)

    assert result.structured is not None
    assert result.coverage is not None
    assert result.grounding is not None
    # counterfactual is a distinct function only invoked after negative
    # feedback triggers a full-context retry — not part of this pipeline
    assert not hasattr(result, "counterfactual")
