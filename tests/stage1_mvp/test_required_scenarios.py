import pytest

from tokenthrift.config import ARTIFACTS_DIR, RETRIEVAL_TOP_K
from tokenthrift.core.types import Policy
from tokenthrift.corpus.documents import load_all_chunks
from tokenthrift.pruner.interface import Pruner
from tokenthrift.pruner.model import PrunerModel, resolve_latest_version
from tokenthrift.retrieval.tfidf_retriever import TfidfRetriever
from tokenthrift.safety.policy import PRESETS


@pytest.fixture(scope="module")
def pruner() -> Pruner:
    chunks = load_all_chunks()
    retriever = TfidfRetriever(chunks)
    model = PrunerModel.load(resolve_latest_version(ARTIFACTS_DIR))
    return Pruner(retriever=retriever, model=model, model_version="test")


def test_no_chunk_scores_above_threshold_still_returns_a_nonempty_prompt(pruner):
    policy = PRESETS["aggressive"]  # highest threshold, easiest to clear entirely
    result = pruner.prune(
        "What is the weather like on Mars today?", policy, k=RETRIEVAL_TOP_K)

    assert len(result.retained) >= 1
    assert any(d.safety_override == "top_ranked_result" for d in result.retained)


def test_retrieved_context_exceeding_token_budget_fills_in_stable_order(pruner):
    # threshold=0.0 makes every retrieved chunk clear the classifier
    # threshold, isolating budget enforcement (not threshold filtering) as
    # the only reason anything beyond the mandatory top chunk gets pruned.
    tiny_budget_policy = Policy(
        preset_name="custom", threshold=0.0, min_context=1, token_budget=1)
    result = pruner.prune(
        "How do I invite teammates to my workspace?", tiny_budget_policy,
        k=RETRIEVAL_TOP_K)

    assert len(result.retained) == 1
    assert result.retained[0].safety_override == "top_ranked_result"
    assert result.pruned
    assert all(d.reasons == ("budget_exceeded",) for d in result.pruned)


def test_two_identical_setting_runs_produce_a_fair_comparison(pruner):
    policy = PRESETS["balanced"]
    query = "How do I connect Lighthouse to Slack?"

    first = pruner.prune(query, policy, k=RETRIEVAL_TOP_K)
    second = pruner.prune(query, policy, k=RETRIEVAL_TOP_K)

    assert [d.chunk.chunk_id for d in first.retained] == \
        [d.chunk.chunk_id for d in second.retained]
    assert first.baseline_tokens == second.baseline_tokens
    assert first.retained_tokens == second.retained_tokens


def test_short_numeric_chunk_is_not_force_kept_for_an_unrelated_query(pruner):
    policy = PRESETS["balanced"]
    result = pruner.prune("How do I reset my password?", policy, k=RETRIEVAL_TOP_K)

    retained_ids = {d.chunk.chunk_id for d in result.retained}
    # the HTTP status-code table is numeric/dense but unrelated to a
    # password reset question, so it must never be force-kept on that basis
    assert "status-codes::c1" not in retained_ids


def test_code_identifier_dense_evidence_is_considered_despite_low_nl_overlap(pruner):
    policy = PRESETS["conservative"]
    result = pruner.prune(
        "What fields are returned when I fetch a single task by ID?",
        policy, k=RETRIEVAL_TOP_K)

    considered = {d.chunk.chunk_id for d in result.retained} | \
        {d.chunk.chunk_id for d in result.pruned}
    # Whether it clears the threshold depends on the tiny trained model's
    # weights, which is expected to vary; what must hold structurally is
    # that identifier-dense evidence reaches scoring at all via retrieval,
    # rather than being filtered out before the classifier ever sees it.
    assert "api-tasks::c1" in considered
