from tokenthrift.config import ARTIFACTS_DIR
from tokenthrift.corpus.documents import load_all_chunks
from tokenthrift.pruner.interface import Pruner
from tokenthrift.pruner.model import PrunerModel, resolve_latest_version
from tokenthrift.retrieval.embedding_retriever import EmbeddingRetriever
from tokenthrift.safety.policy import PRESETS


def test_embedding_retriever_conforms_to_the_retriever_protocol():
    chunks = load_all_chunks()
    retriever = EmbeddingRetriever(chunks)
    ranked = retriever.retrieve("How do I reset my password?", k=5)

    assert 0 < len(ranked) <= 5
    assert [rc.retrieval_rank for rc in ranked] == list(range(len(ranked)))
    scores = [rc.retrieval_score for rc in ranked]
    assert scores == sorted(scores, reverse=True)


def test_embedding_retriever_is_deterministic():
    chunks = load_all_chunks()
    retriever = EmbeddingRetriever(chunks)
    first = retriever.retrieve("How do I connect Lighthouse to Slack?", k=5)
    second = retriever.retrieve("How do I connect Lighthouse to Slack?", k=5)
    assert [rc.chunk.chunk_id for rc in first] == [rc.chunk.chunk_id for rc in second]


def test_embedding_retriever_is_a_drop_in_replacement_for_the_pruner():
    chunks = load_all_chunks()
    retriever = EmbeddingRetriever(chunks)
    model = PrunerModel.load(resolve_latest_version(ARTIFACTS_DIR))
    pruner = Pruner(retriever=retriever, model=model, model_version="test")

    result = pruner.prune("How do I reset my password?", PRESETS["balanced"], k=8)

    assert result.pruning_enabled is True
    assert len(result.retained) >= 1
