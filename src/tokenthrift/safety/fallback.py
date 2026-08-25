from __future__ import annotations

from pathlib import Path

from tokenthrift.config import FEATURE_VERSION, RETRIEVAL_TOP_K
from tokenthrift.core.tokenizer import count_tokens
from tokenthrift.core.types import ChunkDecision, Policy, PruningResult, RankedChunk
from tokenthrift.pruner.interface import Pruner
from tokenthrift.pruner.model import PrunerModel
from tokenthrift.retrieval.base import Retriever

FALLBACK_OVERRIDE = "fallback_unpruned"


class ModelLoadError(Exception):
    """Raised when a persisted pruner artifact is missing or incompatible
    with the running feature version — never when scoring merely produces
    a low-confidence result, which is not a failure."""


def load_model_or_raise(version_dir: Path | None) -> PrunerModel:
    if version_dir is None or not version_dir.exists():
        raise ModelLoadError(f"no trained pruner artifact found at {version_dir}")
    try:
        model = PrunerModel.load(version_dir)
    except (FileNotFoundError, OSError, KeyError, TypeError, ValueError) as e:
        raise ModelLoadError(
            f"failed to load pruner artifact at {version_dir}: {e}") from e
    if model.metadata.feature_version != FEATURE_VERSION:
        raise ModelLoadError(
            f"artifact feature_version {model.metadata.feature_version!r} does "
            f"not match the running FEATURE_VERSION {FEATURE_VERSION!r}")
    return model


def unpruned_fallback_result(
    query: str, ranked_chunks: list[RankedChunk], policy: Policy, reason: str,
) -> PruningResult:
    """Never claims savings that didn't happen: every retrieved chunk is
    kept, `pruning_enabled` is False, and the reason is explicit rather
    than silently indistinguishable from a normal pruning run."""
    retained = tuple(
        ChunkDecision(
            chunk=rc.chunk, retrieval_rank=rc.retrieval_rank,
            relevance_score=None, kept=True,
            reasons=("pruning_disabled",), safety_override=FALLBACK_OVERRIDE,
        )
        for rc in ranked_chunks
    )
    tokens = sum(count_tokens(rc.chunk.text) for rc in ranked_chunks)
    return PruningResult(
        query=query, retained=retained, pruned=(), policy=policy,
        pruning_enabled=False, disabled_reason=reason,
        retained_tokens=tokens, baseline_tokens=tokens, model_version=None,
    )


class SafePruner:
    """Serving-path wrapper around Pruner that never lets a model/feature
    failure break the user's request: it falls back to the unpruned
    retrieval result and reports why. Training and evaluation code should
    keep using Pruner directly — those paths want failures to raise loudly,
    not be silently absorbed."""

    def __init__(self, retriever: Retriever, model: PrunerModel | None, model_version: str):
        self._retriever = retriever
        self._inner = (
            Pruner(retriever=retriever, model=model, model_version=model_version)
            if model is not None else None
        )

    def prune(self, query: str, policy: Policy, k: int = RETRIEVAL_TOP_K) -> PruningResult:
        if self._inner is None:
            ranked = self._retriever.retrieve(query, k=k)
            return unpruned_fallback_result(
                query, ranked, policy, reason="no trained pruner artifact available")
        try:
            return self._inner.prune(query, policy, k=k)
        except Exception as e:  # noqa: BLE001 — conservative fallback must catch any pruning failure
            ranked = self._retriever.retrieve(query, k=k)
            return unpruned_fallback_result(
                query, ranked, policy,
                reason=f"pruner failed ({type(e).__name__}): {e}")
