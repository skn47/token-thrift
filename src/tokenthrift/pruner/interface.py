from __future__ import annotations

from tokenthrift.config import RETRIEVAL_TOP_K
from tokenthrift.core.tokenizer import count_tokens
from tokenthrift.core.types import Policy, PruningResult
from tokenthrift.features.extractor import extract_features
from tokenthrift.pruner.model import PrunerModel
from tokenthrift.retrieval.base import Retriever
from tokenthrift.safety.rules import apply_safety_rules


class Pruner:
    def __init__(self, retriever: Retriever, model: PrunerModel, model_version: str):
        self.retriever = retriever
        self.model = model
        self.model_version = model_version

    def prune(self, query: str, policy: Policy, k: int = RETRIEVAL_TOP_K) -> PruningResult:
        ranked = self.retriever.retrieve(query, k=k)
        baseline_tokens = sum(count_tokens(rc.chunk.text) for rc in ranked)

        if not ranked:
            return PruningResult(
                query=query, retained=(), pruned=(), policy=policy,
                pruning_enabled=True, disabled_reason=None,
                retained_tokens=0, baseline_tokens=0,
                model_version=self.model_version,
            )

        features = [extract_features(query, rc, ranked) for rc in ranked]
        score_values = self.model.predict_proba(features)
        scores = {rc.chunk.chunk_id: s for rc, s in zip(ranked, score_values)}

        retained, pruned, budget_conflict = apply_safety_rules(ranked, scores, policy)
        retained_tokens = sum(count_tokens(d.chunk.text) for d in retained)

        return PruningResult(
            query=query,
            retained=tuple(retained),
            pruned=tuple(pruned),
            policy=policy,
            pruning_enabled=True,
            disabled_reason=None,
            retained_tokens=retained_tokens,
            baseline_tokens=baseline_tokens,
            model_version=self.model_version,
            budget_conflict=budget_conflict,
        )
