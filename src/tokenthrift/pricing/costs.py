from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from tokenthrift.config import PRICING_PATH


@dataclass(frozen=True)
class CostEstimate:
    input_cost_usd: float
    output_cost_usd: float
    pricing_version: str
    known: bool = True

    @property
    def total_cost_usd(self) -> float:
        return self.input_cost_usd + self.output_cost_usd


def load_pricing_table(path: Path = PRICING_PATH) -> dict:
    return json.loads(path.read_text())


def estimate_cost(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int = 0,
    pricing_table: dict | None = None,
) -> CostEstimate:
    """Never raises: an unrecognized provider/model (expected for custom
    endpoints and most OpenRouter models, whose catalogs we don't curate
    prices for) returns known=False with zeroed dollar fields rather than
    a fabricated cost — showing $0.00 would be a false savings claim, so
    callers must render "cost n/a" instead of a number when known=False."""
    table = pricing_table if pricing_table is not None else load_pricing_table()
    rates = table.get("providers", {}).get(provider, {}).get("models", {}).get(model)
    if rates is None:
        return CostEstimate(
            input_cost_usd=0.0, output_cost_usd=0.0,
            pricing_version=table["pricing_version"], known=False,
        )
    input_cost = input_tokens / 1_000_000 * rates["input_per_million"]
    output_cost = output_tokens / 1_000_000 * rates["output_per_million"]
    return CostEstimate(
        input_cost_usd=input_cost,
        output_cost_usd=output_cost,
        pricing_version=table["pricing_version"],
        known=True,
    )
