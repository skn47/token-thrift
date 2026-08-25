from __future__ import annotations

import streamlit as st

from tokenthrift.pricing.costs import CostEstimate


def render_top_metrics(
    baseline_tokens: int,
    pruned_tokens: int,
    baseline_cost: CostEstimate,
    pruned_cost: CostEstimate,
    pruning_latency_seconds: float,
    baseline_latency_seconds: float,
    pruned_latency_seconds: float,
) -> None:
    pct_saved = 0.0
    if baseline_tokens > 0:
        pct_saved = max(baseline_tokens - pruned_tokens, 0) / baseline_tokens

    cols = st.columns(4)
    cols[0].metric(
        "Context tokens saved", f"{pct_saved:.0%}",
        f"{baseline_tokens} -> {pruned_tokens}")
    if baseline_cost.known and pruned_cost.known:
        cost_saved = max(
            baseline_cost.input_cost_usd - pruned_cost.input_cost_usd, 0.0)
        cols[1].metric(
            "Estimated input cost saved", f"${cost_saved:.5f}",
            help=f"Pricing version: {pruned_cost.pricing_version}")
    else:
        cols[1].metric(
            "Estimated input cost saved", "n/a",
            help="No sourced price for this provider/model — cost isn't "
                 "estimated rather than shown as zero.")
    cols[2].metric(
        "Pruning latency", f"{pruning_latency_seconds * 1000:.0f} ms")
    cols[3].metric(
        "Generation latency (this run)", f"{pruned_latency_seconds:.2f}s",
        f"baseline {baseline_latency_seconds:.2f}s")
    st.caption(
        "Latency figures are from a single run, not a repeated-run median "
        "or percentile comparison — see the held-out evaluation report "
        "(scripts/train_model.py output) for repeated-run statistics.")
