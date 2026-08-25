from __future__ import annotations

import streamlit as st

from tokenthrift.core.types import PruningResult, RankedChunk
from tokenthrift.generation.answer import Answer
from tokenthrift.ui.proxy_client import ProxyCallResult


def _chunk_label(chunk) -> str:
    return f"{chunk.doc_title} - {chunk.heading or chunk.chunk_id}"


def render_baseline_column(
    ranked_chunks: list[RankedChunk], answer: Answer, input_tokens: int,
) -> None:
    st.subheader("Without TokenThrift")
    st.caption(f"{len(ranked_chunks)} chunks · ~{input_tokens} input tokens")
    if answer.succeeded:
        st.markdown(answer.text)
    else:
        st.error(f"Generation failed: {answer.error}")
    with st.expander("Context sent to the model"):
        for rc in ranked_chunks:
            st.text(f"[{_chunk_label(rc.chunk)}]\n{rc.chunk.text}")


def render_pruned_column(result: PruningResult, answer: Answer) -> None:
    st.subheader("With TokenThrift")
    if not result.pruning_enabled:
        st.warning(
            f"Pruning disabled: {result.disabled_reason}. Showing unpruned "
            f"results — no token savings are claimed for this run.")
    if result.budget_conflict:
        st.warning(
            "A mandatory chunk (top result or minimum-context fill) "
            "exceeded the token budget and was kept anyway rather than "
            "silently cut — raise the budget or lower the minimum to "
            "resolve this.")
    st.caption(
        f"{len(result.retained)} kept / {len(result.pruned)} pruned · "
        f"~{result.retained_tokens} input tokens")
    if answer.succeeded:
        st.markdown(answer.text)
    else:
        st.error(f"Generation failed: {answer.error}")

    tab_kept, tab_pruned = st.tabs([
        f"Kept ({len(result.retained)})", f"Pruned ({len(result.pruned)})"])
    with tab_kept:
        if not result.retained:
            st.text("(none retained)")
        for d in result.retained:
            score_text = (
                f"score {d.relevance_score:.2f}"
                if d.relevance_score is not None else "score n/a")
            label = f"KEPT ({score_text}) — {', '.join(d.reasons)}"
            body = f"[{_chunk_label(d.chunk)}]\n{d.chunk.text}"
            if d.safety_override is not None:
                st.warning(f"{label}\n\n{body}")
            else:
                st.success(f"{label}\n\n{body}")

    with tab_pruned:
        if not result.pruned:
            st.text("(none pruned)")
        for d in result.pruned:
            score_text = (
                f"score {d.relevance_score:.2f}"
                if d.relevance_score is not None else "score n/a")
            label = f"PRUNED ({score_text}) — {', '.join(d.reasons)}"
            body = f"[{_chunk_label(d.chunk)}]\n{d.chunk.text}"
            st.error(f"{label}\n\n{body}")


def render_proxy_result(result: ProxyCallResult, baseline_input_tokens: int) -> None:
    """Full-width panel below the baseline/pruned columns — shows what a
    real HTTP call through a running proxy actually did, using the proxy's
    own generic TF-IDF scorer rather than the trained classifier those two
    columns use. A failed/unreachable proxy never touches the columns
    above; this is the only place its outcome is shown."""
    with st.container(border=True):
        st.subheader("🔌 Live via TokenThrift Proxy")
        if result.error is not None:
            st.error(f"Proxy call failed: {result.error}")
            return
        tokens_pruned = result.tokens_pruned or 0
        st.caption(
            f"{tokens_pruned} token(s) pruned by the proxy · "
            f"baseline was ~{baseline_input_tokens} input tokens · "
            f"{result.latency_seconds:.2f}s")
        st.markdown(result.text)
        st.caption(
            "This ran through a real HTTP call to your TokenThrift proxy — "
            "the same request a coding agent pointed at this proxy would "
            "send.")
