from __future__ import annotations

import time

import streamlit as st

from tokenthrift.config import RETRIEVAL_TOP_K
from tokenthrift.core.tokenizer import count_tokens
from tokenthrift.core.types import Policy
from tokenthrift.corpus.documents import load_all_chunks
from tokenthrift.corpus.ingest import chunk_local_folder, resolve_folder_path
from tokenthrift.corpus.registry import CUSTOM_CORPUS_SENTINEL, resolve_corpus
from tokenthrift.generation.factory import build_client
from tokenthrift.generation.prompt import build_marked_prompt, build_prompt
from tokenthrift.generation.providers import PROVIDER_PRESETS
from tokenthrift.pricing.costs import estimate_cost
from tokenthrift.pruner.model import PrunerModel, resolve_latest_version
from tokenthrift.retrieval.tfidf_retriever import TfidfRetriever
from tokenthrift.safety.fallback import ModelLoadError, SafePruner, load_model_or_raise
from tokenthrift.safety.policy import default_policy
from tokenthrift.session.calibration import CalibrationState
from tokenthrift.session.sgd_adapter import CanaryExample, build_canary_set
from tokenthrift.session.state import SessionState
from tokenthrift.ui.comparison import (
    render_baseline_column,
    render_proxy_result,
    render_pruned_column,
)
from tokenthrift.ui.controls import render_feedback_controls
from tokenthrift.ui.history import render_adaptation_history
from tokenthrift.ui.metrics_panel import render_top_metrics
from tokenthrift.ui.proxy_client import call_through_proxy
from tokenthrift.ui.sidebar import render_sidebar
from tokenthrift.ui.theme import inject_theme


@st.cache_resource
def _load_retriever(corpus_id: str, custom_corpus_path: str) -> TfidfRetriever:
    if corpus_id == CUSTOM_CORPUS_SENTINEL:
        chunks = chunk_local_folder(resolve_folder_path(custom_corpus_path))
    else:
        chunks = load_all_chunks(resolve_corpus(corpus_id).corpus_dir)
    return TfidfRetriever(chunks)


@st.cache_resource
def _load_model(corpus_id: str) -> tuple[PrunerModel | None, str | None]:
    """Never raises: a missing/incompatible artifact is reported back as a
    reason string so the caller can fall back rather than crash the app."""
    try:
        artifacts_dir = resolve_corpus(corpus_id).artifacts_dir
        return load_model_or_raise(resolve_latest_version(artifacts_dir)), None
    except ModelLoadError as e:
        return None, str(e)


@st.cache_resource
def _load_canary(corpus_id: str) -> list[CanaryExample]:
    return build_canary_set(corpus_id=corpus_id)


def _get_session_state() -> SessionState:
    if "tt_session" not in st.session_state:
        st.session_state.tt_session = SessionState(
            calibration=CalibrationState(policy=default_policy()),
            model_version="latest",
        )
    return st.session_state.tt_session


def main() -> None:
    st.set_page_config(page_title="TokenThrift", page_icon="⚡", layout="wide")
    inject_theme()

    session = _get_session_state()
    inputs = render_sidebar(session)

    st.markdown("# ⚡ TokenThrift")
    st.caption("Same answer, a fraction of the context.")

    is_custom_corpus = inputs.corpus_id == CUSTOM_CORPUS_SENTINEL
    model_corpus_id = inputs.model_source_id if is_custom_corpus else inputs.corpus_id

    corpus_ready = True
    if is_custom_corpus:
        if not inputs.custom_corpus_path or not resolve_folder_path(inputs.custom_corpus_path).is_dir():
            corpus_ready = False
        corpus_blurb = "Your own folder — unverified, no measured quality for this content."
        query_label = "Ask about your folder's content"
    else:
        spec = resolve_corpus(inputs.corpus_id)
        corpus_blurb = f"Exploring **{spec.display_name}**: {spec.description}"
        query_label = f"Ask about {spec.display_name}"

    st.caption(corpus_blurb)
    with st.expander("How this comparison works"):
        st.write(
            "Both columns retrieve the same chunks and generate with the "
            "same model, prompt, and temperature — only context selection "
            "differs. Nothing is claimed as saved unless pruning actually ran.")

    query = st.text_input(query_label, key="query_input")
    run = st.button(
        "Compare", type="primary", key="compare_button",
        disabled=not query or not inputs.api_key or not corpus_ready)

    if is_custom_corpus and not corpus_ready:
        st.info("Enter a valid folder path in the sidebar to begin.")

    # Built every rerun (not only on Compare) so Regenerate/Retry/Accept/
    # Incorrect/Mark-irrelevant — each handled by its own button, not
    # Compare — can still call the provider and attempt a guarded SGD update.
    preset = PROVIDER_PRESETS[inputs.provider_id]
    client = (
        build_client(preset, inputs.base_url, inputs.api_key, inputs.model)
        if inputs.api_key else None
    )
    base_model, model_load_error = _load_model(model_corpus_id)
    canary = _load_canary(model_corpus_id)

    if run:
        if model_load_error is not None:
            st.warning(
                f"Pruning will be disabled for this run: {model_load_error}. "
                "Falling back to unpruned retrieval — no savings are claimed.")

        try:
            retriever = _load_retriever(
                inputs.corpus_id, inputs.custom_corpus_path or "")
        except ValueError as e:
            st.error(f"Could not read that folder: {e}")
            return
        # Once session calibration is on and a session-local SGD update has
        # been accepted, pruning uses the adapted classifier going forward
        # — never the shared base model, which stays untouched either way.
        active_model = base_model
        if inputs.calibration_enabled and session.cloned_model is not None:
            active_model = session.cloned_model
        safe_pruner = SafePruner(retriever=retriever, model=active_model, model_version="latest")

        if inputs.calibration_enabled:
            policy = session.policy
        else:
            policy = Policy(
                preset_name=inputs.preset_name,
                threshold=inputs.threshold,
                min_context=inputs.min_context,
                token_budget=inputs.token_budget,
            )

        with st.spinner("Retrieving and pruning..."):
            t0 = time.perf_counter()
            result = safe_pruner.prune(query, policy, k=RETRIEVAL_TOP_K)
            pruning_latency = time.perf_counter() - t0
            ranked = retriever.retrieve(query, k=RETRIEVAL_TOP_K)

        baseline_prompt = build_prompt(query, [rc.chunk for rc in ranked])
        pruned_prompt = build_prompt(query, [d.chunk for d in result.retained])

        with st.spinner("Generating (standard)..."):
            baseline_answer = client.generate(baseline_prompt)
        with st.spinner("Generating (pruned)..."):
            pruned_answer = client.generate(pruned_prompt)

        proxy_result = None
        if inputs.proxy_enabled:
            # Sends the full unpruned retrieval set, marked — the point is
            # to demonstrate the proxy's own generic TF-IDF scorer doing
            # real pruning server-side, not to re-send what the trained
            # model already pruned above.
            marked_prompt = build_marked_prompt(query, [rc.chunk for rc in ranked])
            with st.spinner("Generating (via live proxy)..."):
                proxy_result = call_through_proxy(
                    inputs.proxy_base_url, preset.wire_format,
                    inputs.api_key, inputs.model, marked_prompt)

        baseline_input_tokens = count_tokens(baseline_prompt)
        pruned_input_tokens = count_tokens(pruned_prompt)
        baseline_cost = estimate_cost(inputs.provider_id, inputs.model, baseline_input_tokens)
        pruned_cost = estimate_cost(inputs.provider_id, inputs.model, pruned_input_tokens)

        # Stashed in session_state (not a local variable) so the comparison
        # stays rendered across reruns triggered by *other* widgets —
        # Accept/Incorrect, the calibration toggle, Reset session — none of
        # which re-click Compare but all of which must not blank the page.
        st.session_state.last_result = {
            "query": query,
            "ranked": ranked,
            "baseline_answer": baseline_answer,
            "baseline_input_tokens": baseline_input_tokens,
            "baseline_cost": baseline_cost,
            "result": result,
            "pruned_answer": pruned_answer,
            "pruning_latency": pruning_latency,
            "pruned_cost": pruned_cost,
            "proxy_result": proxy_result,
        }
        st.session_state.pop("last_retry", None)

    if "last_result" not in st.session_state:
        st.info("Add an API key and a question, then hit Compare.")
        return

    data = st.session_state.last_result
    render_top_metrics(
        baseline_tokens=data["baseline_input_tokens"],
        pruned_tokens=data["result"].retained_tokens,
        baseline_cost=data["baseline_cost"],
        pruned_cost=data["pruned_cost"],
        pruning_latency_seconds=data["pruning_latency"],
        baseline_latency_seconds=data["baseline_answer"].latency_seconds,
        pruned_latency_seconds=data["pruned_answer"].latency_seconds,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        render_baseline_column(
            data["ranked"], data["baseline_answer"], data["baseline_input_tokens"])
    with col_b:
        render_pruned_column(data["result"], data["pruned_answer"])
        # Guarded SGD only ever engages while session calibration is on —
        # matches ui.controls.CONTROL_EXPLANATION's stated behavior exactly.
        sgd_base_model = base_model if inputs.calibration_enabled else None
        sgd_canary = canary if inputs.calibration_enabled else None
        render_feedback_controls(
            data["query"], client, session,
            base_model=sgd_base_model, canary=sgd_canary)

    proxy_result = data.get("proxy_result")
    if proxy_result is not None:
        render_proxy_result(proxy_result, data["baseline_input_tokens"])

    st.divider()
    render_adaptation_history(session)


main()
