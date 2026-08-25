from __future__ import annotations

import streamlit as st

from tokenthrift.features.extractor import extract_features
from tokenthrift.feedback.attribution import attribute
from tokenthrift.feedback.events import (
    AcceptAnswer,
    MarkChunkIrrelevant,
    RegenerateSameContext,
    RetryFullContext,
    ThumbsDown,
)
from tokenthrift.generation.factory import ChatClient
from tokenthrift.generation.prompt import build_prompt
from tokenthrift.pruner.model import PrunerModel
from tokenthrift.session.sgd_adapter import CanaryExample
from tokenthrift.session.state import SessionState
from tokenthrift.validation.counterfactual import compare_counterfactual
from tokenthrift.validation.pipeline import run_layered_check

CONTROL_EXPLANATION = (
    "Regenerate changes only generation. Retry and Accept/Incorrect may "
    "adjust the pruning policy. Retry and Mark-irrelevant can also feed "
    "an experimental session-local model update — bounded, checked, and "
    "reset along with everything else."
)


def render_feedback_controls(
    query: str,
    client: ChatClient | None,
    session: SessionState,
    base_model: PrunerModel | None = None,
    canary: list[CanaryExample] | None = None,
) -> None:
    data = st.session_state.get("last_result")
    if data is None:
        return

    result = data["result"]
    pruned_answer = data["pruned_answer"]
    ranked = data["ranked"]
    can_generate = client is not None
    rc_by_chunk_id = {rc.chunk.chunk_id: rc for rc in ranked}

    st.caption(CONTROL_EXPLANATION)
    cols = st.columns(4)

    if cols[0].button(
        "🔁 Regenerate", key="fb_regenerate", disabled=not can_generate,
        help="Same context, a fresh model call.",
    ):
        pruned_prompt = build_prompt(query, [d.chunk for d in result.retained])
        new_answer = client.generate(pruned_prompt)
        decision = attribute(RegenerateSameContext())
        session.apply_adaptation(decision, base_model, canary)
        st.session_state.last_result["pruned_answer"] = new_answer
        st.rerun()

    if cols[1].button(
        "🧩 Retry with full context", key="fb_retry", disabled=not can_generate,
        help="Checks whether missing evidence caused the problem.",
    ):
        full_context_chunks = [rc.chunk for rc in ranked]
        full_prompt = build_prompt(query, full_context_chunks)
        restored_answer = client.generate(full_prompt)
        cf = compare_counterfactual(
            pruned_answer.text, result, restored_answer.text, full_context_chunks)
        features_by_chunk_id = {
            cid: extract_features(query, rc_by_chunk_id[cid], ranked)
            for cid in cf.cited_restored_chunk_ids
            if cid in rc_by_chunk_id
        }
        decision = attribute(RetryFullContext(
            improved=cf.improved,
            cited_restored_chunk_ids=cf.cited_restored_chunk_ids,
            features_by_chunk_id=features_by_chunk_id))
        session.apply_adaptation(decision, base_model, canary)
        st.session_state.last_retry = {
            "answer": restored_answer, "counterfactual": cf,
            "reason": decision.reason,
        }
        st.rerun()

    if cols[2].button("👍 Accept", key="fb_accept"):
        retained_chunks = [d.chunk for d in result.retained]
        validation = run_layered_check(query, pruned_answer.text, retained_chunks)
        decision = attribute(AcceptAnswer(grounded=validation.grounding.is_grounded))
        session.apply_adaptation(decision, base_model, canary)
        st.rerun()

    if cols[3].button("👎 Incorrect", key="fb_incorrect"):
        decision = attribute(ThumbsDown())
        session.apply_adaptation(decision, base_model, canary)
        st.rerun()

    chunk_ids = [d.chunk.chunk_id for d in result.retained]
    if chunk_ids:
        with st.expander("A kept chunk shouldn't have been"):
            selected = st.selectbox("Chunk", chunk_ids, key="fb_mark_chunk_select")
            if st.button("Mark irrelevant", key="fb_mark_irrelevant"):
                features = None
                if selected in rc_by_chunk_id:
                    features = extract_features(query, rc_by_chunk_id[selected], ranked)
                decision = attribute(
                    MarkChunkIrrelevant(chunk_id=selected, features=features))
                session.apply_adaptation(decision, base_model, canary)
                st.rerun()

    retry_info = st.session_state.get("last_retry")
    if retry_info is not None:
        cf = retry_info["counterfactual"]
        cited = ", ".join(sorted(cf.cited_restored_chunk_ids)) or "none"
        if cf.improved:
            st.success(
                f"Retry improved the answer using restored evidence "
                f"(cited: {cited}). {retry_info['reason']}")
        else:
            st.info(f"Retry did not show pruning was the cause. {retry_info['reason']}")
        with st.expander("Full-context retry answer"):
            answer = retry_info["answer"]
            st.markdown(answer.text if answer.succeeded else f"(generation failed: {answer.error})")
