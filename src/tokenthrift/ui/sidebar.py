from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass

import streamlit as st

from tokenthrift.corpus.ingest import resolve_folder_path
from tokenthrift.corpus.registry import CUSTOM_CORPUS_SENTINEL, list_bundled_corpora
from tokenthrift.generation.providers import DEFAULT_PROVIDER_ID, PROVIDER_PRESETS
from tokenthrift.safety.policy import PRESETS
from tokenthrift.session.calibration import MAX_THRESHOLD, MIN_THRESHOLD
from tokenthrift.session.state import SessionState
from tokenthrift.ui.proxy_client import check_health, set_auto_mark_tool_results

DEFAULT_PROXY_URL = "http://localhost:8787"


@dataclass(frozen=True)
class SidebarInputs:
    api_key: str
    provider_id: str
    base_url: str
    model: str
    preset_name: str
    threshold: float
    min_context: int
    token_budget: int
    calibration_enabled: bool
    corpus_id: str
    custom_corpus_path: str | None
    model_source_id: str | None
    proxy_base_url: str
    proxy_enabled: bool


def _browse_for_folder() -> None:
    """Runs on the machine hosting this Streamlit process, not the
    viewer's browser — only meaningful when running TokenThrift locally,
    which is how this app is used today."""
    try:
        result = subprocess.run(
            ["zenity", "--file-selection", "--directory", "--title=Choose a folder"],
            capture_output=True, text=True, timeout=120, check=False)
    except (subprocess.TimeoutExpired, OSError):
        return
    selected = result.stdout.strip()
    if selected:
        st.session_state["custom_corpus_path"] = selected


def _render_corpus_section() -> tuple[str, str | None, str | None]:
    bundled = list_bundled_corpora()
    labels = {c.corpus_id: c.display_name for c in bundled}
    labels[CUSTOM_CORPUS_SENTINEL] = "My own folder"

    with st.sidebar.container(border=True):
        st.markdown("**📚 Knowledge base**")
        corpus_id = st.selectbox(
            "Corpus", options=[*labels.keys()],
            format_func=lambda cid: labels[cid], key="corpus_select",
            label_visibility="collapsed")

        custom_corpus_path: str | None = None
        model_source_id: str | None = None
        if corpus_id == CUSTOM_CORPUS_SENTINEL:
            st.caption("Folder path")
            path_col, browse_col = st.columns([6, 1], vertical_alignment="top")
            with path_col:
                custom_corpus_path = st.text_input(
                    "Folder path", key="custom_corpus_path",
                    label_visibility="collapsed",
                    placeholder="/path/to/your/docs (.md or .txt)")
            with browse_col:
                zenity_available = shutil.which("zenity") is not None
                st.button(
                    "📁", key="browse_folder_button",
                    on_click=_browse_for_folder, disabled=not zenity_available,
                    help=(
                        "Browse for a folder — opens a native picker on the "
                        "machine running this app (only useful when running "
                        "TokenThrift locally)." if zenity_available else
                        "Native folder picker needs the 'zenity' package, "
                        "not found on this machine — type the path instead."))

            if custom_corpus_path:
                resolved = resolve_folder_path(custom_corpus_path)
                if resolved.is_dir():
                    st.caption(f"✅ {resolved}")
                else:
                    st.caption(f"❌ Not a folder: {resolved}")

            model_source_id = st.selectbox(
                "Pruner to borrow", options=[c.corpus_id for c in bundled],
                format_func=lambda cid: labels[cid], key="model_source_select",
                help="Your own folder has no relevance labels, so pruning "
                     "borrows a model trained on one of the corpora above.")
            st.caption("⚠️ No ground truth for this content — unverified, live look only.")
        else:
            spec = next(c for c in bundled if c.corpus_id == corpus_id)
            st.caption(spec.description)

    return corpus_id, custom_corpus_path, model_source_id


def _apply_provider_defaults(provider_id: str, preset) -> None:
    if st.session_state.get("_last_provider_id") != provider_id:
        st.session_state["model_input"] = preset.default_model

        api_keys = st.session_state.setdefault("_api_keys_by_provider", {})
        if provider_id not in api_keys:
            api_keys[provider_id] = (
                os.environ.get(preset.env_var, "") if preset.env_var else "")
        st.session_state["api_key_input"] = api_keys[provider_id]

        st.session_state["_last_provider_id"] = provider_id


def _render_model_section() -> tuple[str, str, str, str]:
    with st.sidebar.container(border=True):
        st.markdown("**🤖 Model**")
        provider_ids = list(PROVIDER_PRESETS.keys())
        provider_id = st.selectbox(
            "Provider", options=provider_ids,
            format_func=lambda pid: PROVIDER_PRESETS[pid].display_name,
            index=provider_ids.index(DEFAULT_PROVIDER_ID), key="provider_select",
            label_visibility="collapsed")
        preset = PROVIDER_PRESETS[provider_id]
        _apply_provider_defaults(provider_id, preset)

        if preset.editable_base_url:
            base_url = st.text_input(
                "Base URL", key="base_url_input",
                placeholder="https://your-endpoint.example/v1")
        else:
            base_url = preset.base_url
            st.caption(f"Endpoint: `{base_url}`")

        model = st.text_input("Model", key="model_input")

        api_key = st.text_input(
            "API key", type="password", key="api_key_input",
            placeholder="Paste your key",
            help="Kept only in this session's memory — never written to "
                 "disk, a database, analytics, or logs.")
        st.session_state.setdefault("_api_keys_by_provider", {})[provider_id] = api_key

    return provider_id, base_url, model, api_key


def _render_pruning_section() -> tuple[str, float, int, int, bool]:
    with st.sidebar.container(border=True):
        st.markdown("**✂️ Pruning**")
        preset_names = list(PRESETS.keys())
        preset_name = st.radio(
            "Preset", options=preset_names,
            index=preset_names.index("balanced"), horizontal=True)
        preset = PRESETS[preset_name]

        calibration_enabled = st.toggle(
            "Auto-tune from my feedback", value=False, key="calibration_enabled",
            help="Learns from Accept/Incorrect within safe bounds instead "
                 "of using the fixed sliders below.")

        with st.expander("Fine-tune manually", expanded=False):
            if calibration_enabled:
                st.caption("Ignored while auto-tuning is on.")
            threshold = st.slider(
                "Relevance threshold", 0.0, 1.0, float(preset.threshold), 0.01,
                disabled=calibration_enabled)
            min_context = st.number_input(
                "Minimum chunks kept", min_value=1, max_value=10,
                value=preset.min_context, disabled=calibration_enabled)
            token_budget = st.number_input(
                "Token budget", min_value=200, max_value=8000,
                value=preset.token_budget, step=100, disabled=calibration_enabled)

    return preset_name, float(threshold), int(min_context), int(token_budget), calibration_enabled


def _render_session_section(
    session: SessionState, calibration_enabled: bool, preset,
) -> None:
    with st.sidebar.container(border=True):
        st.markdown("**🧠 This session**")
        calibrated = session.policy
        st.caption(
            f"Auto-tune {'on' if calibration_enabled else 'off'} · "
            f"threshold {calibrated.threshold:.2f} · "
            f"{len(session.adaptation_history)} event(s)",
            help=f"Bounds: threshold [{MIN_THRESHOLD:.2f}, {MAX_THRESHOLD:.2f}]. "
                 f"Preset default: {preset.threshold:.2f}. Full log below "
                 "the comparison.")
        if session.sgd_active:
            st.caption(
                f"⚡ Experimental live-learning active "
                f"({session.accepted_updates} accepted, "
                f"{session.rejected_updates} rejected)",
                help="A session-local model update is in effect. Reset "
                     "restores the shared base model exactly.")
        elif session.rejected_updates:
            st.caption(
                f"Experimental live-learning: {session.rejected_updates} "
                "update(s) rejected by its safety guard so far.")

        if st.button("↺ Reset session", key="reset_session_button"):
            session.reset()
            st.rerun()


def _sync_auto_mark_toggle() -> None:
    """on_change callback for the auto-mark toggle — pushes the new value to
    the running proxy immediately (rather than waiting for some other
    widget interaction to notice it changed), so the checkbox is the actual
    write path, not just local UI state."""
    proxy_base_url = st.session_state.get("proxy_base_url_input", DEFAULT_PROXY_URL)
    enabled = st.session_state["proxy_auto_mark_toggle"]
    if set_auto_mark_tool_results(proxy_base_url, enabled):
        st.session_state.pop("_proxy_auto_mark_error", None)
    else:
        st.session_state["_proxy_auto_mark_error"] = True


def _render_proxy_section() -> tuple[str, bool]:
    with st.sidebar.container(border=True):
        st.markdown("**🔌 Proxy (optional)**")
        proxy_base_url = st.text_input(
            "Proxy URL", value=DEFAULT_PROXY_URL, key="proxy_base_url_input")
        proxy_enabled = st.toggle(
            "Also run this query through the proxy, live", value=False,
            key="proxy_enabled_toggle",
            help="Sends the same question through a running TokenThrift "
                 "proxy in front of your provider — the same call a coding "
                 "agent pointed at this proxy would make — and shows what "
                 "it actually pruned.")
        st.toggle(
            "Auto-mark tool results (Codex, Claude Code, etc.)", value=False,
            key="proxy_auto_mark_toggle", on_change=_sync_auto_mark_toggle,
            help="Prunes tool-result content (file reads, command output, "
                 "doc lookups) on the running proxy without needing "
                 "<tokenthrift:context> markers. Applies to every request "
                 "the proxy handles, not just this panel — plain pasted "
                 "text outside a tool result still needs manual markers.")
        if st.session_state.get("_proxy_auto_mark_error"):
            st.caption("⚠️ Could not reach the proxy to update this setting.")
        if st.button("Check connection", key="proxy_health_button"):
            health = check_health(proxy_base_url)
            if health.reachable:
                upstream_status = (
                    "configured" if health.upstream_configured else "NOT configured")
                auto_mark_status = "on" if health.auto_mark_tool_results else "off"
                st.success(
                    f"Reachable · upstream {upstream_status} · "
                    f"policy: {health.policy_preset} · auto-mark: {auto_mark_status}")
            else:
                st.error(f"Unreachable: {health.error}")

    return proxy_base_url, proxy_enabled


def render_sidebar(session: SessionState) -> SidebarInputs:
    st.sidebar.markdown("## ⚡ TokenThrift")
    st.sidebar.caption("Trim your RAG context without losing the answer.")

    corpus_id, custom_corpus_path, model_source_id = _render_corpus_section()
    provider_id, base_url, model, api_key = _render_model_section()
    preset_name, threshold, min_context, token_budget, calibration_enabled = (
        _render_pruning_section())
    _render_session_section(session, calibration_enabled, PRESETS[preset_name])
    proxy_base_url, proxy_enabled = _render_proxy_section()

    return SidebarInputs(
        api_key=api_key,
        provider_id=provider_id,
        base_url=base_url,
        model=model,
        preset_name=preset_name,
        threshold=threshold,
        min_context=min_context,
        token_budget=token_budget,
        calibration_enabled=calibration_enabled,
        corpus_id=corpus_id,
        custom_corpus_path=custom_corpus_path,
        model_source_id=model_source_id,
        proxy_base_url=proxy_base_url,
        proxy_enabled=proxy_enabled,
    )
