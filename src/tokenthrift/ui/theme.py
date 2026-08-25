from __future__ import annotations

import streamlit as st

_CSS = """
<style>
:root {
    --tt-accent: #6c5ce7;
    --tt-accent-soft: rgba(108, 92, 231, 0.10);
    --tt-good: #1e8e5a;
    --tt-good-soft: rgba(30, 142, 90, 0.10);
}
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter,
        Roboto, Helvetica, Arial, sans-serif;
}
h1, h2, h3 { letter-spacing: -0.01em; }
[data-testid="stMetric"] {
    background: var(--tt-accent-soft);
    border-radius: 14px;
    padding: 0.85rem 1.1rem;
}
[data-testid="stMetricLabel"] { font-weight: 600; opacity: 0.85; }
[data-testid="stMetricDelta"] svg { display: none; }
button[kind="primary"] {
    border-radius: 999px;
    font-weight: 600;
}
[data-testid="stExpander"], [data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 14px;
}
[data-testid="stTabs"] button[role="tab"] {
    border-radius: 8px 8px 0 0;
}
</style>
"""


def inject_theme() -> None:
    """One-shot CSS injection for a warmer, more consumer-facing look.
    Purely presentational — no widget behavior changes, so every existing
    key-driven AppTest interaction is unaffected."""
    st.markdown(_CSS, unsafe_allow_html=True)
