from __future__ import annotations

import streamlit as st

from tokenthrift.session.state import SessionState


def render_adaptation_history(session: SessionState) -> None:
    st.subheader("Session activity")
    cols = st.columns(3)
    cols[0].metric(
        "Accepted SGD updates", session.accepted_updates,
        help="Guarded live-learning updates applied this session.")
    cols[1].metric(
        "Rejected SGD updates", session.rejected_updates,
        help="Updates the safety guard blocked before they took effect.")
    cols[2].metric("Session labels", len(session.labels))

    if not session.adaptation_history:
        st.caption("No activity yet this session.")
        return

    with st.expander(
        f"Full activity log ({len(session.adaptation_history)} event(s))",
    ):
        st.caption(
            "Every feedback action is logged with its reason, even when "
            "it changed nothing.")
        for entry in reversed(session.adaptation_history[-25:]):
            if entry.startswith("session-local SGD update rejected"):
                st.error(entry)
            elif entry.startswith("session-local SGD update accepted"):
                st.success(entry)
            else:
                st.text(entry)

        if session.labels:
            st.divider()
            st.caption(f"Label store ({len(session.labels)}):")
            for label in session.labels:
                st.text(
                    f"{label.chunk_id}: relevant={label.relevant} "
                    f"weight={label.weight} confidence={label.confidence} "
                    f"provenance={label.provenance}")
