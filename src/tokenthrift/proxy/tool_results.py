from __future__ import annotations

from tokenthrift.proxy.chunking import MARKER_CLOSE, MARKER_OPEN


def wrap_as_marked(text: str) -> str:
    """Force-treats `text` as one prunable block for auto-marked tool-result
    content, unless it already contains a marker — avoids double-wrapping
    when a caller manually marked part of a tool result too."""
    if MARKER_OPEN in text:
        return text
    return f"{MARKER_OPEN}\n{text}\n{MARKER_CLOSE}"
