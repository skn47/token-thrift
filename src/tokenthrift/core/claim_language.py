"""Shared source of truth for banned absolute-claim phrasing (cross-cutting
constraint: TokenThrift reports targets and measured outcomes, never
guarantees, until a real held-out run backs a specific number)."""

BANNED_PHRASES = (
    "slashes cost by",
    "guarantees",
    "guaranteed",
    "proves correct",
    "100% accurate",
    "always correct",
    "risk-free",
    "no downside",
)


def find_banned_claim(text: str) -> str | None:
    lowered = text.lower()
    for phrase in BANNED_PHRASES:
        if phrase in lowered:
            return phrase
    return None
