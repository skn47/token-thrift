from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Answer:
    text: str
    latency_seconds: float
    input_tokens: int | None
    output_tokens: int | None
    model: str
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None
