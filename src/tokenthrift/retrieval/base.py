from __future__ import annotations

from typing import Protocol

from tokenthrift.core.types import RankedChunk


class Retriever(Protocol):
    def retrieve(self, query: str, k: int) -> list[RankedChunk]: ...
