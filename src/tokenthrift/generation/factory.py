from __future__ import annotations

from typing import Protocol

from tokenthrift.generation.answer import Answer
from tokenthrift.generation.anthropic_client import AnthropicClient
from tokenthrift.generation.openai_compatible_client import OpenAICompatibleClient
from tokenthrift.generation.providers import WIRE_ANTHROPIC, WIRE_OPENAI, ProviderPreset


class ChatClient(Protocol):
    model: str

    def generate(self, prompt: str, temperature: float = 0.0) -> Answer: ...


def build_client(
    preset: ProviderPreset, base_url: str, api_key: str, model: str,
) -> ChatClient:
    """Single dispatch point from a chosen provider preset to a concrete
    client — the only place in the codebase that needs to know how many
    wire formats exist, so the UI never has to branch on provider identity
    itself."""
    if preset.wire_format == WIRE_ANTHROPIC:
        return AnthropicClient(api_key=api_key, model=model, base_url=base_url)
    if preset.wire_format == WIRE_OPENAI:
        return OpenAICompatibleClient(base_url=base_url, api_key=api_key, model=model)
    raise ValueError(f"unknown wire_format {preset.wire_format!r} for {preset.provider_id!r}")
