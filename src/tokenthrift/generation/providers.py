from __future__ import annotations

from dataclasses import dataclass

WIRE_OPENAI = "openai"
WIRE_ANTHROPIC = "anthropic"


@dataclass(frozen=True)
class ProviderPreset:
    provider_id: str
    display_name: str
    wire_format: str          # "openai" | "anthropic"
    base_url: str             # "" for the custom preset (user-supplied)
    default_model: str
    env_var: str | None       # convenience API-key default source
    editable_base_url: bool = False


PROVIDER_PRESETS: dict[str, ProviderPreset] = {
    "groq": ProviderPreset(
        "groq", "Groq", WIRE_OPENAI,
        "https://api.groq.com/openai/v1", "openai/gpt-oss-20b", "GROQ_API_KEY"),
    "openai": ProviderPreset(
        "openai", "OpenAI", WIRE_OPENAI,
        "https://api.openai.com/v1", "gpt-4o-mini", "OPENAI_API_KEY"),
    "anthropic": ProviderPreset(
        "anthropic", "Anthropic", WIRE_ANTHROPIC,
        "https://api.anthropic.com", "claude-sonnet-5", "ANTHROPIC_API_KEY"),
    "openrouter": ProviderPreset(
        "openrouter", "OpenRouter", WIRE_OPENAI,
        "https://openrouter.ai/api/v1", "openai/gpt-oss-20b", "OPENROUTER_API_KEY"),
    "custom": ProviderPreset(
        "custom", "Custom (OpenAI-compatible)", WIRE_OPENAI,
        "", "", None, editable_base_url=True),
}

DEFAULT_PROVIDER_ID = "groq"
