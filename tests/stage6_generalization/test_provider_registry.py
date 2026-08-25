from tokenthrift.generation.factory import build_client
from tokenthrift.generation.providers import (
    DEFAULT_PROVIDER_ID,
    PROVIDER_PRESETS,
    WIRE_ANTHROPIC,
    WIRE_OPENAI,
)


def test_default_provider_is_groq_and_present_in_the_registry():
    assert DEFAULT_PROVIDER_ID in PROVIDER_PRESETS
    assert DEFAULT_PROVIDER_ID == "groq"


def test_every_preset_has_a_valid_shape():
    for provider_id, preset in PROVIDER_PRESETS.items():
        assert preset.provider_id == provider_id
        assert preset.wire_format in (WIRE_OPENAI, WIRE_ANTHROPIC)
        assert preset.display_name
        if not preset.editable_base_url:
            assert preset.base_url.startswith("https://")
            assert preset.default_model


def test_only_the_custom_preset_has_an_editable_base_url():
    for provider_id, preset in PROVIDER_PRESETS.items():
        if provider_id == "custom":
            assert preset.editable_base_url is True
            assert preset.base_url == ""
        else:
            assert preset.editable_base_url is False


def test_build_client_dispatches_on_wire_format():
    from tokenthrift.generation.anthropic_client import AnthropicClient
    from tokenthrift.generation.openai_compatible_client import OpenAICompatibleClient

    openai_client = build_client(
        PROVIDER_PRESETS["groq"], PROVIDER_PRESETS["groq"].base_url,
        "key", "some-model")
    assert isinstance(openai_client, OpenAICompatibleClient)

    anthropic_client = build_client(
        PROVIDER_PRESETS["anthropic"], PROVIDER_PRESETS["anthropic"].base_url,
        "key", "some-model")
    assert isinstance(anthropic_client, AnthropicClient)
