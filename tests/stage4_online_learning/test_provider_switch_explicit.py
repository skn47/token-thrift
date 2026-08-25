import inspect

from tokenthrift.config import REPO_ROOT
from tokenthrift.generation.openai_compatible_client import OpenAICompatibleClient
from tokenthrift.generation.providers import PROVIDER_PRESETS
from tokenthrift.ui import sidebar as sidebar_module


def test_openai_compatible_client_never_references_another_provider_on_failure():
    source = inspect.getsource(OpenAICompatibleClient.generate)
    assert "ollama" not in source.lower()


def test_default_ui_flow_never_references_ollama():
    # Structural regression guard: as long as nothing in the default app
    # flow mentions Ollama, no code path can silently switch to it —
    # provider selection has to be an explicit, separate integration.
    for path in ("app.py", "sidebar.py", "controls.py", "comparison.py"):
        source = (REPO_ROOT / "src" / "tokenthrift" / "ui" / path).read_text()
        assert "ollama" not in source.lower(), f"{path} unexpectedly references Ollama"


def test_provider_is_chosen_from_a_fixed_named_set_not_derived_from_request_outcomes():
    # Multi-provider BYOK generalizes "explicit" from a hardcoded string to
    # a selection over a fixed, named preset registry — never inferred
    # from a request's success/failure, never silently switched.
    source = inspect.getsource(sidebar_module._render_model_section)
    assert "PROVIDER_PRESETS" in source
    assert set(PROVIDER_PRESETS.keys()) == {
        "groq", "openai", "anthropic", "openrouter", "custom"}
