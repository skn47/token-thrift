import pytest

from tokenthrift.config import REPO_ROOT
from tokenthrift.generation.ollama_client import OllamaClient, is_ollama_reachable


def test_is_ollama_reachable_never_raises_and_returns_a_bool():
    # Whether or not a local Ollama server happens to be running in this
    # environment, the check itself must never raise or hang.
    result = is_ollama_reachable(base_url="http://localhost:11434", timeout=0.5)
    assert isinstance(result, bool)


def test_is_ollama_reachable_returns_false_for_an_unroutable_host_without_raising():
    assert is_ollama_reachable(base_url="http://192.0.2.1:11434", timeout=0.2) is False


def test_generate_against_a_closed_port_surfaces_an_error_not_a_crash():
    client = OllamaClient(base_url="http://localhost:1")  # nothing listens on port 1
    answer = client.generate("hello")
    assert not answer.succeeded
    assert answer.error is not None
    assert answer.text == ""


def _first_available_ollama_model() -> str | None:
    import httpx
    try:
        response = httpx.get("http://localhost:11434/api/tags", timeout=0.5)
        models = response.json().get("models", [])
        return models[0]["name"] if models else None
    except httpx.HTTPError:
        return None


@pytest.mark.skipif(
    not is_ollama_reachable(), reason="no local Ollama server available")
def test_generate_against_a_real_local_ollama_server_succeeds():
    # Only runs when this environment actually has Ollama reachable —
    # exercises the real happy path rather than only the failure path.
    model = _first_available_ollama_model()
    assert model is not None, "Ollama is reachable but has no pulled models"
    client = OllamaClient(model=model)
    answer = client.generate("Reply with exactly one word: hello")
    assert answer.succeeded
    assert answer.error is None
    assert answer.text != ""


def test_ollama_is_never_imported_by_the_default_ui_flow():
    # Structural check: provider selection must be explicit, never
    # automatic — the default app module must not reference Ollama at all,
    # regardless of whether a local server happens to be reachable.
    app_source = (REPO_ROOT / "src" / "tokenthrift" / "ui" / "app.py").read_text()
    assert "ollama" not in app_source.lower()
