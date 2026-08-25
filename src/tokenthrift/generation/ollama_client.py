from __future__ import annotations

import time

import httpx

from tokenthrift.generation.answer import Answer

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.1"


def is_ollama_reachable(base_url: str = DEFAULT_OLLAMA_URL, timeout: float = 0.5) -> bool:
    """Never called automatically — only when the user has explicitly
    selected Ollama as the provider. A hosted Streamlit server generally
    cannot reach a visitor's local Ollama process without a separate local
    bridge (per the reference doc's stretch-goal caveat); this fails
    gracefully to False rather than hanging or raising."""
    try:
        response = httpx.get(f"{base_url}/api/tags", timeout=timeout)
        return response.status_code == 200
    except httpx.HTTPError:
        return False


class OllamaClient:
    """Local-only generation client with the same Answer-returning
    interface as GroqClient. Callers must check is_ollama_reachable() (or
    handle a failed Answer) before relying on this — there is no automatic
    fallback to or from Groq anywhere in this client; provider selection
    is always an explicit, user-driven choice made in the UI, never a
    silent switch."""

    def __init__(self, base_url: str = DEFAULT_OLLAMA_URL, model: str = DEFAULT_MODEL):
        self.base_url = base_url
        self.model = model

    def generate(self, prompt: str, temperature: float = 0.0) -> Answer:
        start = time.perf_counter()
        try:
            response = httpx.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model, "prompt": prompt, "stream": False,
                    "options": {"temperature": temperature},
                },
                timeout=60.0,
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPError as e:
            latency = time.perf_counter() - start
            return Answer(
                text="", latency_seconds=latency, input_tokens=None,
                output_tokens=None, model=self.model, error=str(e),
            )

        latency = time.perf_counter() - start
        return Answer(
            text=payload.get("response", ""),
            latency_seconds=latency,
            input_tokens=payload.get("prompt_eval_count"),
            output_tokens=payload.get("eval_count"),
            model=self.model,
            error=None,
        )
