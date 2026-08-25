from tokenthrift.pricing.costs import estimate_cost, load_pricing_table


def test_known_provider_and_model_returns_a_real_estimate():
    estimate = estimate_cost("openai", "gpt-4o-mini", input_tokens=1_000_000)
    assert estimate.known is True
    assert estimate.input_cost_usd > 0.0


def test_unpriced_openrouter_or_custom_model_never_fabricates_a_cost():
    for provider, model in [
        ("openrouter", "some/arbitrary-model"),
        ("custom", "whatever-model"),
    ]:
        estimate = estimate_cost(provider, model, input_tokens=1_000_000)
        assert estimate.known is False
        assert estimate.input_cost_usd == 0.0
        assert estimate.output_cost_usd == 0.0
        assert estimate.total_cost_usd == 0.0


def test_pricing_table_has_no_provider_field_at_the_top_level():
    # Multi-provider pricing nests under "providers" — a stale top-level
    # "provider" key (the old single-provider schema) would silently make
    # every provider's lookup wrong.
    table = load_pricing_table()
    assert "provider" not in table
    assert "providers" in table
    assert set(table["providers"]) >= {"groq", "openai", "anthropic"}
