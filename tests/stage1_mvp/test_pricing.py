import pytest

from tokenthrift.pricing.costs import estimate_cost, load_pricing_table


def test_cost_estimate_uses_configured_price_and_reports_pricing_version():
    table = load_pricing_table()
    estimate = estimate_cost(
        "groq", "openai/gpt-oss-20b", input_tokens=1_000_000,
        pricing_table=table)

    assert estimate.known is True
    assert estimate.input_cost_usd == pytest.approx(
        table["providers"]["groq"]["models"]["openai/gpt-oss-20b"]["input_per_million"])
    assert estimate.pricing_version == table["pricing_version"]


def test_unknown_model_is_reported_as_unknown_not_a_fabricated_zero_cost():
    estimate = estimate_cost("groq", "not-a-real-model", input_tokens=100)
    assert estimate.known is False
    assert estimate.input_cost_usd == 0.0
    assert estimate.output_cost_usd == 0.0


def test_unknown_provider_is_reported_as_unknown_not_a_fabricated_zero_cost():
    estimate = estimate_cost("openrouter", "some/model", input_tokens=100)
    assert estimate.known is False
    assert estimate.input_cost_usd == 0.0
    assert estimate.output_cost_usd == 0.0
