from tokenthrift.core.types import Policy

PRESETS: dict[str, Policy] = {
    "conservative": Policy(
        preset_name="conservative", threshold=0.35, min_context=3, token_budget=3000),
    "balanced": Policy(
        preset_name="balanced", threshold=0.5, min_context=2, token_budget=2000),
    "aggressive": Policy(
        preset_name="aggressive", threshold=0.65, min_context=1, token_budget=1200),
}

DEFAULT_PRESET = "balanced"


def default_policy() -> Policy:
    return PRESETS[DEFAULT_PRESET]


def custom_policy(threshold: float, min_context: int, token_budget: int) -> Policy:
    return Policy(
        preset_name="custom",
        threshold=threshold,
        min_context=min_context,
        token_budget=token_budget,
    )
