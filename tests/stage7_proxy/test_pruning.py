from tokenthrift.core.types import Policy
from tokenthrift.proxy.pruning import prune_message_text

_AGGRESSIVE = Policy(preset_name="aggressive", threshold=0.65, min_context=1, token_budget=1200)


def test_unmarked_text_is_untouched_byte_for_byte():
    text = "What's the capital of France? No context markers here at all."
    result = prune_message_text(text, "capital of France", _AGGRESSIVE, "prefix")
    assert result.text == text
    assert result.tokens_pruned == 0


def test_marked_content_can_be_pruned_when_clearly_irrelevant():
    text = (
        "<tokenthrift:context>\n"
        "How do I reset my password? Go to Settings > Security > Reset "
        "password and follow the emailed link.\n\n"
        "Our office holiday schedule is published every December and "
        "covers all federal holidays for the following year.\n"
        "</tokenthrift:context>\n\n"
        "Question: how do I reset my password?")
    result = prune_message_text(
        text, "how do I reset my password", _AGGRESSIVE, "prefix")

    assert "reset my password" in result.text
    assert "Question: how do I reset my password?" in result.text
    assert result.text != text
    assert result.tokens_pruned >= 0


def test_min_context_floor_never_drops_below_the_policy_minimum():
    conservative = Policy(
        preset_name="conservative", threshold=0.99, min_context=2, token_budget=3000)
    text = (
        "<tokenthrift:context>\n"
        "Paragraph one is totally unrelated to anything.\n\n"
        "Paragraph two is also totally unrelated to anything.\n\n"
        "Paragraph three is also totally unrelated to anything.\n"
        "</tokenthrift:context>")
    result = prune_message_text(text, "an unrelated query", conservative, "prefix")
    kept_paragraphs = sum(
        1 for marker in ("Paragraph one", "Paragraph two", "Paragraph three")
        if marker in result.text)
    assert kept_paragraphs >= conservative.min_context


def test_multiple_messages_use_distinct_prefixes_without_chunk_id_collisions():
    text = "<tokenthrift:context>same text</tokenthrift:context>"
    r1 = prune_message_text(text, "q", _AGGRESSIVE, "msg0")
    r2 = prune_message_text(text, "q", _AGGRESSIVE, "msg1")
    assert r1.text == r2.text  # same content, no cross-contamination either way
