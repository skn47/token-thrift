from __future__ import annotations

from dataclasses import dataclass

from tokenthrift.core.tokenizer import count_tokens
from tokenthrift.core.types import Policy
from tokenthrift.proxy.chunking import find_marked_blocks
from tokenthrift.proxy.generic_scorer import rank_and_score
from tokenthrift.safety.rules import apply_safety_rules


@dataclass(frozen=True)
class MessagePruneResult:
    text: str
    tokens_pruned: int


def prune_message_text(
    text: str, query: str, policy: Policy, block_id_prefix: str,
) -> MessagePruneResult:
    """Prunes only the content inside `<tokenthrift:context>` markers in
    `text`, leaving everything else byte-for-byte unchanged. Each marked
    block is scored and filtered independently through the same
    `apply_safety_rules` retention floors the trained pruner uses, using
    `generic_scorer` in place of a trained classifier."""
    blocks = find_marked_blocks(text, block_id_prefix)
    if not blocks:
        return MessagePruneResult(text=text, tokens_pruned=0)

    result_text = text
    tokens_pruned = 0
    # Process blocks back-to-front so earlier spans' offsets stay valid as
    # later (in text order) spans are spliced out first.
    for block in sorted(blocks, key=lambda b: b.start, reverse=True):
        if not block.chunks:
            replacement = ""
        else:
            ranked, scores = rank_and_score(block.chunks, query)
            retained, pruned, _budget_conflict = apply_safety_rules(ranked, scores, policy)
            kept_in_order = sorted(retained, key=lambda d: d.chunk.position)
            replacement = "\n\n".join(d.chunk.text for d in kept_in_order)
            tokens_pruned += sum(count_tokens(d.chunk.text) for d in pruned)
        result_text = result_text[: block.start] + replacement + result_text[block.end :]

    return MessagePruneResult(text=result_text, tokens_pruned=tokens_pruned)
