from tokenthrift.core.types import Chunk
from tokenthrift.generation.prompt import build_marked_prompt, build_prompt


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        doc_id="doc", chunk_id=chunk_id, text=text, source_type="prose",
        doc_title="Doc", heading=None, position=0, doc_chunk_count=1,
    )


def test_context_block_is_wrapped_in_tokenthrift_markers():
    chunks = [_chunk("c1", "Some retrieved context.")]
    prompt = build_marked_prompt("a question", chunks)
    assert "<tokenthrift:context>" in prompt
    assert "</tokenthrift:context>" in prompt
    assert "Some retrieved context." in prompt


def test_question_and_system_instruction_stay_outside_the_markers():
    chunks = [_chunk("c1", "Some retrieved context.")]
    prompt = build_marked_prompt("how do I reset my password", chunks)
    open_marker = prompt.index("<tokenthrift:context>")
    close_marker = prompt.index("</tokenthrift:context>")
    assert prompt.index("Question: how do I reset my password") > close_marker
    assert prompt.index("You are a helpful assistant") < open_marker


def test_matches_build_prompt_content_modulo_the_markers():
    chunks = [_chunk("c1", "Some retrieved context.")]
    plain = build_prompt("q", chunks)
    marked = build_marked_prompt("q", chunks)
    assert marked.replace("<tokenthrift:context>\n", "").replace(
        "\n</tokenthrift:context>", "") == plain


def test_empty_context_still_produces_a_valid_marked_block():
    prompt = build_marked_prompt("q", [])
    assert "<tokenthrift:context>" in prompt
    assert "(no context retrieved)" in prompt
