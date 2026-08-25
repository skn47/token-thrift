from tokenthrift.proxy.chunking import find_marked_blocks, strip_marked_blocks


def test_unmarked_text_produces_no_blocks():
    text = "Just a plain question, nothing marked here."
    assert find_marked_blocks(text, "prefix") == []


def test_a_single_marked_block_is_found_with_correct_span():
    text = "before <tokenthrift:context>some context here</tokenthrift:context> after"
    blocks = find_marked_blocks(text, "prefix")
    assert len(blocks) == 1
    block = blocks[0]
    assert text[block.start : block.end] == (
        "<tokenthrift:context>some context here</tokenthrift:context>")
    assert len(block.chunks) == 1
    assert block.chunks[0].text == "some context here"


def test_multiple_marked_blocks_each_get_their_own_chunks():
    text = (
        "<tokenthrift:context>first block</tokenthrift:context> middle "
        "<tokenthrift:context>second block</tokenthrift:context>")
    blocks = find_marked_blocks(text, "prefix")
    assert len(blocks) == 2
    assert blocks[0].chunks[0].text == "first block"
    assert blocks[1].chunks[0].text == "second block"


def test_chunk_ids_are_unique_across_blocks_via_the_prefix():
    text = (
        "<tokenthrift:context>a</tokenthrift:context> "
        "<tokenthrift:context>b</tokenthrift:context>")
    blocks = find_marked_blocks(text, "msg0")
    chunk_ids = [c.chunk_id for block in blocks for c in block.chunks]
    assert len(chunk_ids) == len(set(chunk_ids))
    assert all(cid.startswith("msg0-") for cid in chunk_ids)


def test_an_empty_marked_block_produces_no_chunks_not_a_crash():
    text = "<tokenthrift:context>   </tokenthrift:context>"
    blocks = find_marked_blocks(text, "prefix")
    assert len(blocks) == 1
    assert blocks[0].chunks == []


def test_a_marked_block_with_multiple_paragraphs_splits_into_multiple_chunks():
    text = (
        "<tokenthrift:context>\n"
        "First paragraph about topic A.\n\n"
        "Second paragraph about topic B.\n"
        "</tokenthrift:context>")
    blocks = find_marked_blocks(text, "prefix")
    assert len(blocks[0].chunks) == 2


def test_strip_marked_blocks_leaves_only_the_text_outside_the_markers():
    text = (
        "<tokenthrift:context>\n"
        "Lots of retrieved context chunks go here, at any length.\n"
        "</tokenthrift:context>\n\n"
        "Question: how do I reset my password?")
    assert strip_marked_blocks(text) == "Question: how do I reset my password?"


def test_strip_marked_blocks_is_a_noop_on_unmarked_text():
    text = "Just a plain question, nothing marked here."
    assert strip_marked_blocks(text) == text


def test_strip_marked_blocks_removes_every_marked_span_not_just_the_first():
    text = (
        "<tokenthrift:context>first</tokenthrift:context> "
        "keep this "
        "<tokenthrift:context>second</tokenthrift:context>")
    stripped = strip_marked_blocks(text)
    assert "first" not in stripped
    assert "second" not in stripped
    assert "keep this" in stripped
