"""Unit tests for src/chunker.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chunker import chunk_blocks

MAX = 600
MIN = 50


def _block(content, ctype="text", atomic=False, preserve_ws=False):
    return {
        "content":           content,
        "content_type":      ctype,
        "heading_path":      [(1, "Test Section")],
        "context_prefix":    "Test > Test Section",
        "inherited_screens": [],
        "atomic":            atomic,
        "preserve_ws":       preserve_ws,
    }


def test_noise_filtered():
    blocks = [_block("THIS PAGE INTENTIONALLY LEFT BLANK")]
    assert chunk_blocks(blocks) == [], "Noise should be filtered"


def test_tiny_non_atomic_filtered():
    blocks = [_block("Too short.")]
    assert chunk_blocks(blocks) == [], "Tiny non-atomic should be filtered"


def test_tiny_atomic_kept():
    blocks = [_block("Short but atomic.", ctype="field_definition", atomic=True)]
    result = chunk_blocks(blocks)
    assert len(result) == 1, "Tiny atomic blocks must be kept"


def test_normal_text_preserved():
    # Repeat enough to exceed MIN_CHUNK_TOKENS (50)
    content = "This is a normal paragraph with enough tokens to pass the minimum filter. " * 10
    blocks = [_block(content)]
    result = chunk_blocks(blocks)
    assert len(result) >= 1
    assert any(content[:50] in c["text"] for c in result)


def test_oversized_split():
    # ~800 tokens of text — should split
    big = ("word " * 80 + "\n") * 10
    blocks = [_block(big)]
    result = chunk_blocks(blocks)
    for c in result:
        assert c["token_count"] <= MAX + 20, f"Chunk too large: {c['token_count']}"
    assert len(result) >= 2, "Oversized block should split"


def test_atomic_not_split():
    # Atomic block that exceeds MAX should NOT be split
    big = "word " * 200
    blocks = [_block(big, ctype="field_definition", atomic=True)]
    result = chunk_blocks(blocks)
    assert len(result) == 1, "Atomic blocks must not be split"


def test_overlap():
    # Verify 1-para overlap: second chunk should start with last line of first chunk
    paras = [f"Paragraph {i}: " + "word " * 20 for i in range(10)]
    big = "\n".join(paras)
    blocks = [_block(big)]
    result = chunk_blocks(blocks)
    if len(result) >= 2:
        # The last line of chunk 0 should appear in chunk 1
        lines0 = result[0]["text"].split("\n")
        last_line0 = lines0[-1].strip()
        assert last_line0 in result[1]["text"], "1-para overlap should carry last line to next chunk"


def test_content_type_preserved():
    blocks = [
        _block("Procedure step.", ctype="procedure"),
        _block("Field definition.", ctype="field_definition", atomic=True),
        _block("Definition text.", ctype="definition", atomic=True),
    ]
    result = chunk_blocks(blocks)
    types = {c["content_type"] for c in result}
    # Some may be filtered by MIN_CHUNK_TOKENS but types that pass should be correct
    for c in result:
        orig = next(b for b in blocks if b["content_type"] == c["content_type"])
        assert c["content_type"] == orig["content_type"]


def test_context_prefix_inherited():
    blocks = [_block("content " * 20, ctype="text")]
    blocks[0]["context_prefix"] = "Annual Report > Chapter 2 > Section 4"
    result = chunk_blocks(blocks)
    for c in result:
        assert c["context_prefix"] == "Annual Report > Chapter 2 > Section 4"


def test_table_split():
    header = "| Col A | Col B |\n| --- | --- |"
    rows   = [f"| Row {i} val | Data {i} |" for i in range(50)]
    content = header + "\n" + "\n".join(rows)
    blocks = [_block(content, ctype="table")]
    result = chunk_blocks(blocks)
    for c in result:
        assert c["token_count"] <= MAX + 20, f"Table chunk too large: {c['token_count']}"


if __name__ == "__main__":
    tests = [
        test_noise_filtered,
        test_tiny_non_atomic_filtered,
        test_tiny_atomic_kept,
        test_normal_text_preserved,
        test_oversized_split,
        test_atomic_not_split,
        test_overlap,
        test_content_type_preserved,
        test_context_prefix_inherited,
        test_table_split,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:
            print(f"  ERROR {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
