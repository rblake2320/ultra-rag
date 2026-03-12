"""
Structure-aware hierarchical chunker — Ultra RAG version.

Improvements over v1:
- Content-type-aware token limits (field_definition: 400, table: 1200, text: 1000)
- Sentence-boundary splitting: never cut mid-sentence
- 150-token sliding overlap between adjacent text chunks
- Tiny chunk merging: orphaned fragments (< 100 tok) merged with neighbors
- Better paragraph grouping: groups short paragraphs before splitting
"""
import re
import unicodedata
from typing import Optional

from .config import get_config


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

def _count_tokens(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        return max(1, int(len(text.split()) * 1.33))


# ---------------------------------------------------------------------------
# Noise detection
# ---------------------------------------------------------------------------

NOISE_PATTERNS = [
    re.compile(r"DISTRIBUTION\s+(?:AUTHORIZED|LIMITED|STATEMENT)", re.I),
    re.compile(r"THIS\s+PAGE\s+INTENTIONALLY\s+LEFT\s+BLANK", re.I),
    re.compile(r"FOR\s+OFFICIAL\s+USE\s+ONLY", re.I),
    re.compile(r"^\s*Page\s+\d+\s*(?:of\s+\d+)?\s*$", re.I),
]


def _is_noise(text: str) -> bool:
    return any(p.search(text) for p in NOISE_PATTERNS)


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def _clean(text: str, preserve_whitespace: bool = False) -> str:
    text = text.replace("\x00", "").replace("\ufeff", "").replace("\u200b", "")
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    if not preserve_whitespace:
        text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return "\n".join(ln.rstrip() for ln in text.split("\n")).strip()


# ---------------------------------------------------------------------------
# Sentence-boundary splitting
# ---------------------------------------------------------------------------

def _split_at_sentence_boundary(text: str, max_pos: int) -> int:
    """
    Find the best split position at or before max_pos (character index).

    Priority:
    1. End of sentence: '. ' or '.\\n'
    2. Paragraph boundary: '\\n\\n'
    3. Line boundary: '\\n'
    4. Last space before max_pos

    Does not look back more than 20% of max_pos to avoid tiny leading chunks.
    Returns the character index at which to cut (exclusive — the slice end).
    """
    # Clamp to valid range
    max_pos = min(max_pos, len(text))
    if max_pos <= 0:
        return 0

    # How far back we're willing to search (20% of target position, min 50 chars)
    lookback = max(50, int(max_pos * 0.20))
    search_start = max(0, max_pos - lookback)
    window = text[search_start:max_pos]

    # 1. Sentence end: '. ' or '.\n'
    for pattern in (". ", ".\n"):
        idx = window.rfind(pattern)
        if idx != -1:
            # position is after the period and the trailing char
            return search_start + idx + len(pattern)

    # 2. Paragraph boundary
    idx = window.rfind("\n\n")
    if idx != -1:
        return search_start + idx + 2

    # 3. Line boundary
    idx = window.rfind("\n")
    if idx != -1:
        return search_start + idx + 1

    # 4. Last space
    idx = window.rfind(" ")
    if idx != -1:
        return search_start + idx + 1

    # No boundary found — hard cut at max_pos
    return max_pos


# ---------------------------------------------------------------------------
# Overlap text splitter
# ---------------------------------------------------------------------------

def _split_text_with_overlap(content: str, block: dict, max_tok: int,
                              overlap_tok: int, min_tok: int) -> list:
    """
    Split a large text block into chunks with sentence-boundary awareness
    and sliding overlap between consecutive chunks.

    The START of chunk[n+1] overlaps with the END of chunk[n] by ~overlap_tok tokens.
    """
    total_chars = len(content)
    if total_chars == 0:
        return []

    total_tok = _count_tokens(content)
    if total_tok == 0:
        return []

    # Estimate chars-per-token for this specific content
    chars_per_tok = total_chars / total_tok

    target_chars = int(max_tok * chars_per_tok)
    overlap_chars = int(overlap_tok * chars_per_tok)

    result = []
    start = 0

    while start < total_chars:
        # Desired end of this chunk
        tentative_end = start + target_chars

        if tentative_end >= total_chars:
            # Last segment: take everything remaining
            chunk_text = content[start:].strip()
            if chunk_text:
                c = _mk(chunk_text, block)
                if c["token_count"] >= min_tok:
                    result.append(c)
            break

        # Find a clean sentence boundary near tentative_end
        split_pos = _split_at_sentence_boundary(content, tentative_end)

        # Safety: never produce a zero-length chunk
        if split_pos <= start:
            split_pos = tentative_end  # hard cut

        chunk_text = content[start:split_pos].strip()
        if chunk_text:
            c = _mk(chunk_text, block)
            if c["token_count"] >= min_tok:
                result.append(c)

        # Advance start, backing up by overlap_chars for continuity
        next_start = split_pos - overlap_chars
        if next_start <= start:
            # Avoid infinite loop: always advance by at least 1 char
            next_start = split_pos
        start = next_start

    return result


# ---------------------------------------------------------------------------
# Table splitter
# ---------------------------------------------------------------------------

def _split_table(content: str, block: dict, max_tok: int, min_tok: int) -> list:
    """
    Split an oversized table block by rows, repeating the header on each
    sub-chunk.  If the whole table fits in max_tok, return it as one chunk.
    """
    total_tok = _count_tokens(content)
    if total_tok <= max_tok:
        return [_mk(content, block)]

    lines = content.split("\n")
    if len(lines) < 3:
        # Tiny table or malformed — keep as-is
        return [_mk(content, block)]

    header = "\n".join(lines[:2])
    header_t = _count_tokens(header)
    result = []
    buf = [header]
    buf_t = header_t

    for row in lines[2:]:
        rt = _count_tokens(row)
        if buf_t + rt > max_tok and len(buf) > 2:
            c = _mk("\n".join(buf), block)
            if c["token_count"] >= min_tok:
                result.append(c)
            buf = [header, row]
            buf_t = header_t + rt
        else:
            buf.append(row)
            buf_t += rt

    if len(buf) > 2:
        c = _mk("\n".join(buf), block)
        if c["token_count"] >= min_tok:
            result.append(c)

    return result


# ---------------------------------------------------------------------------
# Tiny-chunk merging pass
# ---------------------------------------------------------------------------

def _merge_tiny_chunks(chunks: list, merge_threshold: int,
                       type_limits: dict, default_max: int) -> list:
    """
    Post-processing: merge any chunk below merge_threshold tokens into an
    adjacent neighbor when:
      - Same content_type OR neighbor is text/body
      - Same heading_path (first 2 levels match)
      - Combined token count <= type-specific max_tok

    Iterates until no further merges can be made (handles cascades).
    """
    if not chunks:
        return chunks

    def _heading_prefix(path: list, levels: int = 2) -> tuple:
        return tuple(path[:levels])

    def _max_for_type(ctype: str) -> int:
        return type_limits.get(ctype, default_max)

    def _can_merge(a: dict, b: dict) -> bool:
        """Return True if a and b are eligible to merge."""
        # Content-type compatibility
        a_type = a["content_type"]
        b_type = b["content_type"]
        type_ok = (
            a_type == b_type
            or a_type in ("text", "body")
            or b_type in ("text", "body")
        )
        if not type_ok:
            return False

        # Heading proximity (first 2 levels)
        if _heading_prefix(a["heading_path"]) != _heading_prefix(b["heading_path"]):
            return False

        # Combined size must fit in the larger of the two type limits
        limit = max(_max_for_type(a_type), _max_for_type(b_type))
        return (a["token_count"] + b["token_count"]) <= limit

    def _do_merge(a: dict, b: dict) -> dict:
        """Merge b into a, returning a new combined chunk dict."""
        combined_text = a["text"].rstrip() + "\n" + b["text"].lstrip()
        # Use the type/path of whichever is larger
        base = a if a["token_count"] >= b["token_count"] else b
        return {
            "text":              combined_text.strip(),
            "content_type":      base["content_type"],
            "heading_path":      base["heading_path"],
            "context_prefix":    base["context_prefix"],
            "inherited_screens": base["inherited_screens"],
            "token_count":       _count_tokens(combined_text.strip()),
            "source_section":    base["source_section"],
        }

    changed = True
    while changed:
        changed = False
        merged: list = []
        i = 0
        while i < len(chunks):
            chunk = chunks[i]
            if chunk["token_count"] < merge_threshold:
                # Try merge with previous
                if merged and _can_merge(merged[-1], chunk):
                    merged[-1] = _do_merge(merged[-1], chunk)
                    changed = True
                    i += 1
                    continue
                # Try merge with next
                if i + 1 < len(chunks) and _can_merge(chunk, chunks[i + 1]):
                    merged.append(_do_merge(chunk, chunks[i + 1]))
                    changed = True
                    i += 2
                    continue
            # No merge possible (or chunk is large enough)
            merged.append(chunk)
            i += 1
        chunks = merged

    return chunks


# ---------------------------------------------------------------------------
# Chunk dict factory
# ---------------------------------------------------------------------------

def _mk(content: str, block: dict) -> dict:
    """Create a chunk dict from cleaned content + source block metadata."""
    text = content.strip()
    heading_path = block.get("heading_path", [])
    return {
        "text":              text,
        "content_type":      block["content_type"],
        "heading_path":      [t for _, t in heading_path],
        "context_prefix":    block.get("context_prefix", ""),
        "inherited_screens": block.get("inherited_screens", []),
        "token_count":       _count_tokens(text),
        "source_section":    heading_path[-1][1] if heading_path else "",
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

DEFAULT_TYPE_LIMITS = {
    "field_definition": 400,
    "table":            1200,
    "procedure":        1000,
    "screen_print":     800,
    "text":             1000,
    "body":             1000,
    "box":              900,
    "list":             800,
}


def chunk_blocks(blocks: list) -> list:
    """
    Convert blocks (from parsers.py) to final chunks.

    Each chunk dict contains:
        text, content_type, heading_path, context_prefix,
        inherited_screens, token_count, source_section
    """
    cfg             = get_config().get("chunking", {})
    max_tok         = cfg.get("max_tokens", 1000)
    min_tok         = cfg.get("min_tokens", 50)
    overlap_tok     = cfg.get("overlap_tokens", 150)
    merge_threshold = cfg.get("merge_threshold", 100)
    type_limits     = {**DEFAULT_TYPE_LIMITS, **cfg.get("type_limits", {})}

    chunks: list = []

    for block in blocks:
        content = _clean(block["content"], block.get("preserve_ws", False))
        atomic  = block.get("atomic", False)
        ctype   = block.get("content_type", "text")

        if not content.strip() or _is_noise(content):
            continue

        tok = _count_tokens(content)

        if not atomic and tok < min_tok:
            continue  # too small and not protected

        # Per-type ceiling (atomic blocks bypass the ceiling)
        type_max = type_limits.get(ctype, max_tok)

        if atomic or tok <= type_max:
            # Fits as a single chunk (or must not be split)
            chunks.append(_mk(content, block))

        elif ctype == "table":
            chunks.extend(_split_table(content, block, type_max, min_tok))

        else:
            # General text/procedure/screen_print/etc.
            chunks.extend(
                _split_text_with_overlap(
                    content, block, type_max, overlap_tok, min_tok
                )
            )

    # Post-processing: merge orphaned tiny fragments
    chunks = _merge_tiny_chunks(chunks, merge_threshold, type_limits, max_tok)

    # Recompute token counts after merging (merge already does this, but
    # ensure consistency for any chunk that passed through unchanged)
    for c in chunks:
        if "token_count" not in c:
            c["token_count"] = _count_tokens(c["text"])

    return chunks
