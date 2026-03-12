"""
Structure-aware hierarchical chunker.
Converts parsed blocks → final chunks with token enforcement.
"""
import re
from typing import Optional

from .config import get_config


def _count_tokens(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        return max(1, int(len(text.split()) * 1.33))


NOISE_PATTERNS = [
    re.compile(r"DISTRIBUTION\s+(?:AUTHORIZED|LIMITED|STATEMENT)", re.I),
    re.compile(r"THIS\s+PAGE\s+INTENTIONALLY\s+LEFT\s+BLANK", re.I),
    re.compile(r"FOR\s+OFFICIAL\s+USE\s+ONLY", re.I),
    re.compile(r"^\s*Page\s+\d+\s*(?:of\s+\d+)?\s*$", re.I),
]


def _is_noise(text: str) -> bool:
    return any(p.search(text) for p in NOISE_PATTERNS)


def _clean(text: str, preserve_whitespace: bool = False) -> str:
    import unicodedata
    text = text.replace("\x00", "").replace("\ufeff", "").replace("\u200b", "")
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    if not preserve_whitespace:
        text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return "\n".join(ln.rstrip() for ln in text.split("\n")).strip()


def chunk_blocks(blocks: list) -> list:
    """
    Convert blocks (from parsers.py) to final chunks.
    Each chunk: {text, content_type, heading_path, context_prefix,
                 inherited_screens, token_count}
    """
    cfg      = get_config().get("chunking", {})
    max_tok  = cfg.get("max_tokens", 600)
    min_tok  = cfg.get("min_tokens", 50)

    chunks = []

    def _mk(content: str, block: dict) -> dict:
        return {
            "text":              content.strip(),
            "content_type":      block["content_type"],
            "heading_path":      [t for _, t in block.get("heading_path", [])],
            "context_prefix":    block.get("context_prefix", ""),
            "inherited_screens": block.get("inherited_screens", []),
            "token_count":       _count_tokens(content.strip()),
            "source_section":    block.get("heading_path", [{}])[-1][1]
                                  if block.get("heading_path") else "",
        }

    def _split_text(content: str, block: dict) -> None:
        """Split oversized non-atomic block at paragraph boundaries, 1-para overlap."""
        paras = [ln for ln in content.split("\n") if ln.strip()]
        buf, buf_t = [], 0
        for para in paras:
            pt = _count_tokens(para)
            if buf_t + pt > max_tok and buf:
                ch = _mk("\n".join(buf), block)
                if ch["token_count"] >= min_tok:
                    chunks.append(ch)
                # Overlap: keep last paragraph
                overlap = [buf[-1]] if buf else []
                buf  = overlap + [para]
                buf_t = sum(_count_tokens(x) for x in buf)
            else:
                buf.append(para)
                buf_t += pt
        if buf:
            ch = _mk("\n".join(buf), block)
            if ch["token_count"] >= min_tok:
                chunks.append(ch)

    def _split_table(content: str, block: dict) -> None:
        """Split oversized table by rows."""
        lines = content.split("\n")
        if len(lines) < 3:
            chunks.append(_mk(content, block))
            return
        header  = "\n".join(lines[:2])
        header_t = _count_tokens(header)
        buf, buf_t = [header], header_t
        for row in lines[2:]:
            rt = _count_tokens(row)
            if buf_t + rt > max_tok and len(buf) > 2:
                ch = _mk("\n".join(buf), block)
                if ch["token_count"] >= min_tok:
                    chunks.append(ch)
                buf, buf_t = [header, row], header_t + rt
            else:
                buf.append(row)
                buf_t += rt
        if len(buf) > 2:
            ch = _mk("\n".join(buf), block)
            if ch["token_count"] >= min_tok:
                chunks.append(ch)

    for block in blocks:
        content     = _clean(block["content"], block.get("preserve_ws", False))
        atomic      = block.get("atomic", False)
        ctype       = block.get("content_type", "text")

        if not content.strip() or _is_noise(content):
            continue

        tok = _count_tokens(content)

        if not atomic and tok < min_tok:
            continue  # noise filter (atomic exempt)

        if atomic or tok <= max_tok:
            chunks.append(_mk(content, block))
        elif ctype == "table":
            _split_table(content, block)
        else:
            _split_text(content, block)

    return chunks
