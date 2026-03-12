"""
Parsers: DOCX (structure-aware), PDF, TXT/MD.
Returns list of blocks for the chunker.

Each block is a dict:
    {
        "content":      str,
        "content_type": str,           # text | table | definition | procedure | screen_print
        "heading_path": [(level, text), ...],
        "context_prefix": str,         # breadcrumb for contextual retrieval
        "atomic":       bool,          # do not split this block
        "preserve_ws":  bool,          # preserve whitespace (preformatted)
    }
"""
import re
from pathlib import Path
from typing import Iterator


# ── DOCX Heading Style Maps ──────────────────────────────────────────────────
# Standard Word heading styles only — no domain-specific custom styles.
HEADING_STYLES = {
    "Heading 1", "Heading 2", "Heading 3", "Heading 4",
    "Heading 5", "Heading 6", "Heading 7",
    "Title", "Subtitle",
}

HEADING_LEVEL_MAP = {
    "Heading 1": 1, "Heading 2": 2, "Heading 3": 3, "Heading 4": 4,
    "Heading 5": 5, "Heading 6": 6, "Heading 7": 7,
    "Title":     1, "Subtitle": 2,
}

# Definition-like box styles (standard Word styles)
BOX_TITLE_STYLES    = {"Quote", "Intense Quote"}
BOX_TEXT_STYLES     = {"Body Text", "Body Text 2", "Body Text 3"}
DEFINITION_TITLE_STYLES = set()   # extend with your custom styles
DEFINITION_BODY_STYLES  = set()

# Procedure / list styles
PROCEDURE_STYLES = {"List Number", "List Number 2", "List Number 3",
                    "List Bullet", "List Bullet 2", "List Bullet 3"}

# Preformatted / code styles
SCREEN_STYLES = {"HTML Preformatted", "Preformatted Text", "Code"}

# Styles to skip entirely
SKIP_STYLES = {
    "toc 1", "toc 2", "toc 3", "toc 4", "TOC Heading",
    "Header", "Footer", "Balloon Text", "annotation text",
    "No Spacing", "Inside Address",
}

# Noise text patterns — apply to all parsers
NOISE_PATTERNS = [
    re.compile(r"THIS\s+PAGE\s+INTENTIONALLY\s+LEFT\s+BLANK", re.I),
    re.compile(r"^\s*Page\s+\d+\s*(?:of\s+\d+)?\s*$", re.I),
    re.compile(r"^\s*\d+\s*$"),            # bare page numbers
    re.compile(r"^[-=_]{10,}$"),            # decorative rule lines
]


def is_noise(text: str) -> bool:
    return any(p.search(text) for p in NOISE_PATTERNS)


def heading_level_from_style(style_name: str) -> int:
    """Return 1-7 heading level for a Word style name."""
    if style_name in HEADING_LEVEL_MAP:
        return HEADING_LEVEL_MAP[style_name]
    m = re.match(r"Heading\s*(\d+)", style_name, re.I)
    return int(m.group(1)) if m else 1


def _iter_docx_body(doc) -> Iterator:
    """Yield (type, obj) for paragraphs and tables in document order."""
    from docx.text.paragraph import Paragraph as DocxParagraph
    from docx.table import Table as DocxTable
    for child in doc.element.body:
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if tag == "p":
            yield "para", DocxParagraph(child, doc)
        elif tag == "tbl":
            yield "table", DocxTable(child, doc)


def _dedup_row_cells(row) -> list:
    seen_ids, result = set(), []
    for cell in row.cells:
        cid = id(cell._tc)
        if cid not in seen_ids:
            seen_ids.add(cid)
            result.append(cell.text.strip())
    return result


def _table_to_markdown(table) -> str:
    rows = [_dedup_row_cells(r) for r in table.rows]
    if not rows:
        return ""
    max_cols = max(len(r) for r in rows)
    rows = [r + [""] * (max_cols - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |",
             "| " + " | ".join(["---"] * max_cols) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def _get_style_name(para) -> str:
    return para.style.name if para.style and para.style.name else "Normal"


def parse_docx(path: Path) -> list:
    """
    Structure-aware DOCX parser.

    Tracks heading hierarchy to build context breadcrumbs and outputs
    typed blocks: text, table, definition, procedure, screen_print.
    """
    from docx import Document
    doc = Document(str(path))

    blocks = []
    heading_stack = []   # [(level, text), ...]

    cur_lines   = []
    box_title   = None
    box_texts   = []
    def_title   = None
    def_body    = []
    proc_lines  = []
    in_proc     = False

    def get_context_prefix():
        parts = [path.stem]
        parts.extend(txt for _, txt in heading_stack)
        return " > ".join(parts)

    def get_heading_path():
        return list(heading_stack)

    def emit(content, ctype, atomic=False, preserve_ws=False):
        if not content.strip() or is_noise(content):
            return
        blocks.append({
            "content":        content,
            "content_type":   ctype,
            "heading_path":   get_heading_path(),
            "context_prefix": get_context_prefix(),
            "atomic":         atomic,
            "preserve_ws":    preserve_ws,
        })

    def flush_box():
        nonlocal box_title, box_texts
        if box_title is not None:
            content = box_title + ("\n" + "\n".join(box_texts) if box_texts else "")
            emit(content, "definition", atomic=True)
        box_title, box_texts = None, []

    def flush_def():
        nonlocal def_title, def_body
        if def_title is not None and def_body:
            emit(def_title + "\n" + "\n".join(def_body), "definition", atomic=True)
        def_title, def_body = None, []

    def flush_proc():
        nonlocal proc_lines, in_proc
        if proc_lines:
            emit("\n".join(proc_lines), "procedure")
        proc_lines.clear()
        in_proc = False

    def flush_cur():
        nonlocal cur_lines
        if cur_lines:
            emit("\n".join(cur_lines), "text")
        cur_lines.clear()

    def update_heading(level, text):
        nonlocal heading_stack
        heading_stack = [(l, t) for l, t in heading_stack if l < level]
        heading_stack.append((level, text))

    for elem_type, elem in _iter_docx_body(doc):
        if elem_type == "table":
            flush_cur(); flush_proc(); flush_box(); flush_def()
            md = _table_to_markdown(elem)
            if md.strip():
                emit(md, "table")
            continue

        style = _get_style_name(elem)
        text  = elem.text.strip()
        if not text or style in SKIP_STYLES:
            continue

        if style in HEADING_STYLES:
            flush_cur(); flush_proc(); flush_box(); flush_def()
            update_heading(heading_level_from_style(style), text)

        elif style in BOX_TITLE_STYLES:
            flush_cur(); flush_proc(); flush_box(); flush_def()
            box_title, box_texts = text, []

        elif style in BOX_TEXT_STYLES:
            if box_title is not None:
                box_texts.append(text)
            else:
                cur_lines.append(text)

        elif style in DEFINITION_TITLE_STYLES:
            flush_cur(); flush_proc(); flush_box(); flush_def()
            def_title, def_body = text, []

        elif style in DEFINITION_BODY_STYLES:
            if def_title is not None:
                def_body.append(text)
            else:
                cur_lines.append(text)

        elif style in PROCEDURE_STYLES:
            flush_cur(); flush_box(); flush_def()
            proc_lines.append(text)
            in_proc = True

        elif style in SCREEN_STYLES:
            flush_cur(); flush_proc(); flush_box(); flush_def()
            raw = elem.text   # preserve whitespace for preformatted content
            if raw.strip() and not is_noise(raw):
                blocks.append({
                    "content":        raw,
                    "content_type":   "screen_print",
                    "heading_path":   get_heading_path(),
                    "context_prefix": get_context_prefix(),
                    "atomic":         True,
                    "preserve_ws":    True,
                })

        else:
            if in_proc:
                flush_proc()
            if box_title is not None:
                flush_box()
            if def_title is not None and not def_body:
                cur_lines.append(def_title)
                def_title = None
            cur_lines.append(text)

    flush_cur(); flush_proc(); flush_box(); flush_def()
    return blocks


def parse_pdf(path: Path) -> list:
    """
    Page-based PDF parser using PyMuPDF (fitz).

    Returns one text block per page plus any detected tables.
    Falls back to pypdf if fitz is not installed.
    """
    blocks = []

    try:
        import fitz
        doc = fitz.open(str(path))
        for pn, page in enumerate(doc, 1):
            text = page.get_text("text").strip()
            if text and not is_noise(text):
                blocks.append({
                    "content":        text,
                    "content_type":   "text",
                    "heading_path":   [(1, f"Page {pn}")],
                    "context_prefix": f"{path.stem} > Page {pn}",
                    "atomic":         False,
                    "preserve_ws":    False,
                })
            # Extract tables if available
            try:
                for t in page.find_tables():
                    data = t.extract()
                    if data and len(data) > 1:
                        hdrs = " | ".join(str(c) for c in data[0])
                        sep  = "| --- " * len(data[0]) + "|"
                        rows = "\n".join(" | ".join(str(c) for c in r) for r in data[1:])
                        md   = hdrs + "\n" + sep + "\n" + rows
                        if md.strip():
                            blocks.append({
                                "content":        md,
                                "content_type":   "table",
                                "heading_path":   [(1, f"Page {pn}")],
                                "context_prefix": f"{path.stem} > Page {pn}",
                                "atomic":         False,
                                "preserve_ws":    False,
                            })
            except Exception:
                pass
        doc.close()

    except ImportError:
        # Fallback to pypdf
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            for pn, page in enumerate(reader.pages, 1):
                text = (page.extract_text() or "").strip()
                if text and not is_noise(text):
                    blocks.append({
                        "content":        text,
                        "content_type":   "text",
                        "heading_path":   [(1, f"Page {pn}")],
                        "context_prefix": f"{path.stem} > Page {pn}",
                        "atomic":         False,
                        "preserve_ws":    False,
                    })
        except Exception as exc:
            blocks.append({
                "content":        f"[PDF parse error: {exc}]",
                "content_type":   "text",
                "heading_path":   [],
                "context_prefix": path.stem,
                "atomic":         False,
                "preserve_ws":    False,
            })

    return blocks


def parse_txt(path: Path) -> list:
    """
    Markdown / plaintext parser.

    Uses # headings as section boundaries. Non-heading lines are
    accumulated into text blocks under the current heading context.
    """
    try:
        import chardet
        raw = path.read_bytes()
        enc = chardet.detect(raw).get("encoding", "utf-8") or "utf-8"
        text = raw.decode(enc, errors="replace")
    except ImportError:
        text = path.read_text(encoding="utf-8", errors="replace")

    blocks     = []
    cur_heading = [path.stem]   # breadcrumb stack
    cur_lines   = []

    def flush():
        if cur_lines:
            content = "\n".join(cur_lines).strip()
            if content and not is_noise(content):
                blocks.append({
                    "content":        content,
                    "content_type":   "text",
                    "heading_path":   [(i + 1, h) for i, h in enumerate(cur_heading)],
                    "context_prefix": " > ".join(cur_heading),
                    "atomic":         False,
                    "preserve_ws":    False,
                })
        cur_lines.clear()

    for line in text.split("\n"):
        m = re.match(r"^(#{1,6})\s+(.+)$", line)
        if m:
            flush()
            level   = len(m.group(1))
            heading = m.group(2).strip()
            # Trim heading stack to current level
            cur_heading = [h for i, h in enumerate(cur_heading) if i < level - 1]
            cur_heading.append(heading)
        else:
            cur_lines.append(line)
    flush()
    return blocks


# ── Public registry ──────────────────────────────────────────────────────────
PARSERS: dict = {
    ".docx": parse_docx,
    ".doc":  parse_docx,
    ".pdf":  parse_pdf,
    ".txt":  parse_txt,
    ".md":   parse_txt,
    ".rst":  parse_txt,
}
