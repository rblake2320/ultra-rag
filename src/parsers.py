"""
Parsers: DOCX (structure-aware), PDF, TXT/MD.
Returns list of blocks for the chunker.
"""
import re
from pathlib import Path
from typing import Iterator


# ── DOCX Style Maps ──────────────────────────────────────────────────────
HEADING_STYLES = {
    "Heading 1", "Heading 2", "Heading 3", "Heading 4", "Heading 5",
    "Heading 6", "Heading 7", "Heading0",
    "User G 1", "User G 2", "User G 3",
    "2 TITLE", "Title1", "Title2", "Title3", "Title4", "Title10", "Title11",
    "A1Title", "TITLE Char Char", "TOC Heading",
    "Style Title + Before:  6 pt After:  6 pt1",
    'Style Title + Left:  0"',
}
HEADING_LEVEL_MAP = {
    "Heading 1": 1, "Heading 2": 2, "Heading 3": 3, "Heading 4": 4,
    "Heading 5": 5, "Heading 6": 6, "Heading 7": 7, "Heading0": 1,
    "User G 1": 2, "User G 2": 3, "User G 3": 4,
    "2 TITLE": 2,
    "Title1": 4, "Title2": 5, "Title3": 6, "Title4": 7,
    "Title10": 4, "Title11": 4,
    "A1Title": 3, "TITLE Char Char": 3, "TOC Heading": 1,
    "Style Title + Before:  6 pt After:  6 pt1": 3,
    'Style Title + Left:  0"': 3,
}
BOX_TITLE_STYLES      = {"BOX TITLE"}
BOX_TEXT_STYLES       = {"BOX TEXT", "boxtext", "Shadow Box"}
DEFINITION_TITLE_STYLES = {"Definitions Title"}
DEFINITION_BODY_STYLES  = {"Definition Body"}
PROCEDURE_STYLES      = {"Out1", "Out1.1", "Out2", "Out3", "Out5"}
SCREEN_STYLES         = {"Screen", "Screen Print", "Key Pre", "HTML Preformatted"}
SKIP_STYLES           = {
    "toc 1", "toc 2", "toc 3", "toc 4", "TOC", "TOC Heading",
    "Header", "Footer", "Balloon Text", "No Spacing",
    "Inside Address", "annotation text",
}

NOISE_PATTERNS = [
    re.compile(r"DISTRIBUTION\s+(?:AUTHORIZED|LIMITED|STATEMENT)", re.I),
    re.compile(r"THIS\s+PAGE\s+INTENTIONALLY\s+LEFT\s+BLANK", re.I),
    re.compile(r"FOR\s+OFFICIAL\s+USE\s+ONLY", re.I),
    re.compile(r"^\s*Page\s+\d+\s*(?:of\s+\d+)?\s*$", re.I),
]

SCREEN_PATTERNS = [
    re.compile(r"\bScreen\s+(\d{3,4})\b", re.I),
    re.compile(r"\bSCN\s*(\d{3,4})\b", re.I),
    re.compile(r"\b(\d{3})\s+Screen\b", re.I),
]


def is_noise(text: str) -> bool:
    return any(p.search(text) for p in NOISE_PATTERNS)


def extract_screens_from_text(text: str) -> list:
    screens = set()
    for pat in SCREEN_PATTERNS:
        for m in pat.finditer(text):
            screens.add(m.group(1))
    return sorted(screens)


def heading_level_from_style(style_name: str) -> int:
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
    Structure-aware DOCX parser. Returns list of blocks:
        {content, content_type, heading_path [(level, text),...],
         context_prefix, inherited_screens, atomic, preserve_ws}
    """
    from docx import Document
    doc = Document(str(path))

    blocks = []
    heading_stack = []  # [(level, text, [screens]), ...]

    cur_lines = []
    box_title, box_texts = None, []
    def_title, def_body  = None, []
    proc_lines = []
    in_proc = False

    def get_inherited_screens():
        s = set()
        for _, _, scrs in heading_stack:
            s.update(scrs)
        return sorted(s)

    def get_context_prefix():
        parts = [path.stem]
        for _, txt, _ in heading_stack:
            parts.append(txt)
        return " > ".join(parts)

    def get_heading_path():
        return [(lvl, txt) for lvl, txt, _ in heading_stack]

    def emit(content, ctype, atomic=False, preserve_ws=False):
        if not content.strip() or is_noise(content):
            return
        blocks.append({
            "content":           content,
            "content_type":      ctype,
            "heading_path":      get_heading_path(),
            "context_prefix":    get_context_prefix(),
            "inherited_screens": get_inherited_screens(),
            "atomic":            atomic,
            "preserve_ws":       preserve_ws,
        })

    def flush_box():
        nonlocal box_title, box_texts
        if box_title is not None:
            content = box_title + ("\n" + "\n".join(box_texts) if box_texts else "")
            emit(content, "field_definition", atomic=True)
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
        proc_lines, in_proc = [], False

    def flush_cur():
        nonlocal cur_lines
        if cur_lines:
            emit("\n".join(cur_lines), "text")
        cur_lines = []

    def update_heading(level, text):
        nonlocal heading_stack
        heading_stack = [(l, t, s) for l, t, s in heading_stack if l < level]
        heading_stack.append((level, text, extract_screens_from_text(text)))

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
            raw = elem.text  # preserve whitespace
            if raw.strip() and not is_noise(raw):
                blocks.append({
                    "content":           raw,
                    "content_type":      "screen_print",
                    "heading_path":      get_heading_path(),
                    "context_prefix":    get_context_prefix(),
                    "inherited_screens": get_inherited_screens(),
                    "atomic":            True,
                    "preserve_ws":       True,
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
    """Page-based PDF parser. Returns one block per page."""
    import fitz
    doc = fitz.open(str(path))
    blocks = []
    for pn, page in enumerate(doc, 1):
        text = page.get_text("text").strip()
        if text and not is_noise(text):
            blocks.append({
                "content":           text,
                "content_type":      "text",
                "heading_path":      [(1, f"Page {pn}")],
                "context_prefix":    f"{path.stem} > Page {pn}",
                "inherited_screens": extract_screens_from_text(text),
                "atomic":            False,
                "preserve_ws":       False,
            })
        try:
            for t in page.find_tables():
                data = t.extract()
                if data and len(data) > 1:
                    hdrs = " | ".join(str(c) for c in data[0])
                    rows = "\n".join(" | ".join(str(c) for c in r) for r in data[1:])
                    md = hdrs + "\n" + "| --- " * len(data[0]) + "|\n" + rows
                    if md.strip():
                        blocks.append({
                            "content":           md,
                            "content_type":      "table",
                            "heading_path":      [(1, f"Page {pn}")],
                            "context_prefix":    f"{path.stem} > Page {pn}",
                            "inherited_screens": [],
                            "atomic":            False,
                            "preserve_ws":       False,
                        })
        except Exception:
            pass
    doc.close()
    return blocks


def parse_txt(path: Path) -> list:
    """Markdown/plaintext parser using # headers as section boundaries."""
    try:
        import chardet
        raw = path.read_bytes()
        enc = chardet.detect(raw).get("encoding", "utf-8") or "utf-8"
        text = raw.decode(enc, errors="replace")
    except ImportError:
        text = path.read_text(encoding="utf-8", errors="replace")

    blocks = []
    cur_heading = [path.stem]
    cur_lines   = []

    def flush():
        if cur_lines:
            content = "\n".join(cur_lines)
            if content.strip() and not is_noise(content):
                blocks.append({
                    "content":           content,
                    "content_type":      "text",
                    "heading_path":      [(i+1, h) for i, h in enumerate(cur_heading)],
                    "context_prefix":    " > ".join(cur_heading),
                    "inherited_screens": extract_screens_from_text(content),
                    "atomic":            False,
                    "preserve_ws":       False,
                })
        cur_lines.clear()

    for line in text.split("\n"):
        m = re.match(r"^(#{1,6})\s+(.+)$", line)
        if m:
            flush()
            level = len(m.group(1))
            heading = m.group(2).strip()
            # Keep parent headings at lower levels
            cur_heading = [h for i, h in enumerate(cur_heading) if i < level - 1]
            cur_heading.append(heading)
        else:
            cur_lines.append(line)
    flush()
    return blocks


PARSERS = {
    ".docx": parse_docx,
    ".doc":  parse_docx,
    ".pdf":  parse_pdf,
    ".txt":  parse_txt,
    ".md":   parse_txt,
}
