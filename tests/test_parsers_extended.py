import shutil
from pathlib import Path

from src.parsers import PARSERS, parse_csv, parse_html


TMP = Path(__file__).parent / ".tmp-parsers"


def test_parser_registry_includes_launch_document_types():
    for suffix in [".csv", ".tsv", ".html", ".htm", ".xlsx", ".xlsm", ".pptx"]:
        assert suffix in PARSERS


def test_parse_csv_emits_table_block():
    shutil.rmtree(TMP, ignore_errors=True)
    TMP.mkdir(exist_ok=True)
    try:
        path = TMP / "sample.csv"
        path.write_text("name,score\nalpha,10\nbeta,9\n", encoding="utf-8")

        blocks = parse_csv(path)

        assert len(blocks) == 1
        assert blocks[0]["content_type"] == "table"
        assert "| name | score |" in blocks[0]["content"]
    finally:
        shutil.rmtree(TMP, ignore_errors=True)


def test_parse_html_strips_script_content():
    shutil.rmtree(TMP, ignore_errors=True)
    TMP.mkdir(exist_ok=True)
    try:
        path = TMP / "sample.html"
        path.write_text(
            "<html><head><script>secret()</script></head>"
            "<body><h1>Launch</h1><p>Observable RAG</p></body></html>",
            encoding="utf-8",
        )

        blocks = parse_html(path)

        assert len(blocks) == 1
        assert "Launch" in blocks[0]["content"]
        assert "Observable RAG" in blocks[0]["content"]
        assert "secret" not in blocks[0]["content"]
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
