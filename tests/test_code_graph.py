from __future__ import annotations

import json
import shutil
from pathlib import Path

from src.code_graph import build_code_graph, extract_sql_file, render_report, write_artifacts
import ultra_ingest


TMP = Path(__file__).parent / ".tmp-code-graph"


def _clean_tmp() -> Path:
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir()
    return TMP


def test_python_code_graph_extracts_symbols_imports_and_calls():
    root = _clean_tmp()
    app = root / "app.py"
    app.write_text(
        "\n".join(
            [
                "import json",
                "",
                "def helper(value):",
                "    return json.dumps(value)",
                "",
                "def main():",
                "    return helper({'ok': True})",
            ]
        ),
        encoding="utf-8",
    )

    graph = build_code_graph([root])
    data = graph.to_dict()

    labels = {node["label"] for node in data["nodes"]}
    assert "app.py" in labels
    assert "helper" in labels
    assert "main" in labels
    assert "json" in labels

    edges = {(e["relation"], e["confidence"]) for e in data["edges"]}
    assert ("imports", "EXTRACTED") in edges
    assert ("contains", "EXTRACTED") in edges
    assert ("calls", "EXTRACTED") in edges
    assert ("calls", "INFERRED") in edges


def test_sql_code_graph_extracts_tables_and_foreign_keys():
    root = _clean_tmp()
    schema = root / "schema.sql"
    schema.write_text(
        """
        CREATE TABLE users (
            id bigint primary key
        );
        CREATE TABLE orders (
            id bigint primary key,
            user_id bigint REFERENCES users(id)
        );
        """,
        encoding="utf-8",
    )

    graph = extract_sql_file(schema, root)
    data = graph.to_dict()

    labels = {node["label"] for node in data["nodes"]}
    assert "users" in labels
    assert "orders" in labels
    assert any(edge["relation"] == "references" for edge in data["edges"])


def test_code_graph_writes_portable_artifacts():
    root = _clean_tmp()
    (root / "service.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    graph = build_code_graph([root])

    stats = write_artifacts(graph, root / "out", "test")

    graph_json = Path(stats["graph_json"])
    report_md = Path(stats["report_md"])
    graph_html = Path(stats["graph_html"])
    assert graph_json.exists()
    assert report_md.exists()
    assert graph_html.exists()
    assert json.loads(graph_json.read_text(encoding="utf-8"))["nodes"]
    assert "Suggested Questions" in report_md.read_text(encoding="utf-8")
    assert "Ultra RAG Code Graph" in graph_html.read_text(encoding="utf-8")


def test_code_graph_report_counts_confidence_labels():
    graph = build_code_graph([_clean_tmp()])
    graph.add_node("a", label="A", type="function")
    graph.add_node("b", label="B", type="function")
    graph.add_edge("a", "b", "calls", "AMBIGUOUS")

    report = render_report(graph, "demo")

    assert "AMBIGUOUS: 1" in report
    assert "calls: 1" in report


def test_ultra_ingest_all_includes_code_graph_before_llm_stages():
    stages = ultra_ingest._parse_stages("all")

    assert "code_graph" in stages
    assert stages.index("code_graph") < stages.index("contextual")
    assert "code_graph" not in ultra_ingest.LLM_STAGES
