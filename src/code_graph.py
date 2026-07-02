"""
Deterministic code graph extraction and export.

This module adds a lightweight, non-LLM graph layer for source code. It is not a
replacement for the LLM KG extractor; it complements it with auditable edges
that come directly from syntax and simple schema patterns.
"""
from __future__ import annotations

import ast
import html
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .config import get_config


CODE_EXTENSIONS = {".py", ".sql"}


def _normalize_id(*parts: str) -> str:
    text = "_".join(p for p in parts if p)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[^\w]+", "_", text, flags=re.UNICODE)
    text = re.sub(r"_+", "_", text)
    return text.strip("_").casefold() or "node"


def _rel_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


@dataclass
class CodeGraph:
    nodes: dict[str, dict] = field(default_factory=dict)
    edges: list[dict] = field(default_factory=list)

    def add_node(self, node_id: str, **attrs) -> None:
        existing = self.nodes.get(node_id, {})
        merged = {**existing, **{k: v for k, v in attrs.items() if v is not None}}
        merged["id"] = node_id
        self.nodes[node_id] = merged

    def add_edge(
        self,
        source: str,
        target: str,
        relation: str,
        confidence: str = "EXTRACTED",
        **attrs,
    ) -> None:
        edge = {
            "source": source,
            "target": target,
            "relation": relation,
            "confidence": confidence,
            **{k: v for k, v in attrs.items() if v is not None},
        }
        key = (
            edge["source"],
            edge["target"],
            edge["relation"],
            edge.get("source_file", ""),
            edge.get("source_location", ""),
        )
        existing = {
            (
                e["source"],
                e["target"],
                e["relation"],
                e.get("source_file", ""),
                e.get("source_location", ""),
            )
            for e in self.edges
        }
        if key not in existing:
            self.edges.append(edge)

    def to_dict(self) -> dict:
        return {
            "nodes": sorted(self.nodes.values(), key=lambda n: n["id"]),
            "edges": sorted(
                self.edges,
                key=lambda e: (
                    e["source"],
                    e["target"],
                    e["relation"],
                    e.get("source_location", ""),
                ),
            ),
        }


class _PythonVisitor(ast.NodeVisitor):
    def __init__(self, graph: CodeGraph, path: Path, root: Path) -> None:
        self.graph = graph
        self.path = path
        self.root = root
        self.rel = _rel_path(path, root)
        self.file_id = _normalize_id(self.rel)
        self.scope_stack: list[str] = []
        self.symbols: dict[str, str] = {}

    def visit_Module(self, node: ast.Module) -> None:
        self.graph.add_node(
            self.file_id,
            label=self.rel,
            type="file",
            file_type="code",
            source_file=self.rel,
            source_location="L1",
        )
        self._collect_symbols(node)
        self.generic_visit(node)

    def _collect_symbols(self, node: ast.Module) -> None:
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                node_id = _normalize_id(self.rel, child.name)
                self.symbols[child.name] = node_id

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._add_import(alias.name, node.lineno)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if not node.module:
            return
        for alias in node.names:
            imported = f"{node.module}.{alias.name}" if alias.name != "*" else node.module
            self._add_import(imported, node.lineno)

    def _add_import(self, imported: str, line: int) -> None:
        target_id = _normalize_id("external", imported)
        self.graph.add_node(
            target_id,
            label=imported,
            type="external_module",
            file_type="code",
            source_file="",
            source_location=None,
        )
        self.graph.add_edge(
            self.file_id,
            target_id,
            "imports",
            "EXTRACTED",
            source_file=self.rel,
            source_location=f"L{line}",
        )

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        class_id = self.symbols.get(node.name, _normalize_id(self.rel, node.name))
        self._add_symbol(class_id, node.name, "class", node.lineno)
        for base in node.bases:
            base_name = self._call_name(base)
            if base_name:
                base_id = self.symbols.get(base_name) or _normalize_id("external", base_name)
                self.graph.add_node(
                    base_id,
                    label=base_name,
                    type="class",
                    file_type="code",
                    source_file="" if base_id.startswith("external_") else self.rel,
                )
                self.graph.add_edge(
                    class_id,
                    base_id,
                    "inherits",
                    "EXTRACTED",
                    source_file=self.rel,
                    source_location=f"L{node.lineno}",
                )
        self.scope_stack.append(class_id)
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node, "function")

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node, "function")

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef, node_type: str) -> None:
        symbol_id = _normalize_id(self.rel, *(self._scope_names()), node.name)
        if not self.scope_stack:
            symbol_id = self.symbols.get(node.name, symbol_id)
        self._add_symbol(symbol_id, node.name, node_type, node.lineno)
        self.scope_stack.append(symbol_id)
        self.generic_visit(node)
        self.scope_stack.pop()

    def _scope_names(self) -> list[str]:
        labels = []
        for node_id in self.scope_stack:
            label = self.graph.nodes.get(node_id, {}).get("label")
            if label:
                labels.append(label)
        return labels

    def _add_symbol(self, node_id: str, label: str, node_type: str, line: int) -> None:
        self.graph.add_node(
            node_id,
            label=label,
            type=node_type,
            file_type="code",
            source_file=self.rel,
            source_location=f"L{line}",
        )
        parent = self.scope_stack[-1] if self.scope_stack else self.file_id
        self.graph.add_edge(
            parent,
            node_id,
            "contains",
            "EXTRACTED",
            source_file=self.rel,
            source_location=f"L{line}",
        )

    def visit_Call(self, node: ast.Call) -> None:
        caller = self.scope_stack[-1] if self.scope_stack else self.file_id
        callee_name = self._call_name(node.func)
        if callee_name:
            short_name = callee_name.rsplit(".", 1)[-1]
            callee_id = self.symbols.get(short_name)
            confidence = "EXTRACTED"
            if callee_id is None:
                callee_id = _normalize_id("external", callee_name)
                confidence = "INFERRED"
                self.graph.add_node(
                    callee_id,
                    label=callee_name,
                    type="external_call",
                    file_type="code",
                    source_file="",
                )
            self.graph.add_edge(
                caller,
                callee_id,
                "calls",
                confidence,
                source_file=self.rel,
                source_location=f"L{getattr(node, 'lineno', 0)}",
            )
        self.generic_visit(node)

    @staticmethod
    def _call_name(node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parent = _PythonVisitor._call_name(node.value)
            return f"{parent}.{node.attr}" if parent else node.attr
        return None


_CREATE_TABLE_RE = re.compile(
    r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([A-Za-z_][\w.]*).*?;",
    re.IGNORECASE | re.DOTALL,
)
_FK_RE = re.compile(
    r"REFERENCES\s+([A-Za-z_][\w.]*)",
    re.IGNORECASE,
)


def extract_sql_file(path: Path, root: Path) -> CodeGraph:
    graph = CodeGraph()
    rel = _rel_path(path, root)
    file_id = _normalize_id(rel)
    graph.add_node(file_id, label=rel, type="file", file_type="code", source_file=rel, source_location="L1")
    text = path.read_text(encoding="utf-8", errors="replace")

    for match in _CREATE_TABLE_RE.finditer(text):
        table = match.group(1)
        table_id = _normalize_id("sql", table)
        line = text[: match.start()].count("\n") + 1
        graph.add_node(
            table_id,
            label=table,
            type="table",
            file_type="code",
            source_file=rel,
            source_location=f"L{line}",
        )
        graph.add_edge(file_id, table_id, "defines", "EXTRACTED", source_file=rel, source_location=f"L{line}")
        body = match.group(0)
        for fk in _FK_RE.finditer(body):
            target = fk.group(1)
            target_id = _normalize_id("sql", target)
            graph.add_node(target_id, label=target, type="table", file_type="code", source_file="")
            graph.add_edge(
                table_id,
                target_id,
                "references",
                "EXTRACTED",
                source_file=rel,
                source_location=f"L{line + body[: fk.start()].count(chr(10))}",
            )
    return graph


def extract_python_file(path: Path, root: Path) -> CodeGraph:
    graph = CodeGraph()
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError:
        rel = _rel_path(path, root)
        graph.add_node(
            _normalize_id(rel),
            label=rel,
            type="file",
            file_type="code",
            source_file=rel,
            source_location="L1",
            parse_error="syntax_error",
        )
        return graph
    _PythonVisitor(graph, path, root).visit(tree)
    return graph


def extract_code_file(path: Path, root: Path) -> CodeGraph:
    if path.suffix.lower() == ".py":
        return extract_python_file(path, root)
    if path.suffix.lower() == ".sql":
        return extract_sql_file(path, root)
    return CodeGraph()


def collect_code_files(paths: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for base in paths:
        if not base.exists():
            continue
        if base.is_file() and base.suffix.lower() in CODE_EXTENSIONS:
            files.append(base)
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file() and path.suffix.lower() in CODE_EXTENSIONS:
                if any(part in {".git", "__pycache__", ".venv", "venv", "node_modules"} for part in path.parts):
                    continue
                files.append(path)
    return sorted(set(files))


def build_code_graph(paths: Iterable[Path]) -> CodeGraph:
    path_list = [Path(p) for p in paths]
    root = path_list[0].resolve() if len(path_list) == 1 and path_list[0].is_dir() else Path.cwd().resolve()
    graph = CodeGraph()
    for path in collect_code_files(path_list):
        subgraph = extract_code_file(path, root)
        for node in subgraph.nodes.values():
            node_id = node["id"]
            graph.add_node(node_id, **{k: v for k, v in node.items() if k != "id"})
        for edge in subgraph.edges:
            graph.add_edge(**edge)
    return graph


def render_report(graph: CodeGraph, collection: str) -> str:
    data = graph.to_dict()
    degree: dict[str, int] = {}
    for edge in data["edges"]:
        degree[edge["source"]] = degree.get(edge["source"], 0) + 1
        degree[edge["target"]] = degree.get(edge["target"], 0) + 1

    top_nodes = sorted(degree.items(), key=lambda item: item[1], reverse=True)[:10]
    relation_counts: dict[str, int] = {}
    confidence_counts: dict[str, int] = {}
    for edge in data["edges"]:
        relation_counts[edge["relation"]] = relation_counts.get(edge["relation"], 0) + 1
        confidence_counts[edge["confidence"]] = confidence_counts.get(edge["confidence"], 0) + 1

    lines = [
        f"# Code Graph Report: {collection}",
        "",
        f"- Nodes: {len(data['nodes'])}",
        f"- Edges: {len(data['edges'])}",
        "",
        "## Top Nodes",
    ]
    for node_id, count in top_nodes:
        label = graph.nodes.get(node_id, {}).get("label", node_id)
        lines.append(f"- {label} ({count} connections)")
    lines.extend(["", "## Relations"])
    for rel, count in sorted(relation_counts.items()):
        lines.append(f"- {rel}: {count}")
    lines.extend(["", "## Confidence"])
    for conf, count in sorted(confidence_counts.items()):
        lines.append(f"- {conf}: {count}")
    lines.extend(["", "## Suggested Questions"])
    lines.extend([
        "- What are the most connected code entities?",
        "- Which modules import the most external dependencies?",
        "- Which functions or classes bridge different files?",
        "- Which inferred call edges need review?",
    ])
    return "\n".join(lines) + "\n"


def write_artifacts(graph: CodeGraph, out_dir: Path, collection: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = graph.to_dict()
    graph_json = out_dir / "graph.json"
    report_md = out_dir / "GRAPH_REPORT.md"
    graph_html = out_dir / "graph.html"

    graph_json.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    report = render_report(graph, collection)
    report_md.write_text(report, encoding="utf-8")

    rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(node.get('label', node['id']))}</td>"
        f"<td>{html.escape(node.get('type', ''))}</td>"
        f"<td>{html.escape(node.get('source_file', ''))}</td>"
        f"<td>{html.escape(node.get('source_location', '') or '')}</td>"
        "</tr>"
        for node in data["nodes"][:500]
    )
    graph_html.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>Ultra RAG Code Graph - {html.escape(collection)}</title>"
        "<style>body{font-family:Arial,sans-serif;margin:24px}"
        "table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:6px}"
        "th{background:#f3f3f3;text-align:left}</style></head><body>"
        f"<h1>Ultra RAG Code Graph: {html.escape(collection)}</h1>"
        f"<p>{len(data['nodes'])} nodes, {len(data['edges'])} edges</p>"
        "<h2>Nodes</h2><table><tr><th>Label</th><th>Type</th><th>Source</th><th>Location</th></tr>"
        f"{rows}</table></body></html>",
        encoding="utf-8",
    )
    return {
        "graph_json": str(graph_json),
        "report_md": str(report_md),
        "graph_html": str(graph_html),
        "nodes": len(data["nodes"]),
        "edges": len(data["edges"]),
    }


def upsert_code_graph(conn, collection: str, graph: CodeGraph) -> dict:
    data = graph.to_dict()
    id_map: dict[str, int] = {}
    with conn.cursor() as cur:
        cur.execute(
            """
            DELETE FROM rag.relationships
            WHERE collection = %s
              AND context LIKE %s
            """,
            (collection, '%"source": "code_graph"%'),
        )
        for node in data["nodes"]:
            description = json.dumps(
                {
                    "source": "code_graph",
                    "source_file": node.get("source_file"),
                    "source_location": node.get("source_location"),
                    "node_id": node["id"],
                },
                ensure_ascii=False,
            )
            cur.execute(
                """
                INSERT INTO rag.entities (collection, name, entity_type, description)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (collection, name) DO UPDATE
                    SET entity_type = EXCLUDED.entity_type,
                        description = EXCLUDED.description
                RETURNING id
                """,
                (collection, node.get("label", node["id"]), node.get("type", "code"), description),
            )
            id_map[node["id"]] = cur.fetchone()[0]

        rel_count = 0
        for edge in data["edges"]:
            source_id = id_map.get(edge["source"])
            target_id = id_map.get(edge["target"])
            if not source_id or not target_id:
                continue
            context = json.dumps(
                {
                    "source": "code_graph",
                    "confidence": edge.get("confidence"),
                    "source_file": edge.get("source_file"),
                    "source_location": edge.get("source_location"),
                },
                ensure_ascii=False,
            )
            cur.execute(
                """
                INSERT INTO rag.relationships
                    (collection, source_id, target_id, rel_type, context)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (collection, source_id, target_id, edge.get("relation", "related_to"), context),
            )
            rel_count += 1
    conn.commit()
    return {"entities_upserted": len(id_map), "relationships_inserted": rel_count}


def process_collection(conn, collection: str, out_root: Path | None = None) -> dict:
    cfg = get_config()
    collection_cfg = cfg.get("collections", {}).get(collection, {})
    paths = [Path(p) for p in collection_cfg.get("paths", [])]
    graph = build_code_graph(paths)
    configured_out = cfg.get("code_graph", {}).get("output_dir", "graph-out")
    out_dir = (out_root or Path(configured_out)) / collection
    artifact_stats = write_artifacts(graph, out_dir, collection)
    db_stats = upsert_code_graph(conn, collection, graph) if conn is not None else {}
    return {**artifact_stats, **db_stats}
