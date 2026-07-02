#!/usr/bin/env python3
"""Build deterministic code graph artifacts for an Ultra RAG collection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Ultra RAG code graph artifacts")
    parser.add_argument("collection", help="Collection name from config.yaml")
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="Write graph artifacts without upserting entities/relationships",
    )
    parser.add_argument(
        "--out",
        default="graph-out",
        help="Output root directory (default: graph-out)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable stats",
    )
    args = parser.parse_args()

    from src.code_graph import process_collection

    conn = None
    if not args.no_db:
        from src.db import create_schema, get_conn
        from src.db_ultra import create_ultra_schema

        conn = get_conn()
        create_schema(conn)
        create_ultra_schema(conn)

    try:
        stats = process_collection(conn, args.collection, out_root=Path(args.out))
    finally:
        if conn is not None:
            conn.close()

    if args.json:
        print(json.dumps(stats, indent=2))
    else:
        print(
            f"Code graph built: {stats['nodes']} nodes, {stats['edges']} edges\n"
            f"  graph:  {stats['graph_json']}\n"
            f"  report: {stats['report_md']}\n"
            f"  html:   {stats['graph_html']}"
        )


if __name__ == "__main__":
    main()
