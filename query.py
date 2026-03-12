#!/usr/bin/env python3
"""
Ultra RAG Query CLI — search your knowledge base.

Uses hybrid vector + keyword search (BM25 + pgvector cosine similarity)
with Reciprocal Rank Fusion and optional NIM reranking.

For the full intelligent query pipeline with auto-routing, CRAG, Self-RAG,
HyDE, and provenance, use ultra_query.py.

Usage:
  python query.py "my question"                         # default collection
  python query.py "my question" --collection default    # explicit collection
  python query.py "my question" --top 10                # more results
  python query.py "my question" --type table            # filter by content type
  python query.py "my question" --tier 1                # keyword-only
  python query.py "my question" --tier 2                # vector-only
  python query.py "my question" --tier 3                # hybrid (default)
  python query.py "my question" --json                  # raw JSON output
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import get_config
from src.db     import get_conn, create_schema
from src.search import search


def main():
    parser = argparse.ArgumentParser(
        description="Ultra RAG Query — search the knowledge base",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("query",               help="Search query text")
    parser.add_argument(
        "--collection", "-c",
        default="default",
        help="Collection name (default: default)",
    )
    parser.add_argument(
        "--top", "-k",
        type=int, default=5,
        help="Number of results to return (default: 5)",
    )
    parser.add_argument(
        "--type",
        default=None,
        help="Filter by content_type (e.g. text, table, definition, procedure)",
    )
    parser.add_argument(
        "--tier",
        type=int, choices=[1, 2, 3],
        default=None,
        help="Force search tier: 1=keyword, 2=vector, 3=hybrid",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON",
    )
    args = parser.parse_args()

    conn = get_conn()
    try:
        create_schema(conn)

        results = search(
            conn=conn,
            query=args.query,
            collection=args.collection,
            top_k=args.top,
            content_type=args.type,
            force_tier=args.tier,
        )

        if args.json:
            out = []
            for r in results:
                d = dict(r)
                d.pop("embedding", None)
                out.append(d)
            print(json.dumps(out, indent=2, default=str))
            return

        # ── Pretty print ──────────────────────────────────────────────────────
        print(f"\n{'='*70}")
        print(f"Query:      {args.query}")
        print(f"Collection: {args.collection}   Top-{args.top}")
        if args.type:
            print(f"Filter:     content_type = {args.type}")
        print(f"{'='*70}")

        if not results:
            print("\nNo results found.")
            print("\nTips:")
            print("  - Make sure you have ingested documents: python ingest.py default --embed")
            print("  - Try a different collection name with --collection")
            print("  - Try broadening your query")
            return

        for i, r in enumerate(results, 1):
            ctx   = r.get("context_prefix", "") or ""
            ctype = r.get("content_type", "text")
            score = r.get("score", 0)
            tok   = r.get("token_count", "?")

            print(f"\n[{i}] {ctype.upper():<20}  score={score:.5f}  {tok} tokens")
            if ctx:
                print(f"    {ctx}")
            print()

            content = r.get("content", "")
            if len(content) > 800:
                content = content[:800] + "…"
            for line in content.split("\n"):
                print(f"    {line}")

        print(f"\n{'='*70}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
