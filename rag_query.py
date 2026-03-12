#!/usr/bin/env python3
"""
RAG Query CLI — search the knowledge base.

Usage:
  python rag_query.py "equipment ID screen 207" --collection imds
  python rag_query.py "NFSPC0" --collection imds --type field_definition
  python rag_query.py "how to close a job" --collection imds --top 10
  python rag_query.py "screen 207" --collection imds --tier 1   # keyword only
  python rag_query.py "screen 207" --collection imds --tier 2   # vector only
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
    parser = argparse.ArgumentParser(description="RAG Query CLI")
    parser.add_argument("query",       help="Search query")
    parser.add_argument("--collection",default="imds", help="Collection name")
    parser.add_argument("--top",  "-k", type=int, default=5, help="Top results")
    parser.add_argument("--type",       help="Filter by content_type")
    parser.add_argument("--screen",     help="Filter by IMDS screen number")
    parser.add_argument("--tier",  type=int, choices=[1,2,3], help="Force search tier")
    parser.add_argument("--json",       action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    conn = get_conn()
    try:
        create_schema(conn)

        meta_filter = None
        if args.screen:
            meta_filter = {"imds_screens": [args.screen]}

        results = search(
            conn=conn,
            query=args.query,
            collection=args.collection,
            top_k=args.top,
            content_type=args.type,
            metadata_filter=meta_filter,
            force_tier=args.tier,
        )

        if args.json:
            # Serialize (embeddings are not JSON-serializable)
            out = []
            for r in results:
                d = dict(r)
                d.pop("embedding", None)
                out.append(d)
            print(json.dumps(out, indent=2, default=str))
            return

        print(f"\n{'='*70}")
        print(f"Query:      {args.query}")
        print(f"Collection: {args.collection}  Top-{args.top}")
        print(f"{'='*70}")

        if not results:
            print("No results found.")
            return

        for i, r in enumerate(results, 1):
            ctx   = r.get("context_prefix", "") or ""
            ctype = r.get("content_type", "text")
            score = r.get("score", 0)
            tok   = r.get("token_count", 0)
            meta  = r.get("chunk_metadata", {}) or {}
            screens = meta.get("imds_screens", [])

            print(f"\n[{i}] {ctype.upper():<18} score={score:.5f}  {tok} tokens")
            if ctx:
                print(f"    {ctx}")
            if screens:
                print(f"    Screens: {', '.join(screens)}")
            print()
            # Show content (up to 800 chars)
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
