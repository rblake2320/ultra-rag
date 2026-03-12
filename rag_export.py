#!/usr/bin/env python3
"""
RAG Export CLI — export a collection to ChromaDB.

Usage:
  python rag_export.py --collection imds
  python rag_export.py --collection imds --output D:\\my-chroma
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.db           import get_conn, create_schema
from src.chroma_export import export_to_chroma


def main():
    parser = argparse.ArgumentParser(description="Export RAG collection to ChromaDB")
    parser.add_argument("--collection", default="imds", help="Collection name")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: D:\\rag-ingest\\chroma-export\\<collection>)")
    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else (
        Path(__file__).parent / "chroma-export" / args.collection
    )

    conn = get_conn()
    try:
        create_schema(conn)
        n = export_to_chroma(conn, args.collection, output_dir)
        print(f"Exported {n} chunks to {output_dir}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
