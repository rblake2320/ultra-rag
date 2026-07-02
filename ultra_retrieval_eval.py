#!/usr/bin/env python3
"""Run deterministic retrieval-quality evaluation for Ultra RAG."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _live_search(collection: str):
    from src.db import get_conn
    from src.search import search

    conn = get_conn()

    def _search(query: str, k: int) -> list[dict]:
        rows = search(conn, query, collection, top_k=k)
        for row in rows:
            if "stable_id" not in row and row.get("id") is not None:
                row["stable_id"] = str(row["id"])
        return rows

    return conn, _search


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval results against golden query relevance labels"
    )
    parser.add_argument(
        "--golden",
        default="eval/golden_retrieval.jsonl",
        help="JSONL golden cases path (default: eval/golden_retrieval.jsonl)",
    )
    parser.add_argument(
        "--fixture-results",
        default=None,
        help="Optional JSON fixture mapping queries to ordered result rows",
    )
    parser.add_argument(
        "--collection",
        default="default",
        help="Collection for live DB search when --fixture-results is omitted",
    )
    parser.add_argument("--k", type=int, default=5, help="Evaluation cutoff")
    parser.add_argument("--json", action="store_true", help="Print JSON only")
    args = parser.parse_args()

    from src.retrieval_eval import evaluate_results, load_golden_queries, search_fixture

    cases = load_golden_queries(args.golden)
    conn = None
    if args.fixture_results:
        search_fn = search_fixture(args.fixture_results)
    else:
        conn, search_fn = _live_search(args.collection)

    try:
        result = evaluate_results(cases, search_fn, k=args.k)
    finally:
        if conn is not None:
            conn.close()

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        status = "PASS" if result["passed"] else "FAIL"
        print(f"Retrieval eval: {status}")
        print(f"  cases:       {result['n_cases']}")
        print(f"  precision@{args.k}: {result['precision_at_k']:.3f}")
        print(f"  recall@{args.k}:    {result['recall_at_k']:.3f}")
        print(f"  mrr:         {result['mrr']:.3f}")
        print(f"  ndcg@{args.k}:      {result['ndcg_at_k']:.3f}")
        for case in result["cases"]:
            if not case["passed"]:
                print(f"  failed: {case['query']}")

    if not result["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
