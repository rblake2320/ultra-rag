#!/usr/bin/env python3
"""
Compare two Ultra RAG eval runs side-by-side with metric deltas.

Usage:
    cd D:\\rag-ingest
    python eval/compare_runs.py                                    # list all runs
    python eval/compare_runs.py --a baseline --b "bge-reranker-v2-m3"
    python eval/compare_runs.py --a 1 --b 3                        # by run ID
    python eval/compare_runs.py --collection personal --list
"""
import argparse
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)-8s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

METRICS = [
    ("faithfulness",       "Faithfulness"),
    ("answer_relevancy",   "Answer Relevancy"),
    ("context_precision",  "Context Precision"),
    ("context_recall",     "Context Recall"),
]


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _list_runs(conn, collection: str) -> list:
    import psycopg2.extras
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT id, run_name, faithfulness, answer_relevancy,
                   context_precision, context_recall, n_questions, created_at
            FROM   rag.eval_runs
            WHERE  collection = %s
            ORDER  BY created_at DESC
            LIMIT  100
        """, (collection,))
        return [dict(r) for r in cur.fetchall()]


def _fetch_run(conn, collection: str, ref: str) -> dict | None:
    """Fetch a single run by ID (integer) or run_name (string)."""
    import psycopg2.extras
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        # Try by ID first
        if ref.isdigit():
            cur.execute("""
                SELECT id, run_name, faithfulness, answer_relevancy,
                       context_precision, context_recall, n_questions, created_at
                FROM   rag.eval_runs
                WHERE  id = %s AND collection = %s
            """, (int(ref), collection))
        else:
            # Match by run_name (most recent if duplicates)
            cur.execute("""
                SELECT id, run_name, faithfulness, answer_relevancy,
                       context_precision, context_recall, n_questions, created_at
                FROM   rag.eval_runs
                WHERE  run_name ILIKE %s AND collection = %s
                ORDER  BY created_at DESC
                LIMIT  1
            """, (f"%{ref}%", collection))
        row = cur.fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _print_run_list(runs: list, collection: str) -> None:
    if not runs:
        print(f"\nNo eval runs found for collection '{collection}'.")
        return
    print(f"\n{'='*80}")
    print(f"Eval Runs — collection: {collection}")
    print(f"{'='*80}")
    hdr = (
        f"{'ID':>4}  {'Name':<28}  {'Faith':>6}  {'AnsRel':>6}  "
        f"{'CtxPre':>6}  {'CtxRec':>6}  {'N':>4}  {'Date':<20}"
    )
    print(hdr)
    print("─" * 80)
    for r in runs:
        name = (r.get("run_name") or "unnamed")[:28]
        ts   = str(r.get("created_at", ""))[:19]
        print(
            f"{r['id']:>4}  {name:<28}  "
            f"{r['faithfulness'] or 0:>6.3f}  "
            f"{r['answer_relevancy'] or 0:>6.3f}  "
            f"{r['context_precision'] or 0:>6.3f}  "
            f"{r['context_recall'] or 0:>6.3f}  "
            f"{r['n_questions'] or 0:>4}  {ts:<20}"
        )
    print(f"{'='*80}\n")


def _delta_str(before: float, after: float) -> str:
    """Format delta with sign and color-like indicator."""
    diff = after - before
    if abs(diff) < 0.001:
        return "  (no change)"
    arrow = "▲" if diff > 0 else "▼"
    return f"  {arrow} {diff:+.4f}"


def _print_comparison(run_a: dict, run_b: dict) -> None:
    name_a = (run_a.get("run_name") or f"run #{run_a['id']}")[:30]
    name_b = (run_b.get("run_name") or f"run #{run_b['id']}")[:30]

    print(f"\n{'='*72}")
    print("Eval Run Comparison")
    print(f"{'='*72}")
    print(f"  A: [{run_a['id']:>3}] {name_a}")
    print(f"  B: [{run_b['id']:>3}] {name_b}")
    print(f"{'─'*72}")
    header = f"  {'Metric':<22}  {'A':>7}  {'B':>7}  {'Delta':<16}"
    print(header)
    print(f"{'─'*72}")

    improvements = 0
    regressions  = 0

    for key, label in METRICS:
        a_val = run_a.get(key) or 0.0
        b_val = run_b.get(key) or 0.0
        delta = _delta_str(a_val, b_val)
        diff  = b_val - a_val
        if diff > 0.001:
            improvements += 1
        elif diff < -0.001:
            regressions += 1
        print(f"  {label:<22}  {a_val:>7.4f}  {b_val:>7.4f}  {delta}")

    print(f"{'─'*72}")
    print(f"  Questions A: {run_a.get('n_questions') or 0}   B: {run_b.get('n_questions') or 0}")
    print(f"  Improvements: {improvements}   Regressions: {regressions}")

    # Overall verdict
    total_a = sum((run_a.get(k) or 0.0) for k, _ in METRICS)
    total_b = sum((run_b.get(k) or 0.0) for k, _ in METRICS)
    avg_a   = total_a / len(METRICS)
    avg_b   = total_b / len(METRICS)
    verdict = "B is better" if avg_b > avg_a + 0.001 else (
              "A is better" if avg_a > avg_b + 0.001 else "roughly equal")
    print(f"\n  Average score — A: {avg_a:.4f}  B: {avg_b:.4f}  → {verdict}")
    print(f"{'='*72}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare two Ultra RAG eval runs side-by-side",
    )
    parser.add_argument(
        "--collection", "-c",
        default="imds",
        help="Collection name (default: imds)",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all runs for the collection (no comparison)",
    )
    parser.add_argument(
        "--a",
        default=None,
        help="First run: ID (integer) or run_name substring",
    )
    parser.add_argument(
        "--b",
        default=None,
        help="Second run: ID (integer) or run_name substring",
    )
    args = parser.parse_args()

    from src.db import get_conn
    conn = get_conn()
    try:
        runs = _list_runs(conn, args.collection)

        if args.list or (not args.a and not args.b):
            _print_run_list(runs, args.collection)
            if not args.a and not args.b:
                print("Tip: use --a <name_or_id> --b <name_or_id> to compare two runs.")
            return 0

        if not args.a or not args.b:
            # Auto-pick: compare last two runs
            if len(runs) < 2:
                print("Need at least 2 eval runs to compare. Run baseline_eval.py first.")
                return 1
            run_a, run_b = runs[1], runs[0]
            print(f"Auto-comparing last two runs: #{run_a['id']} vs #{run_b['id']}")
        else:
            run_a = _fetch_run(conn, args.collection, args.a)
            run_b = _fetch_run(conn, args.collection, args.b)
            if run_a is None:
                print(f"Run not found: '{args.a}'")
                return 1
            if run_b is None:
                print(f"Run not found: '{args.b}'")
                return 1

        _print_comparison(run_a, run_b)
        return 0

    except Exception as exc:
        log.error("Error: %s", exc, exc_info=True)
        return 1
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
