#!/usr/bin/env python3
"""
Ultra RAG Eval: generate synthetic evaluation datasets and run RAGAS evaluation.

Generates Q&A pairs at three difficulty levels (extractive, abstractive,
multi-hop) inspired by NeMo Data Designer, then evaluates the RAG pipeline
with four RAGAS metrics: faithfulness, answer relevancy, context precision,
and context recall.

Usage:
  python ultra_eval.py imds --generate 50              # generate 50 questions
  python ultra_eval.py imds --run                      # run eval on existing questions
  python ultra_eval.py imds --generate 50 --run        # generate + eval in one pass
  python ultra_eval.py imds --generate 20 --run --run-name "v1-baseline"
  python ultra_eval.py imds --report                   # show past eval runs
  python ultra_eval.py imds --export /tmp/eval.jsonl   # export questions to JSONL
"""
import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# ── Logging ───────────────────────────────────────────────────────────────────
logs_dir = Path(__file__).parent / "logs"
logs_dir.mkdir(exist_ok=True)
log_file = logs_dir / f"ultra_eval_{datetime.now():%Y%m%d}.log"

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── Project root ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_existing_questions(conn, collection: str, limit: int = 500) -> list:
    """Load evaluation questions from rag.eval_questions for this collection."""
    import psycopg2.extras  # noqa: PLC0415
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT id AS eval_question_id, question, ground_truth,
                   difficulty, source_chunk_ids, created_at
            FROM   rag.eval_questions
            WHERE  collection = %s
            ORDER  BY created_at DESC
            LIMIT  %s
        """, (collection, limit))
        return [dict(r) for r in cur.fetchall()]


def _print_report(conn, collection: str) -> None:
    """Print a formatted table of past eval runs for *collection*."""
    import psycopg2.extras  # noqa: PLC0415
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT id, run_name, faithfulness, answer_relevancy,
                   context_precision, context_recall, n_questions, created_at
            FROM   rag.eval_runs
            WHERE  collection = %s
            ORDER  BY created_at DESC
            LIMIT  50
        """, (collection,))
        rows = cur.fetchall()

    if not rows:
        print(f"\nNo eval runs found for collection '{collection}'.")
        return

    print(f"\n{'='*80}")
    print(f"Eval Runs — collection: {collection}")
    print(f"{'='*80}")
    hdr = (
        f"{'ID':>4}  {'Name':<24}  {'Faith':>6}  {'AnsRel':>6}  "
        f"{'CtxPre':>6}  {'CtxRec':>6}  {'N':>4}  {'Date':<20}"
    )
    print(hdr)
    print("─" * 80)
    for r in rows:
        name = (r.get("run_name") or "unnamed")[:24]
        ts   = str(r.get("created_at", ""))[:19]
        print(
            f"{r['id']:>4}  {name:<24}  "
            f"{r['faithfulness'] or 0:>6.3f}  "
            f"{r['answer_relevancy'] or 0:>6.3f}  "
            f"{r['context_precision'] or 0:>6.3f}  "
            f"{r['context_recall'] or 0:>6.3f}  "
            f"{r['n_questions'] or 0:>4}  {ts:<20}"
        )
    print(f"{'='*80}\n")


def _print_metrics(scores: dict) -> None:
    """Pretty-print RAGAS metric scores."""
    print(f"\n{'='*50}")
    print("RAGAS Evaluation Results")
    print(f"{'='*50}")
    metrics = [
        ("Faithfulness",       "faithfulness"),
        ("Answer Relevancy",   "answer_relevancy"),
        ("Context Precision",  "context_precision"),
        ("Context Recall",     "context_recall"),
    ]
    for label, key in metrics:
        val = scores.get(key, 0.0)
        bar = "█" * int(val * 20) + "░" * (20 - int(val * 20))
        print(f"  {label:<22}  {val:.4f}  |{bar}|")
    print(f"\n  Questions evaluated: {scores.get('n_questions', 0)}")
    if scores.get("run_name"):
        print(f"  Run name: {scores['run_name']}")
    if scores.get("run_id"):
        print(f"  Run id:   {scores['run_id']}")
    print(f"{'='*50}\n")


def _print_question_summary(questions: list) -> None:
    """Print a summary of generated questions."""
    if not questions:
        print("No questions generated.")
        return

    by_difficulty: dict[str, int] = {}
    for q in questions:
        d = q.get("difficulty", "unknown")
        by_difficulty[d] = by_difficulty.get(d, 0) + 1

    print(f"\n  Generated {len(questions)} questions:")
    for diff, count in sorted(by_difficulty.items()):
        print(f"    {diff:<14} {count}")

    # Show a few examples
    print("\n  Sample questions:")
    shown = 0
    for q in questions:
        if shown >= 3:
            break
        diff = q.get("difficulty", "?")
        text = q.get("question", "")
        if text:
            print(f"    [{diff}] {text[:90]}")
            shown += 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Ultra RAG Eval — synthetic dataset generation + RAGAS evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "collection",
        help="Collection name (must exist in the RAG database)",
    )
    parser.add_argument(
        "--generate", "-g",
        metavar="N",
        type=int,
        default=None,
        help="Generate N synthetic Q&A questions",
    )
    parser.add_argument(
        "--run", "-r",
        action="store_true",
        help="Run RAGAS evaluation on (generated or existing) questions",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Show past eval run results for this collection",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Human-readable name for this eval run",
    )
    parser.add_argument(
        "--export",
        metavar="PATH",
        default=None,
        help="Export questions to a JSONL file at PATH",
    )
    parser.add_argument(
        "--distribution",
        default="0.4,0.4,0.2",
        help="Question distribution as extractive,abstractive,multi_hop fractions "
             "(default: 0.4,0.4,0.2)",
    )
    parser.add_argument(
        "--strategy",
        default=None,
        choices=["hybrid", "kg_local", "kg_global", "multi_hop", "hyde", "compound"],
        help="Search strategy for evaluation retrieval (default: auto-route)",
    )
    args = parser.parse_args()

    # Must do at least one of: generate, run, report
    if not any([args.generate, args.run, args.report, args.export]):
        parser.print_help()
        print("\nError: specify at least one of --generate N, --run, --report, or --export.")
        sys.exit(1)

    # Parse distribution
    try:
        parts = [float(x.strip()) for x in args.distribution.split(",")]
        if len(parts) == 3:
            distribution = {
                "extractive":  parts[0],
                "abstractive": parts[1],
                "multi_hop":   parts[2],
            }
        else:
            log.warning("Invalid --distribution; using defaults")
            distribution = None
    except ValueError:
        log.warning("Could not parse --distribution; using defaults")
        distribution = None

    # Connect + schema
    from src.db       import get_conn             # noqa: PLC0415
    from src.db_ultra import create_ultra_schema  # noqa: PLC0415

    conn = get_conn()
    try:
        create_ultra_schema(conn)

        questions: list[dict] = []

        # ── Report only ───────────────────────────────────────────────
        if args.report:
            _print_report(conn, args.collection)
            if not args.generate and not args.run:
                return

        # ── Generate ──────────────────────────────────────────────────
        if args.generate:
            from src.eval_generator import EvalDatasetGenerator  # noqa: PLC0415

            log.info("=" * 60)
            log.info("Generating %d questions for collection '%s'",
                     args.generate, args.collection)
            log.info("=" * 60)

            generator = EvalDatasetGenerator(conn, args.collection)
            questions = generator.generate_dataset(
                n_questions=args.generate,
                distribution=distribution,
            )
            _print_question_summary(questions)

            if args.export:
                generator.export_jsonl(questions, args.export)
                print(f"\n  Exported to: {args.export}")

        # ── Run eval (generate + existing) ───────────────────────────
        if args.run:
            from src.eval_runner import RAGASEvalRunner  # noqa: PLC0415
            from src.search      import search as base_search  # noqa: PLC0415

            # If we didn't generate in this run, load existing questions
            if not questions:
                log.info("Loading existing questions for collection '%s'", args.collection)
                questions = _load_existing_questions(conn, args.collection)
                if not questions:
                    print(f"\nNo questions found for collection '{args.collection}'.")
                    print("Run with --generate N first to create questions.")
                    return
                log.info("Loaded %d existing questions", len(questions))

            # Build search function with optional strategy override
            if args.strategy:
                from ultra_query import _run_strategy  # noqa: PLC0415
                def search_fn(q: str, top_k: int = 5):
                    return _run_strategy(conn, q, args.collection, args.strategy, top_k)
                log.info("Using strategy: %s", args.strategy)
            else:
                def search_fn(q: str, top_k: int = 5):
                    return base_search(conn, q, args.collection, top_k=top_k)

            run_name = args.run_name or f"eval_{datetime.now():%Y%m%d_%H%M}"

            log.info("=" * 60)
            log.info("Running RAGAS evaluation — run_name=%s", run_name)
            log.info("=" * 60)

            runner = RAGASEvalRunner(conn, args.collection)
            scores = runner.evaluate_dataset(questions, search_fn, run_name=run_name)

            _print_metrics(scores)

        # ── Export only (no generate) ─────────────────────────────────
        if args.export and not args.generate:
            if not questions:
                questions = _load_existing_questions(conn, args.collection)
            if questions:
                from src.eval_generator import EvalDatasetGenerator  # noqa: PLC0415
                gen = EvalDatasetGenerator(conn, args.collection)
                gen.export_jsonl(questions, args.export)
                print(f"\nExported {len(questions)} questions to: {args.export}")
            else:
                print(f"No questions to export for collection '{args.collection}'.")

        # ── Report (at end if combined with other actions) ────────────
        if args.report and (args.generate or args.run):
            _print_report(conn, args.collection)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as exc:
        log.error("Fatal error: %s", exc, exc_info=True)
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
