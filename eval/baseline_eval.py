#!/usr/bin/env python3
"""
Baseline evaluation wrapper for Ultra RAG.

Generates 50 synthetic Q&A pairs, runs RAGAS evaluation, prints results,
asserts minimum quality thresholds, and saves results to baseline_results.json.

Usage:
    cd D:\\rag-ingest
    python eval/baseline_eval.py                              # imds, baseline run
    python eval/baseline_eval.py --collection personal        # different collection
    python eval/baseline_eval.py --run-name "bge-reranker-v2-m3"
    python eval/baseline_eval.py --n 30 --no-generate         # reuse existing questions
"""
import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# ── Project root on sys.path ──────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# ── Logging ───────────────────────────────────────────────────────────────────
logs_dir = _ROOT / "logs"
logs_dir.mkdir(exist_ok=True)
log_file = logs_dir / f"baseline_eval_{datetime.now():%Y%m%d}.log"

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

# ── Quality thresholds ────────────────────────────────────────────────────────
THRESHOLDS = {
    "context_recall":    0.60,
    "faithfulness":      0.70,
}

RESULTS_FILE = Path(__file__).parent / "baseline_results.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_existing_questions(conn, collection: str, limit: int = 500) -> list:
    import psycopg2.extras
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


def _print_metrics(scores: dict) -> None:
    print(f"\n{'='*55}")
    print("RAGAS Baseline Evaluation Results")
    print(f"{'='*55}")
    metrics = [
        ("Faithfulness",       "faithfulness"),
        ("Answer Relevancy",   "answer_relevancy"),
        ("Context Precision",  "context_precision"),
        ("Context Recall",     "context_recall"),
    ]
    for label, key in metrics:
        val = scores.get(key, 0.0) or 0.0
        bar = "█" * int(val * 20) + "░" * (20 - int(val * 20))
        threshold = THRESHOLDS.get(key)
        flag = ""
        if threshold is not None:
            flag = "  ✓" if val >= threshold else f"  ✗ (need ≥{threshold})"
        print(f"  {label:<22}  {val:.4f}  |{bar}|{flag}")
    print(f"\n  Questions evaluated: {scores.get('n_questions', 0)}")
    if scores.get("run_name"):
        print(f"  Run name: {scores['run_name']}")
    if scores.get("run_id"):
        print(f"  Run id:   {scores['run_id']}")
    print(f"{'='*55}\n")


def _check_thresholds(scores: dict) -> list:
    """Return list of (metric, actual, threshold) tuples that FAILED."""
    failures = []
    for metric, threshold in THRESHOLDS.items():
        actual = scores.get(metric, 0.0) or 0.0
        if actual < threshold:
            failures.append((metric, actual, threshold))
    return failures


def _save_results(scores: dict, run_name: str, collection: str) -> None:
    """Append this run's results to baseline_results.json."""
    record = {
        "timestamp":   datetime.now().isoformat(),
        "run_name":    run_name,
        "collection":  collection,
        "scores":      scores,
        "thresholds":  THRESHOLDS,
        "passed":      len(_check_thresholds(scores)) == 0,
    }

    existing: list = []
    if RESULTS_FILE.exists():
        try:
            existing = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
            if not isinstance(existing, list):
                existing = [existing]
        except Exception:
            existing = []

    existing.append(record)
    RESULTS_FILE.write_text(
        json.dumps(existing, indent=2, default=str),
        encoding="utf-8",
    )
    log.info("Results saved to %s", RESULTS_FILE)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ultra RAG baseline evaluation — generate Q&A + run RAGAS",
    )
    parser.add_argument(
        "--collection", "-c",
        default="imds",
        help="Collection to evaluate (default: imds)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=50,
        help="Number of Q&A pairs to generate (default: 50)",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Label for this eval run (default: baseline_<timestamp>)",
    )
    parser.add_argument(
        "--no-generate",
        action="store_true",
        help="Skip generation — reuse the most recent existing questions",
    )
    parser.add_argument(
        "--no-assert",
        action="store_true",
        help="Print results but do not exit non-zero on threshold failures",
    )
    parser.add_argument(
        "--strategy",
        default=None,
        choices=["hybrid", "kg_local", "kg_global", "multi_hop", "hyde", "compound"],
        help="Search strategy override for retrieval (default: standard hybrid)",
    )
    args = parser.parse_args()

    run_name = args.run_name or f"baseline_{datetime.now():%Y%m%d_%H%M}"

    from src.db       import get_conn
    from src.db_ultra import create_ultra_schema

    conn = get_conn()
    try:
        create_ultra_schema(conn)

        questions: list = []

        # ── Step 1: generate Q&A pairs ────────────────────────────────
        if not args.no_generate:
            from src.eval_generator import EvalDatasetGenerator

            log.info("=" * 60)
            log.info("Generating %d questions for collection '%s'", args.n, args.collection)
            log.info("=" * 60)

            generator = EvalDatasetGenerator(conn, args.collection)
            questions = generator.generate_dataset(n_questions=args.n)
            log.info("Generated %d questions", len(questions))
        else:
            log.info("--no-generate: loading existing questions for '%s'", args.collection)
            questions = _load_existing_questions(conn, args.collection, limit=args.n)
            if not questions:
                log.error(
                    "No existing questions found for '%s'. "
                    "Run without --no-generate first.",
                    args.collection,
                )
                return 1
            log.info("Loaded %d existing questions", len(questions))

        # ── Step 2: run RAGAS evaluation ──────────────────────────────
        from src.eval_runner import RAGASEvalRunner
        from src.search      import search as base_search

        if args.strategy:
            from ultra_query import _run_strategy
            def search_fn(q: str, top_k: int = 5):
                return _run_strategy(conn, q, args.collection, args.strategy, top_k)
            log.info("Using strategy override: %s", args.strategy)
        else:
            def search_fn(q: str, top_k: int = 5):
                return base_search(conn, q, args.collection, top_k=top_k)

        log.info("=" * 60)
        log.info("Running RAGAS evaluation — run_name=%s", run_name)
        log.info("=" * 60)

        runner = RAGASEvalRunner(conn, args.collection)
        scores = runner.evaluate_dataset(questions, search_fn, run_name=run_name)

        # ── Step 3: print + save ──────────────────────────────────────
        _print_metrics(scores)
        _save_results(scores, run_name, args.collection)

        # ── Step 4: assert thresholds ─────────────────────────────────
        failures = _check_thresholds(scores)
        if failures:
            print("THRESHOLD FAILURES:")
            for metric, actual, threshold in failures:
                print(f"  {metric}: got {actual:.4f}, need >= {threshold}")
            if not args.no_assert:
                print("\nRun with --no-assert to skip exit code on failures.")
                return 1
        else:
            print("All quality thresholds passed.")

        return 0

    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception as exc:
        log.error("Fatal error: %s", exc, exc_info=True)
        return 1
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
