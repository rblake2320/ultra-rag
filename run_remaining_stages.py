"""
Post-KG pipeline runner.

Waits for KG extraction to complete, then sequentially runs:
  communities → contextual → raptor

Usage:
    python run_remaining_stages.py [--skip-wait]

    --skip-wait   Skip the KG completion check and run stages immediately.
"""
import argparse
import logging
import subprocess
import sys
import time

import psycopg2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DB_DSN   = "postgresql://postgres:%3FBooker78%21@localhost:5432/postgres"
TOTAL_CHUNKS = 4015
CHECK_INTERVAL = 120   # seconds between KG progress checks
DONE_THRESHOLD = 0.98  # consider KG done when 98%+ chunks have entity links


def _kg_progress() -> tuple[int, int]:
    """Return (chunks_with_entities, total_chunks)."""
    conn = psycopg2.connect(DB_DSN)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(DISTINCT ce.chunk_id) FROM rag.chunk_entities ce"
                " JOIN rag.chunks c ON c.id = ce.chunk_id WHERE c.collection = 'imds'"
            )
            done = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM rag.chunks WHERE collection = 'imds'")
            total = cur.fetchone()[0]
        return done, total
    finally:
        conn.close()


def wait_for_kg():
    """Block until KG extraction is ≥98% complete."""
    log.info("Waiting for KG extraction to complete…")
    while True:
        done, total = _kg_progress()
        pct = done / total if total else 0
        remaining_s = max(0, (total - done) * 7)
        log.info(
            "KG progress: %d/%d (%.1f%%) — ETA ~%dm",
            done, total, pct * 100, remaining_s // 60,
        )
        if pct >= DONE_THRESHOLD:
            log.info("KG extraction complete (%d/%d chunks).", done, total)
            return
        time.sleep(CHECK_INTERVAL)


def run_stage(stage: str) -> bool:
    """Run `python ultra_ingest.py imds --stages <stage>` and return success."""
    log.info("=" * 60)
    log.info("Starting stage: %s", stage.upper())
    log.info("=" * 60)
    result = subprocess.run(
        [sys.executable, "ultra_ingest.py", "imds", "--stages", stage],
        cwd="D:/rag-ingest",
    )
    if result.returncode == 0:
        log.info("Stage %s completed successfully.", stage)
        return True
    else:
        log.error("Stage %s FAILED (exit code %d).", stage, result.returncode)
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-wait", action="store_true",
                        help="Skip KG completion check")
    args = parser.parse_args()

    if not args.skip_wait:
        wait_for_kg()

    for stage in ("communities", "contextual", "raptor"):
        ok = run_stage(stage)
        if not ok:
            log.error("Pipeline halted at stage '%s'.", stage)
            sys.exit(1)

    log.info("All remaining stages complete! Ultra RAG pipeline is fully built.")
    log.info("Dashboard: http://localhost:8300")


if __name__ == "__main__":
    main()
