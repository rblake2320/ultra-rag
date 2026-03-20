#!/usr/bin/env python3
"""
Bridge Worker — drains rag.bridge_queue into MemoryWeb.

Polls rag.bridge_queue for pending rows, writes each as a JSONL session file,
POSTs the path to MemoryWeb's /api/ingest/session, marks the row done,
and sets memoryweb_ingested=true on the linked query_log row.

Run: python bridge_worker.py [--once] [--interval 60]
  --once      Process all pending rows once then exit (for scheduled task mode)
  --interval  Seconds between polls in daemon mode (default: 60)
"""
import argparse
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
import psycopg2
import psycopg2.extras

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s bridge_worker — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path(__file__).parent / "logs" / "bridge_worker.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

DB_DSN = "host=localhost port=5432 dbname=postgres user=postgres password=?Booker78!"
MEMORYWEB_URL = os.environ.get("MEMORYWEB_URL", "http://localhost:8100")
SESSION_DIR = Path(__file__).parent / "data" / "mw-sessions"
MAX_ATTEMPTS = 3


def get_conn():
    return psycopg2.connect(DB_DSN)


def process_pending(conn) -> int:
    """Process all pending bridge_queue rows. Returns count processed."""
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    processed = 0

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        # Claim rows atomically — skip locked so parallel workers don't double-process
        cur.execute("""
            SELECT id, query_text, collection, summary, strategy, quality_score, query_log_id
            FROM rag.bridge_queue
            WHERE status = 'pending' AND attempts < %s
            ORDER BY created_at
            LIMIT 20
            FOR UPDATE SKIP LOCKED
        """, (MAX_ATTEMPTS,))
        rows = cur.fetchall()

        for row in rows:
            row_id = row["id"]
            collection = row["collection"]
            query_text = row["query_text"]
            summary = row["summary"]

            # Mark as processing
            cur.execute(
                "UPDATE rag.bridge_queue SET status='processing', attempts=attempts+1 WHERE id=%s",
                (row_id,)
            )
            conn.commit()

            try:
                # Write JSONL session file
                ts = datetime.now(timezone.utc).isoformat()
                fname = f"rag_{collection}_{uuid.uuid4().hex[:8]}.jsonl"
                fpath = SESSION_DIR / fname
                lines = [
                    json.dumps({"type": "user", "content": query_text.replace("\x00", ""), "timestamp": ts}),
                    json.dumps({"type": "assistant", "content": f"[Ultra RAG {collection}] {summary.replace(chr(0), '')}", "timestamp": ts}),
                ]
                fpath.write_text("\n".join(lines) + "\n", encoding="utf-8")

                # POST to MemoryWeb
                with httpx.Client(timeout=15.0) as client:
                    resp = client.post(
                        f"{MEMORYWEB_URL}/api/ingest/session",
                        json={"path": str(fpath), "force": False},
                    )

                if resp.status_code == 200:
                    # Mark done
                    cur.execute(
                        "UPDATE rag.bridge_queue SET status='done', processed_at=now() WHERE id=%s",
                        (row_id,)
                    )
                    # Mark query_log row as ingested
                    if row["query_log_id"]:
                        cur.execute(
                            "UPDATE rag.query_log SET memoryweb_ingested=true WHERE id=%s",
                            (row["query_log_id"],)
                        )
                    conn.commit()
                    processed += 1
                    log.info("Queued %s → MemoryWeb (row %d, collection=%s)", fname, row_id, collection)
                else:
                    raise RuntimeError(f"MemoryWeb returned {resp.status_code}: {resp.text[:200]}")

            except Exception as exc:
                log.warning("Row %d failed (attempt %d): %s", row_id, row["attempts"] + 1, exc)
                cur.execute(
                    """UPDATE rag.bridge_queue
                       SET status=CASE WHEN attempts>=%s THEN 'failed' ELSE 'pending' END,
                           error_msg=%s
                       WHERE id=%s""",
                    (MAX_ATTEMPTS, str(exc)[:500], row_id)
                )
                conn.commit()

    return processed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Process once then exit")
    parser.add_argument("--interval", type=int, default=60, help="Poll interval seconds")
    args = parser.parse_args()

    log.info("Bridge worker starting (mode=%s, interval=%ds)", "once" if args.once else "daemon", args.interval)
    conn = get_conn()

    if args.once:
        n = process_pending(conn)
        log.info("Done: processed %d rows", n)
        conn.close()
        return

    while True:
        try:
            n = process_pending(conn)
            if n:
                log.info("Processed %d bridge rows", n)
        except Exception as exc:
            log.error("Worker loop error: %s", exc)
            try:
                conn.close()
            except Exception:
                pass
            conn = get_conn()
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
