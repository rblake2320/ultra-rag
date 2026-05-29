#!/usr/bin/env python3
"""
HotpotQA distractor-set benchmark loader for Ultra RAG.

Downloads a public subset of HotpotQA (CC BY-SA 4.0) — NOT committed to the
repo — ingests its passages into a local ``benchmark`` collection, and loads
the gold question/answer pairs into rag.eval_questions so the standard
``ultra_eval.py benchmark --run`` path works unchanged.

Usage
-----
# First-time setup (ingest + load Q&A):
    python scripts/benchmark_dataset.py --setup

# Re-run eval only (questions already loaded):
    python scripts/benchmark_dataset.py --eval

# Full pipeline (setup then eval):
    python scripts/benchmark_dataset.py --setup --eval

# Limit to N QA pairs for a quick smoke-test:
    python scripts/benchmark_dataset.py --setup --n 50 --eval

Options
-------
--setup     Ingest passages and load Q&A pairs (idempotent).
--eval      Run ultra_eval.py for the benchmark collection and print results.
--n INT     Max Q&A pairs to load (default: 200). Passages always fully loaded.
--run-name  Label for the eval run (default: auto timestamped).
--force     Re-ingest even if the benchmark collection already has chunks.
"""
import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

COLLECTION = "benchmark"
DATASET_NAME = "hotpot_qa"
DATASET_CONFIG = "distractor"
DATASET_SPLIT = "validation"   # validation is public; test has hidden answers
MAX_PASSAGES_PER_DOC = 10      # HotpotQA context paragraphs per question


def _load_dataset(n_qa: int):
    try:
        from datasets import load_dataset  # noqa: PLC0415
    except ImportError:
        log.error("Install 'datasets': pip install datasets")
        sys.exit(1)

    log.info("Downloading HotpotQA distractor validation set (CC BY-SA 4.0)…")
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT, trust_remote_code=False)
    log.info("Downloaded %d examples.", len(ds))
    return ds.select(range(min(n_qa, len(ds))))


def _ingest_passages(subset, conn, force: bool):
    """Write unique passage paragraphs into rag.chunks under the benchmark collection."""
    from src.db import get_conn  # noqa: PLC0415 (already imported but kept local for clarity)
    from src.embedder import get_embedder  # noqa: PLC0415

    with conn.cursor() as cur:
        cur.execute("select count(*) from rag.chunks where collection=%s", (COLLECTION,))
        existing = cur.fetchone()[0]

    if existing and not force:
        log.info("benchmark collection already has %d chunks — skipping ingest (use --force to re-ingest).", existing)
        return

    embedder = get_embedder()

    # Collect unique passages (title, paragraph text)
    seen: set[str] = set()
    passages: list[tuple[str, str]] = []
    for ex in subset:
        for title, sents in zip(ex["context"]["title"], ex["context"]["sentences"]):
            text = " ".join(sents).strip()
            key = f"{title}::{text[:80]}"
            if key not in seen and text:
                seen.add(key)
                passages.append((title, text))

    log.info("Ingesting %d unique passages into collection '%s'…", len(passages), COLLECTION)

    BATCH = 32
    inserted = 0
    with conn.cursor() as cur:
        # Ensure a synthetic document row exists
        cur.execute("""
            INSERT INTO rag.documents
                (collection, file_path, file_name, file_type, file_hash, title)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (collection, file_hash) DO NOTHING
            RETURNING id
        """, (COLLECTION, "hotpotqa://distractor/validation", "hotpotqa_distractor",
              "text", "hotpotqa_distractor_v1", "HotpotQA Distractor Validation"))
        row = cur.fetchone()
        if row:
            doc_id = row[0]
        else:
            cur.execute("select id from rag.documents where collection=%s and file_hash=%s",
                        (COLLECTION, "hotpotqa_distractor_v1"))
            doc_id = cur.fetchone()[0]
        conn.commit()

        for batch_start in range(0, len(passages), BATCH):
            batch = passages[batch_start: batch_start + BATCH]
            texts = [f"{title}\n\n{text}" for title, text in batch]
            embeddings = embedder(texts)
            for idx, ((title, text), emb) in enumerate(zip(batch, embeddings)):
                cur.execute("""
                    INSERT INTO rag.chunks
                        (document_id, collection, chunk_index, content,
                         chunk_metadata, embedding)
                    VALUES (%s, %s, %s, %s, %s::jsonb, %s::vector)
                    ON CONFLICT DO NOTHING
                """, (doc_id, COLLECTION, batch_start + idx,
                      text,
                      json.dumps({"title": title, "source": "hotpotqa_distractor"}),
                      f"[{','.join(str(x) for x in emb)}]"))
            conn.commit()
            inserted += len(batch)
            log.info("  %d / %d passages embedded and stored.", inserted, len(passages))

    log.info("Ingest complete: %d passages in '%s'.", inserted, COLLECTION)


def _load_qa_pairs(subset, conn, n: int):
    """Insert gold Q&A pairs into rag.eval_questions for ultra_eval.py --run."""
    with conn.cursor() as cur:
        cur.execute("select count(*) from rag.eval_questions where collection=%s", (COLLECTION,))
        existing = cur.fetchone()[0]
        if existing:
            log.info("eval_questions already has %d rows for '%s' — skipping.", existing, COLLECTION)
            return

        rows = []
        for ex in list(subset)[:n]:
            question = ex["question"].strip()
            answer = ex["answer"].strip()
            if question and answer:
                rows.append((COLLECTION, question, "extractive", answer))

        cur.executemany("""
            INSERT INTO rag.eval_questions
                (collection, question_text, question_type, ground_truth)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT DO NOTHING
        """, rows)
        conn.commit()
    log.info("Loaded %d Q&A pairs into rag.eval_questions for collection '%s'.", len(rows), COLLECTION)


def _run_eval(run_name: str):
    cmd = [
        sys.executable, str(PROJECT_ROOT / "ultra_eval.py"),
        COLLECTION, "--run", "--run-name", run_name,
    ]
    log.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        log.error("ultra_eval.py exited with code %d", result.returncode)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--setup", action="store_true", help="Ingest passages + load Q&A pairs")
    parser.add_argument("--eval",  action="store_true", help="Run eval and print results")
    parser.add_argument("--n",     type=int, default=200, help="Max Q&A pairs (default 200)")
    parser.add_argument("--run-name", default="", help="Eval run label (auto if omitted)")
    parser.add_argument("--force", action="store_true", help="Re-ingest even if collection exists")
    args = parser.parse_args()

    if not args.setup and not args.eval:
        parser.print_help()
        sys.exit(0)

    from src.db import get_conn  # noqa: PLC0415
    conn = get_conn()

    if args.setup:
        subset = _load_dataset(args.n)
        _ingest_passages(subset, conn, force=args.force)
        _load_qa_pairs(subset, conn, args.n)

    if args.eval:
        import datetime  # noqa: PLC0415
        run_name = args.run_name or f"hotpotqa-{datetime.datetime.now().strftime('%Y%m%d-%H%M')}"
        sys.exit(_run_eval(run_name))


if __name__ == "__main__":
    main()
