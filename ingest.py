#!/usr/bin/env python3
"""
Ultra RAG Ingest CLI — parse, chunk, and embed documents into pgvector.

Usage:
  python ingest.py default                    # ingest default collection
  python ingest.py default --embed            # ingest + embed
  python ingest.py default --watch            # ingest + watch for changes
  python ingest.py default --embed --watch    # ingest + embed + watch
  python ingest.py default --path /my/docs    # override path from config
  python ingest.py default --force            # re-ingest unchanged files

For the full 7-stage pipeline (KG, RAPTOR, communities, etc.) use:
  python ultra_ingest.py default --stages all
"""
import argparse
import hashlib
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Logging ──────────────────────────────────────────────────────────────────
logs_dir = Path(__file__).parent / "logs"
logs_dir.mkdir(exist_ok=True)
log_file = logs_dir / f"ingest_{datetime.now():%Y%m%d}.log"

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── Project root on sys.path ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from src.config  import get_config
from src.db      import create_schema, doc_exists, upsert_document, insert_chunks, get_conn
from src.parsers import PARSERS
from src.chunker import chunk_blocks


# ── Helpers ──────────────────────────────────────────────────────────────────

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


# ── Core ingest functions ─────────────────────────────────────────────────────

def ingest_file(path: Path, conn, collection: str, force: bool = False) -> dict:
    """
    Ingest a single file into the database.

    Skips files whose SHA-256 hash is already in the DB (unless --force).
    Returns a summary dict: {file, chunks, tokens, skipped, error}.
    """
    ext    = path.suffix.lower()
    parser = PARSERS.get(ext)
    if not parser:
        return {"file": path.name, "chunks": 0, "tokens": 0,
                "error": f"No parser for {ext}"}

    file_hash = _sha256(path)

    if not force and doc_exists(conn, collection, file_hash):
        log.info("  SKIP (unchanged): %s", path.name)
        return {"file": path.name, "chunks": 0, "tokens": 0, "skipped": True}

    t0 = time.time()
    log.info("  Parsing: %s", path.name)

    try:
        blocks = parser(path)
    except Exception as e:
        log.error("  Parse error %s: %s", path.name, e)
        return {"file": path.name, "chunks": 0, "tokens": 0, "error": str(e)}

    chunks = chunk_blocks(blocks)
    total_tokens = sum(ch.get("token_count", 0) for ch in chunks)

    doc_id = upsert_document(
        conn=conn,
        collection=collection,
        file_path=str(path),
        file_name=path.name,
        file_type=ext.lstrip("."),
        file_hash=file_hash,
        file_size=path.stat().st_size,
        title=path.stem,
        chunk_count=len(chunks),
        doc_metadata={},
    )
    insert_chunks(conn, doc_id, collection, chunks, str(path))
    conn.commit()

    elapsed = time.time() - t0
    log.info("  OK  %d chunks  %s tokens  (%.1fs)",
             len(chunks), f"{total_tokens:,}", elapsed)
    return {"file": path.name, "chunks": len(chunks), "tokens": total_tokens}


def ingest_collection(
    collection: str,
    conn,
    path_override: str | None = None,
    force: bool = False,
) -> dict:
    """
    Ingest all supported files for a collection.

    Parameters
    ----------
    collection:     Name of the collection in config.yaml.
    conn:           Active psycopg2 connection.
    path_override:  If set, overrides the paths list in config.yaml.
    force:          Re-ingest even if the file hash is unchanged.

    Returns a summary dict.
    """
    cfg      = get_config()
    coll_cfg = cfg["collections"].get(collection)
    if not coll_cfg:
        raise ValueError(
            f"Collection '{collection}' not found in config.yaml. "
            f"Available: {list(cfg['collections'].keys())}"
        )

    if path_override:
        paths = [Path(path_override)]
    else:
        paths = [Path(p) for p in coll_cfg.get("paths", [])]

    excl_dirs  = set(coll_cfg.get("exclude_dirs", []))
    skip_files = set(coll_cfg.get("skip_files",   []))

    # Collect supported files
    files = []
    for base in paths:
        if not base.exists():
            log.warning("Path not found: %s", base)
            continue
        for fpath in sorted(base.rglob("*")):
            if not fpath.is_file():
                continue
            rel = fpath.relative_to(base)
            if any(d in excl_dirs for d in rel.parts):
                continue
            if fpath.name in skip_files:
                continue
            if fpath.suffix.lower() in PARSERS:
                files.append(fpath)

    log.info("Collection '%s': %d files found", collection, len(files))

    summaries, errors = [], []
    for i, fpath in enumerate(files, 1):
        log.info("[%d/%d] %s", i, len(files), fpath.name)
        result = ingest_file(fpath, conn, collection, force=force)
        (errors if result.get("error") else summaries).append(result)

    total_chunks = sum(s.get("chunks", 0) for s in summaries)
    total_tokens = sum(s.get("tokens", 0) for s in summaries)
    skipped      = sum(1 for s in summaries if s.get("skipped"))

    return {
        "collection": collection,
        "files":      len(files),
        "ingested":   len(summaries) - skipped,
        "skipped":    skipped,
        "errors":     len(errors),
        "chunks":     total_chunks,
        "tokens":     total_tokens,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ultra RAG Ingest — parse and chunk documents into pgvector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "collection",
        help="Collection name from config.yaml (e.g. 'default')",
    )
    parser.add_argument(
        "--embed",
        action="store_true",
        help="Embed chunks after ingest (requires Ollama running)",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Watch the collection directories for changes after ingest",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest files even if their hash is unchanged",
    )
    parser.add_argument(
        "--path",
        default=None,
        metavar="DIR",
        help="Override the paths list in config.yaml with a single directory",
    )
    args = parser.parse_args()

    log.info("=" * 65)
    log.info("Ultra RAG Ingest — collection: %s", args.collection)
    log.info("=" * 65)

    conn = get_conn()
    try:
        create_schema(conn)
        summary = ingest_collection(
            args.collection, conn,
            path_override=args.path,
            force=args.force,
        )

        log.info("=" * 65)
        log.info("INGEST COMPLETE")
        log.info("  Collection : %s",   summary["collection"])
        log.info("  Files      : %d",   summary["files"])
        log.info("  Ingested   : %d",   summary["ingested"])
        log.info("  Skipped    : %d  (unchanged)", summary["skipped"])
        log.info("  Errors     : %d",   summary["errors"])
        log.info("  Chunks     : %s",   f"{summary['chunks']:,}")
        log.info("  Tokens     : %s",   f"{summary['tokens']:,}")

        if args.embed and summary["chunks"] > 0:
            log.info("\nStarting embedding...")
            from src.embedder import embed_collection
            n = embed_collection(conn, args.collection)
            log.info("Embedded %d chunks", n)

        if args.watch:
            log.info("\nStarting folder watcher...")
            from src.watcher import watch

            def _ingest_fn(path, conn, coll):
                ingest_file(path, conn, coll)
                if args.embed:
                    from src.embedder import embed_collection
                    embed_collection(conn, coll)

            watch(args.collection, conn, _ingest_fn)

    except KeyboardInterrupt:
        log.info("\nInterrupted by user")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
