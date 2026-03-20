#!/usr/bin/env python3
"""
RAG Ingest CLI — ingest a collection into pgvector.

Usage:
  python rag_ingest.py imds             # ingest from config.yaml
  python rag_ingest.py imds --embed     # ingest + embed
  python rag_ingest.py imds --watch     # ingest + watch for changes
  python rag_ingest.py imds --embed --watch
"""
import argparse
import hashlib
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Logging ──────────────────────────────────────────────────────────────
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

# ── Add project root to path ─────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from src.config  import get_config
from src.db      import create_schema, doc_exists, upsert_document, insert_chunks, get_conn
from src.parsers import PARSERS
from src.chunker import chunk_blocks

# ── IMDS metadata extraction (mirrors rag_forge_run.py) ──────────────────
SCREEN_PATS = [
    re.compile(r"\bScreen\s+(\d{3,4})\b", re.I),
    re.compile(r"\bSCN\s*(\d{3,4})\b", re.I),
    re.compile(r"\b(\d{3})\s+Screen\b", re.I),
]
TRIC_PAT    = re.compile(r"\bTRIC[:\s]+([A-Z]{2,4})\b", re.I)
PROGRAM_IDS = {"NFS4F0","NFSPC0","NFSF10","IFMX","RTLX","DFSX","GUSX","CAMS","SBSS","CDRL"}
PROG_PAT    = re.compile(r"\b(" + "|".join(re.escape(p) for p in PROGRAM_IDS) + r")\b")
WUC_PAT     = re.compile(r"\bWUC[:\s]+([A-Z0-9]{3,})\b", re.I)
JCN_PAT     = re.compile(r"\b\d{4}[A-Z]\d{5,}\b")


def _imds_meta(text: str, inherited_screens: list) -> dict:
    meta = {}
    screens = set(inherited_screens)
    for pat in SCREEN_PATS:
        for m in pat.finditer(text):
            screens.add(m.group(1))
    if screens:
        meta["imds_screens"] = sorted(screens)
    trics = list(set(m.upper() for m in TRIC_PAT.findall(text)))
    if trics:
        meta["tric_codes"] = sorted(trics)
    progs = list(set(PROG_PAT.findall(text)))
    if progs:
        meta["programs"] = sorted(progs)
    wucs = list(set(WUC_PAT.findall(text)))
    if wucs:
        meta["wuc_codes"] = sorted(wucs)
    jcns = list(set(JCN_PAT.findall(text)))
    if jcns:
        meta["jcn_references"] = jcns[:10]
    return meta


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def ingest_file(path: Path, conn, collection: str) -> dict:
    """
    Ingest a single file into the database.
    Returns summary dict: {file, chunks, tokens, skipped, error}
    """
    ext      = path.suffix.lower()
    parser   = PARSERS.get(ext)
    if not parser:
        return {"file": path.name, "chunks": 0, "tokens": 0, "error": f"No parser for {ext}"}

    file_hash = _sha256(path)

    # Skip if already ingested with same hash
    existing = doc_exists(conn, collection, file_hash)
    if existing:
        log.info(f"  SKIP (unchanged): {path.name}")
        return {"file": path.name, "chunks": 0, "tokens": 0, "skipped": True}

    t0 = time.time()
    log.info(f"  Parsing: {path.name}")

    try:
        blocks = parser(path)
    except Exception as e:
        log.error(f"  Parse error {path.name}: {e}")
        return {"file": path.name, "chunks": 0, "tokens": 0, "error": str(e)}

    chunks = chunk_blocks(blocks)

    # Enrich with IMDS metadata — only for the 'imds' collection
    if collection == "imds":
        for ch in chunks:
            ch["imds_meta"] = _imds_meta(ch["text"], ch.get("inherited_screens", []))

    total_tokens = sum(ch.get("token_count", 0) for ch in chunks)

    # Write to DB — one transaction per file
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
    log.info(f"  OK {len(chunks)} chunks, {total_tokens:,} tokens ({elapsed:.1f}s)")
    return {"file": path.name, "chunks": len(chunks), "tokens": total_tokens}


def ingest_collection(collection: str, conn) -> dict:
    """Ingest all files for a collection. Returns summary."""
    cfg      = get_config()
    coll_cfg = cfg["collections"].get(collection)
    if not coll_cfg:
        raise ValueError(f"Collection '{collection}' not found in config.yaml")

    # paths may be plain strings OR dicts: {path: ..., max_depth: N}
    raw_paths  = coll_cfg.get("paths", [])
    excl_dirs  = set(coll_cfg.get("exclude_dirs", []))
    skip_files = set(coll_cfg.get("skip_files", []))

    def _parse_path_entry(entry):
        if isinstance(entry, dict):
            return Path(entry["path"]), entry.get("max_depth", None)
        return Path(str(entry)), None

    path_entries = [_parse_path_entry(e) for e in raw_paths]

    # Collect files using os.walk so excluded dirs are pruned before traversal
    import os
    files = []
    for base, max_depth in path_entries:
        if not base.exists():
            log.warning(f"Path not found: {base}")
            continue
        base_str = str(base)
        base_depth = base_str.rstrip("/\\").count(os.sep)
        for dirpath, dirnames, filenames in os.walk(base_str):
            # Prune excluded dirs IN PLACE (prevents os.walk from descending)
            dirnames[:] = [d for d in dirnames if d not in excl_dirs]
            # Enforce max_depth: stop descending beyond base + max_depth levels
            if max_depth is not None:
                current_depth = dirpath.rstrip("/\\").count(os.sep) - base_depth
                if current_depth >= max_depth:
                    dirnames[:] = []
            for fname in filenames:
                if fname in skip_files:
                    continue
                fpath = Path(dirpath) / fname
                if fpath.suffix.lower() in PARSERS:
                    files.append(fpath)

    log.info(f"Collection '{collection}': {len(files)} files found")

    summaries = []
    errors    = []
    for i, fpath in enumerate(files, 1):
        log.info(f"[{i}/{len(files)}] {fpath.name}")
        result = ingest_file(fpath, conn, collection)
        if result.get("error"):
            errors.append(result)
        else:
            summaries.append(result)

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


def main():
    parser = argparse.ArgumentParser(description="RAG Ingest CLI")
    parser.add_argument("collection", help="Collection name from config.yaml")
    parser.add_argument("--embed",  action="store_true", help="Embed after ingest")
    parser.add_argument("--watch",  action="store_true", help="Watch for changes after ingest")
    parser.add_argument("--force",  action="store_true", help="Re-ingest even if unchanged")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info(f"RAG Ingest — collection: {args.collection}")
    log.info("=" * 60)

    conn = get_conn()
    try:
        create_schema(conn)
        summary = ingest_collection(args.collection, conn)

        log.info("=" * 60)
        log.info("INGEST COMPLETE")
        log.info(f"  Collection: {summary['collection']}")
        log.info(f"  Files:      {summary['files']}")
        log.info(f"  Ingested:   {summary['ingested']}")
        log.info(f"  Skipped:    {summary['skipped']} (unchanged)")
        log.info(f"  Errors:     {summary['errors']}")
        log.info(f"  Chunks:     {summary['chunks']:,}")
        log.info(f"  Tokens:     {summary['tokens']:,}")

        if args.embed and summary["chunks"] > 0:
            log.info("\nStarting embedding...")
            from src.embedder import embed_collection
            n = embed_collection(conn, args.collection)
            log.info(f"Embedded {n} chunks")

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
