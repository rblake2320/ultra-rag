#!/usr/bin/env python3
"""
Migration: upgrade embedding columns from vector(768) to vector(4096).

Run ONCE before switching embedding.model to nvidia/NV-Embed-v2 (4096-dim).

What this does:
  1. ALTERs all vector(768) columns to vector(4096) in rag schema.
  2. Drops and recreates HNSW indexes at new dimensions.
  3. NULLs out all existing embeddings (forces full re-embed run).

After running this script:
    python ultra_ingest.py imds --stages embed
    python ultra_ingest.py personal --stages embed   # if personal collection exists

Tables affected:
    rag.chunks, rag.entities, rag.contextual_contexts,
    rag.communities, rag.summaries, rag.parent_chunks

Usage:
    cd D:\\rag-ingest
    python src/migrate_embedding_dim.py              # dry-run (shows SQL, no execute)
    python src/migrate_embedding_dim.py --apply      # execute migration
    python src/migrate_embedding_dim.py --rollback   # revert to 768 (for emergency)
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
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tables and their vector columns + HNSW index names
# ---------------------------------------------------------------------------

# (table, column, old_dim, new_dim, hnsw_index_name)
_TARGETS = [
    ("rag.chunks",               "embedding",      768, 4096, "idx_chunks_embedding"),
    ("rag.entities",             "embedding",      768, 4096, "idx_entities_embedding"),
    ("rag.contextual_contexts",  "embedding",      768, 4096, "idx_contextual_embedding"),
    ("rag.communities",          "embedding",      768, 4096, "idx_communities_embedding"),
    ("rag.summaries",            "embedding",      768, 4096, "idx_summaries_embedding"),
    ("rag.parent_chunks",        "embedding",      768, 4096, "idx_parent_chunks_embedding"),
]

# ef_construction / m values for HNSW rebuild
_HNSW_EF_CONSTRUCTION = 128
_HNSW_M               = 16


def _table_exists(conn, table: str) -> bool:
    schema, tname = (table.split(".", 1) + [""])[:2]
    with conn.cursor() as cur:
        cur.execute("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = %s AND table_name = %s
            )
        """, (schema, tname))
        return cur.fetchone()[0]


def _column_exists(conn, table: str, column: str) -> bool:
    schema, tname = (table.split(".", 1) + [""])[:2]
    with conn.cursor() as cur:
        cur.execute("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_schema = %s AND table_name = %s AND column_name = %s
            )
        """, (schema, tname, column))
        return cur.fetchone()[0]


def _index_exists(conn, index_name: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT EXISTS (
                SELECT 1 FROM pg_indexes
                WHERE indexname = %s
            )
        """, (index_name,))
        return cur.fetchone()[0]


def _run_migration(conn, new_dim: int, dry_run: bool) -> None:
    """Upgrade all vector columns to new_dim and rebuild HNSW indexes."""
    log.info("=" * 65)
    log.info("Embedding dimension migration: 768 → %d", new_dim)
    log.info("Dry run: %s", dry_run)
    log.info("=" * 65)

    for table, col, old_dim, _, idx_name in _TARGETS:
        if not _table_exists(conn, table):
            log.info("  SKIP  %s — table not found", table)
            continue
        if not _column_exists(conn, table, col):
            log.info("  SKIP  %s.%s — column not found", table, col)
            continue

        log.info("Processing %s.%s ...", table, col)

        # 1. Drop existing HNSW index (can't alter in-place)
        if _index_exists(conn, idx_name):
            sql_drop = f"DROP INDEX IF EXISTS {idx_name};"
            log.info("  DROP INDEX %s", idx_name)
            if not dry_run:
                with conn.cursor() as cur:
                    cur.execute(sql_drop)
                conn.commit()
        else:
            log.info("  (index %s not found, skipping drop)", idx_name)

        # 2. ALTER column type
        sql_alter = (
            f"ALTER TABLE {table} "
            f"ALTER COLUMN {col} TYPE vector({new_dim}) "
            f"USING NULL::vector({new_dim});"
        )
        log.info("  ALTER COLUMN → vector(%d)", new_dim)
        if not dry_run:
            with conn.cursor() as cur:
                cur.execute(sql_alter)
            conn.commit()

        # 3. NULL out existing embeddings (old vectors are incompatible)
        sql_null = f"UPDATE {table} SET {col} = NULL;"
        log.info("  NULL OUT existing embeddings")
        if not dry_run:
            with conn.cursor() as cur:
                cur.execute(sql_null)
            conn.commit()

        # 4. Rebuild HNSW index
        sql_idx = (
            f"CREATE INDEX {idx_name} ON {table} "
            f"USING hnsw ({col} vector_cosine_ops) "
            f"WITH (m = {_HNSW_M}, ef_construction = {_HNSW_EF_CONSTRUCTION});"
        )
        log.info("  CREATE HNSW INDEX %s", idx_name)
        if not dry_run:
            with conn.cursor() as cur:
                cur.execute(sql_idx)
            conn.commit()

        log.info("  ✓ %s.%s done", table, col)

    log.info("")
    if dry_run:
        log.info("DRY RUN complete — no changes made. Add --apply to execute.")
    else:
        log.info("Migration complete.")
        log.info("")
        log.info("Next steps:")
        log.info("  1. Ensure config.yaml embedding.model = nvidia/NV-Embed-v2")
        log.info("  2. Ensure config.yaml embedding.dimensions = 4096")
        log.info("  3. Ensure config.yaml embedding.provider = local")
        log.info("  4. python ultra_ingest.py imds --stages embed")
        log.info("  5. python ultra_ingest.py personal --stages embed  (if applicable)")


def _run_rollback(conn, old_dim: int, dry_run: bool) -> None:
    """Emergency rollback: revert all vector columns to old_dim (default 768)."""
    log.warning("ROLLBACK: reverting to vector(%d)", old_dim)
    for table, col, _odim, _, idx_name in _TARGETS:
        if not _table_exists(conn, table):
            continue
        if not _column_exists(conn, table, col):
            continue

        log.info("Rolling back %s.%s ...", table, col)

        if _index_exists(conn, idx_name):
            sql = f"DROP INDEX IF EXISTS {idx_name};"
            if not dry_run:
                with conn.cursor() as cur:
                    cur.execute(sql)
                conn.commit()

        sql_alter = (
            f"ALTER TABLE {table} "
            f"ALTER COLUMN {col} TYPE vector({old_dim}) "
            f"USING NULL::vector({old_dim});"
        )
        if not dry_run:
            with conn.cursor() as cur:
                cur.execute(sql_alter)
            conn.commit()

        sql_null = f"UPDATE {table} SET {col} = NULL;"
        if not dry_run:
            with conn.cursor() as cur:
                cur.execute(sql_null)
            conn.commit()

        sql_idx = (
            f"CREATE INDEX {idx_name} ON {table} "
            f"USING hnsw ({col} vector_cosine_ops) "
            f"WITH (m = {_HNSW_M}, ef_construction = {_HNSW_EF_CONSTRUCTION});"
        )
        if not dry_run:
            with conn.cursor() as cur:
                cur.execute(sql_idx)
            conn.commit()

        log.info("  ✓ %s.%s rolled back to vector(%d)", table, col, old_dim)

    if dry_run:
        log.info("ROLLBACK dry run complete.")
    else:
        log.info("ROLLBACK complete. Re-embed with original model after reverting config.yaml.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Migrate embedding vector columns 768→4096 for NV-Embed-v2",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually execute the migration (default: dry run)",
    )
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="Rollback migration: revert to vector(768)",
    )
    parser.add_argument(
        "--new-dim",
        type=int,
        default=4096,
        help="Target vector dimension (default: 4096 for NV-Embed-v2)",
    )
    parser.add_argument(
        "--old-dim",
        type=int,
        default=768,
        help="Original vector dimension (default: 768 for nomic-embed-text)",
    )
    args = parser.parse_args()

    dry_run = not args.apply

    if args.rollback and args.apply:
        print("Cannot use --rollback and --apply together.")
        return 1

    from src.db import get_conn
    conn = get_conn()

    try:
        if args.rollback:
            _run_rollback(conn, args.old_dim, dry_run=dry_run)
        else:
            _run_migration(conn, args.new_dim, dry_run=dry_run)
        return 0
    except Exception as exc:
        log.error("Migration error: %s", exc, exc_info=True)
        try:
            conn.rollback()
        except Exception:
            pass
        return 1
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
