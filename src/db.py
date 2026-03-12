"""
PostgreSQL connection and schema management for rag schema.
Uses psycopg2 directly (no ORM overhead for bulk inserts).
"""
import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

import psycopg2
import psycopg2.extras

from .config import get_config

log = logging.getLogger(__name__)


def _dsn() -> str:
    cfg = get_config()["db"]
    return (
        f"host={cfg['host']} port={cfg['port']} "
        f"dbname={cfg['database']} "
        f"user={cfg['user']} password={cfg['password']}"
    )


def get_conn():
    """Return a new psycopg2 connection."""
    return psycopg2.connect(_dsn())


def create_schema(conn) -> None:
    """Create rag schema and tables if they don't exist."""
    with conn.cursor() as cur:
        cur.execute("CREATE SCHEMA IF NOT EXISTS rag;")
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS rag.documents (
                id              BIGSERIAL PRIMARY KEY,
                collection      VARCHAR(100) NOT NULL DEFAULT 'default',
                file_path       TEXT NOT NULL,
                file_name       TEXT NOT NULL,
                file_type       VARCHAR(20) NOT NULL,
                file_hash       VARCHAR(64) NOT NULL,
                file_size       BIGINT,
                title           TEXT,
                chunk_count     INTEGER DEFAULT 0,
                doc_metadata    JSONB DEFAULT '{}',
                ingested_at     TIMESTAMPTZ DEFAULT now(),
                UNIQUE(collection, file_hash)
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS rag.chunks (
                id              BIGSERIAL PRIMARY KEY,
                document_id     BIGINT REFERENCES rag.documents(id) ON DELETE CASCADE,
                collection      VARCHAR(100) NOT NULL,
                chunk_index     INTEGER NOT NULL,
                content         TEXT NOT NULL,
                content_hash    VARCHAR(64) NOT NULL,
                context_prefix  TEXT,
                content_type    VARCHAR(30) NOT NULL DEFAULT 'text',
                heading_path    TEXT[],
                page_number     INTEGER,
                token_count     INTEGER,
                chunk_metadata  JSONB DEFAULT '{}',
                stable_id       VARCHAR(128),
                embedding       vector(768),
                embed_model     VARCHAR(100),
                content_tsv     TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
                UNIQUE(document_id, content_hash)
            );
        """)
        # Indexes
        cur.execute("CREATE INDEX IF NOT EXISTS rag_chunks_collection ON rag.chunks(collection);")
        cur.execute("CREATE INDEX IF NOT EXISTS rag_chunks_ctype ON rag.chunks(content_type);")
        cur.execute("CREATE INDEX IF NOT EXISTS rag_chunks_stable ON rag.chunks(stable_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS rag_chunks_tsv ON rag.chunks USING gin(content_tsv);")
        cur.execute("CREATE INDEX IF NOT EXISTS rag_chunks_meta ON rag.chunks USING gin(chunk_metadata jsonb_path_ops);")
        # HNSW index only if pgvector is ready
        try:
            cur.execute("""
                CREATE INDEX IF NOT EXISTS rag_chunks_emb
                ON rag.chunks USING hnsw(embedding vector_cosine_ops)
                WITH (m=16, ef_construction=200);
            """)
        except Exception as e:
            log.warning(f"Could not create HNSW index: {e}")
        conn.commit()
    log.info("Schema ready: rag.documents + rag.chunks")


def doc_exists(conn, collection: str, file_hash: str) -> Optional[int]:
    """Return document id if this file_hash already exists, else None."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM rag.documents WHERE collection=%s AND file_hash=%s",
            (collection, file_hash)
        )
        row = cur.fetchone()
        return row[0] if row else None


def upsert_document(conn, collection: str, file_path: str, file_name: str,
                    file_type: str, file_hash: str, file_size: int,
                    title: Optional[str], chunk_count: int,
                    doc_metadata: dict) -> int:
    """Insert document record, return id. Deletes existing chunks on re-ingest."""
    with conn.cursor() as cur:
        # Check if exists
        cur.execute(
            "SELECT id FROM rag.documents WHERE collection=%s AND file_hash=%s",
            (collection, file_hash)
        )
        row = cur.fetchone()
        if row:
            doc_id = row[0]
            cur.execute("DELETE FROM rag.chunks WHERE document_id=%s", (doc_id,))
            cur.execute(
                "UPDATE rag.documents SET chunk_count=%s, ingested_at=now() WHERE id=%s",
                (chunk_count, doc_id)
            )
        else:
            cur.execute("""
                INSERT INTO rag.documents
                    (collection, file_path, file_name, file_type, file_hash,
                     file_size, title, chunk_count, doc_metadata)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                RETURNING id
            """, (collection, file_path, file_name, file_type, file_hash,
                  file_size, title, chunk_count, json.dumps(doc_metadata)))
            doc_id = cur.fetchone()[0]
    return doc_id


def insert_chunks(conn, doc_id: int, collection: str, chunks: list,
                  file_path: str) -> None:
    """Bulk insert chunks for a document."""
    with conn.cursor() as cur:
        rows = []
        for idx, ch in enumerate(chunks):
            content      = ch["text"]
            content_hash = hashlib.sha256(content.encode()).hexdigest()[:64]
            stable_id    = hashlib.sha256(
                f"{collection}:{file_path}:{idx}:{content_hash}".encode()
            ).hexdigest()[:64]

            extra_meta   = ch.get("extra_meta", {}) or ch.get("imds_meta", {})
            chunk_meta   = {
                k: v for k, v in extra_meta.items()
            }
            chunk_meta["source_section"] = ch.get("source_section", "")

            rows.append((
                doc_id,
                collection,
                idx,
                content,
                content_hash,
                ch.get("context_prefix"),
                ch.get("content_type", "text"),
                ch.get("heading_path") or [],
                ch.get("source_page"),
                ch.get("token_count"),
                json.dumps(chunk_meta),
                stable_id,
            ))

        psycopg2.extras.execute_values(cur, """
            INSERT INTO rag.chunks
                (document_id, collection, chunk_index, content, content_hash,
                 context_prefix, content_type, heading_path, page_number,
                 token_count, chunk_metadata, stable_id)
            VALUES %s
            ON CONFLICT (document_id, content_hash) DO NOTHING
        """, rows)
