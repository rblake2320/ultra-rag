"""
Parent chunker: creates large parent chunks that contain multiple child chunks.
Parent chunks provide richer generation context after retrieval.

Strategy
--------
* Children are grouped by the first two levels of their heading_path so that
  semantically related chunks (same section) stay together.
* Groups are merged in order until the *target_tokens* budget is exhausted,
  then a new parent is started.
* Every child chunk receives exactly one parent via rag.chunk_parents.
* Parent embeddings are optionally computed via Ollama so parents can also be
  searched directly (useful for long-context re-ranking).
"""
import logging
import time
from typing import Optional

import psycopg2.extras

from .config import get_config
from .embedder import _embed_batch

log = logging.getLogger(__name__)

_DEFAULT_TARGET_TOKENS = 2500


# ---------------------------------------------------------------------------
# Token counter
# ---------------------------------------------------------------------------

def _count_tokens(text: str) -> int:
    """
    Approximate token count for *text*.

    Tries ``tiktoken`` (cl100k_base) first; falls back to word-count × 1.33
    so the module works without tiktoken installed.
    """
    try:
        import tiktoken  # noqa: PLC0415
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return int(len(text.split()) * 1.33)


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class ParentChunker:
    """
    Build a two-tier chunk hierarchy: small retrieval chunks → large parent chunks.

    Usage::

        conn = get_conn()
        pc   = ParentChunker(conn, "my-docs")
        stats = pc.process_collection()
        print(stats)
    """

    def __init__(
        self,
        conn,
        collection: str,
        target_tokens: int = _DEFAULT_TARGET_TOKENS,
    ):
        self.conn          = conn
        self.collection    = collection
        self.target_tokens = target_tokens
        cfg                = get_config()["embedding"]
        self._emb_model    = cfg["model"]
        self._emb_url      = cfg["ollama_url"]
        self._emb_batch    = cfg.get("batch_size", 32)

    # ------------------------------------------------------------------
    # Heading key
    # ------------------------------------------------------------------

    @staticmethod
    def _heading_key(heading_path) -> tuple:
        """Return the first two heading levels as a normalised tuple."""
        if not heading_path:
            return ("",)
        levels = [str(h).strip() for h in heading_path if h]
        return tuple(levels[:2]) if levels else ("",)

    # ------------------------------------------------------------------
    # Process a single document
    # ------------------------------------------------------------------

    def process_document(self, doc_id: int) -> int:
        """
        Create parent chunks for all child chunks belonging to *doc_id*.

        Returns
        -------
        int
            Number of parent chunks inserted.
        """
        # ── Fetch ordered children ───────────────────────────────────────
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT id, content, heading_path, token_count, chunk_index
                FROM   rag.chunks
                WHERE  document_id = %s
                  AND  collection  = %s
                ORDER  BY chunk_index
            """, (doc_id, self.collection))
            children = cur.fetchall()

        if not children:
            return 0

        # ── Group by heading_key ─────────────────────────────────────────
        # We preserve document order within groups by iterating once.
        # Each element: (chunk_id, content, heading_path, token_count)
        groups: dict[tuple, list] = {}
        order:  list[tuple]       = []  # keeps insertion order of group keys

        for chunk_id, content, heading_path, token_count, _ in children:
            key = self._heading_key(heading_path)
            if key not in groups:
                groups[key] = []
                order.append(key)
            tc = token_count or _count_tokens(content)
            groups[key].append((chunk_id, content, heading_path, tc))

        # ── Merge children into parents ──────────────────────────────────
        parents_created = 0
        all_parent_texts: list[str]              = []
        all_parent_meta:  list[tuple]            = []
        # (document_id, collection, heading_path_for_parent, token_count)
        child_map: dict[int, list[int]]          = {}
        # parent_index → [chunk_ids]

        parent_idx = 0
        for key in order:
            group = groups[key]
            current_texts:    list[str] = []
            current_ids:      list[int] = []
            current_tokens:   int       = 0
            # The heading_path for the parent = heading_path of first child
            current_heading:  Optional[list] = None

            for chunk_id, content, heading_path, tc in group:
                if (
                    current_texts
                    and current_tokens + tc > self.target_tokens
                ):
                    # Seal current parent
                    merged = "\n\n".join(current_texts)
                    all_parent_texts.append(merged)
                    all_parent_meta.append(
                        (doc_id, self.collection,
                         current_heading or [], current_tokens)
                    )
                    child_map[parent_idx] = list(current_ids)
                    parent_idx += 1

                    # Start new parent
                    current_texts   = []
                    current_ids     = []
                    current_tokens  = 0
                    current_heading = None

                if current_heading is None and heading_path:
                    current_heading = list(heading_path)
                current_texts.append(content)
                current_ids.append(chunk_id)
                current_tokens += tc

            # Seal last group
            if current_texts:
                merged = "\n\n".join(current_texts)
                all_parent_texts.append(merged)
                all_parent_meta.append(
                    (doc_id, self.collection,
                     current_heading or [], current_tokens)
                )
                child_map[parent_idx] = list(current_ids)
                parent_idx += 1

        if not all_parent_texts:
            return 0

        # ── Embed parent texts in batches ────────────────────────────────
        embeddings: list = [None] * len(all_parent_texts)
        batch_sz = self._emb_batch
        for start in range(0, len(all_parent_texts), batch_sz):
            batch_texts = all_parent_texts[start: start + batch_sz]
            try:
                embs = _embed_batch(batch_texts, self._emb_url, self._emb_model)
                for i, emb in enumerate(embs):
                    embeddings[start + i] = emb
            except Exception as exc:
                log.warning(
                    "Parent embedding batch failed (doc %d, offset %d): %s",
                    doc_id, start, exc,
                )

        # ── Insert parents + mapping ─────────────────────────────────────
        with self.conn.cursor() as cur:
            for idx, (text, meta, emb) in enumerate(
                zip(all_parent_texts, all_parent_meta, embeddings)
            ):
                doc_id_p, coll, heading, token_count = meta
                cur.execute("""
                    INSERT INTO rag.parent_chunks
                        (document_id, collection, content, heading_path,
                         token_count, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s::vector)
                    RETURNING id
                """, (
                    doc_id_p, coll, text,
                    heading, token_count,
                    emb if emb else None,
                ))
                parent_db_id = cur.fetchone()[0]
                parents_created += 1

                # Map each child → this parent (upsert)
                for chunk_id in child_map[idx]:
                    cur.execute("""
                        INSERT INTO rag.chunk_parents (chunk_id, parent_id)
                        VALUES (%s, %s)
                        ON CONFLICT (chunk_id) DO UPDATE
                            SET parent_id = EXCLUDED.parent_id
                    """, (chunk_id, parent_db_id))

        self.conn.commit()
        return parents_created

    # ------------------------------------------------------------------
    # Process entire collection
    # ------------------------------------------------------------------

    def process_collection(self) -> dict:
        """
        Process all documents that do not yet have parent chunks.

        Returns
        -------
        dict
            ``{"documents_processed": int, "parent_chunks_created": int}``
        """
        # Documents that have chunks but no parent_chunks yet
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT c.document_id
                FROM   rag.chunks c
                WHERE  c.collection = %s
                  AND  NOT EXISTS (
                      SELECT 1 FROM rag.parent_chunks p
                      WHERE  p.document_id = c.document_id
                        AND  p.collection  = c.collection
                  )
            """, (self.collection,))
            doc_ids = [r[0] for r in cur.fetchall()]

        if not doc_ids:
            log.info(
                "All documents in '%s' already have parent chunks.", self.collection
            )
            return {"documents_processed": 0, "parent_chunks_created": 0}

        log.info(
            "Building parent chunks for %d documents in '%s'…",
            len(doc_ids), self.collection,
        )
        t0 = time.time()
        total_docs     = 0
        total_parents  = 0

        for doc_id in doc_ids:
            try:
                n = self.process_document(doc_id)
                total_docs    += 1
                total_parents += n
            except Exception as exc:
                self.conn.rollback()
                log.error("Error processing doc %d: %s", doc_id, exc)

        elapsed = time.time() - t0
        log.info(
            "Parent chunking complete: %d docs, %d parents in %.1fs",
            total_docs, total_parents, elapsed,
        )
        return {
            "documents_processed": total_docs,
            "parent_chunks_created": total_parents,
        }

    # ------------------------------------------------------------------
    # Accessor
    # ------------------------------------------------------------------

    def get_parent_content(self, chunk_id: int) -> str:
        """
        Return the content of the parent chunk for *chunk_id*, or ``""`` if none.
        """
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT pc.content
                FROM   rag.parent_chunks pc
                JOIN   rag.chunk_parents cp ON cp.parent_id = pc.id
                WHERE  cp.chunk_id = %s
            """, (chunk_id,))
            row = cur.fetchone()
        return row[0] if row else ""
