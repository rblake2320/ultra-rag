"""
KG extractor: extracts entities and relationships from chunks using an LLM.
LightRAG-style incremental merge with deduplication.

Workflow per chunk:
  1. Send chunk content to LLM → get {entities, relationships} JSON
  2. Upsert entities into rag.entities (ON CONFLICT update description)
  3. Embed all new/updated entities via Ollama nomic-embed-text
  4. Upsert relationships into rag.relationships
  5. Link chunk → entity via rag.chunk_entities

Batch entry point:
    extractor = KGExtractor(conn, "imds")
    stats = extractor.process_collection()

Post-processing:
    extractor.build_synonymy_edges(threshold=0.80)
    extractor.update_specificity()
"""
import json
import logging
import time
from typing import Optional

import numpy as np
import psycopg2
import psycopg2.extras

from .config import get_config
from .embedder import _embed_batch
from .llm import LLMClient

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------

_EXTRACT_SYSTEM = (
    "You are a precise knowledge-graph extraction engine. "
    "Output only valid JSON — no prose, no markdown fences."
)

_EXTRACT_PROMPT_TMPL = """\
Extract all entities and relationships from the text below.

Return a JSON object with exactly two keys:
  "entities"      — list of objects with keys: name (str), type (str), description (str)
  "relationships" — list of objects with keys: source (str), target (str), type (str), context (str)

Guidelines:
- Focus on domain-specific concepts: components, procedures, identifiers, codes, screens, roles.
- Entity names should be short canonical labels (e.g. "Work Order", "Authentication Module", "API Rate Limit").
- Relationship types should be concise verb phrases (e.g. "REQUIRES", "REFERENCES", "BELONGS_TO").
- Only extract relationships whose source AND target appear in the entities list.
- If there is nothing to extract, return {{"entities": [], "relationships": []}}.

{heading_context}
TEXT:
{content}
"""


def _build_prompt(content: str, heading_path: Optional[list]) -> str:
    if heading_path:
        heading_context = "Document section path: " + " > ".join(heading_path) + "\n"
    else:
        heading_context = ""
    return _EXTRACT_PROMPT_TMPL.format(
        heading_context=heading_context,
        content=content[:4000],  # hard cap to stay within context window
    )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class KGExtractor:
    """
    Incremental LightRAG-style entity/relationship extractor.

    Parameters
    ----------
    conn        : psycopg2 connection (must already have Ultra RAG schema).
    collection  : Collection name (e.g. "imds").
    llm_client  : Optional pre-constructed LLMClient; one is created if None.
    """

    def __init__(
        self,
        conn,
        collection: str,
        llm_client: Optional[LLMClient] = None,
    ) -> None:
        self.conn       = conn
        self.collection = collection
        self.llm        = llm_client or LLMClient()

        cfg             = get_config()
        emb_cfg         = cfg.get("embedding", {})
        self._emb_url   = emb_cfg.get("ollama_url", "http://localhost:11434")
        self._emb_model = emb_cfg.get("model", "nomic-embed-text")
        self._emb_batch = emb_cfg.get("batch_size", 32)

    # -----------------------------------------------------------------------
    # Single-chunk extraction
    # -----------------------------------------------------------------------

    def extract_chunk(
        self,
        chunk_id: int,
        content: str,
        heading_path: Optional[list] = None,
    ) -> dict:
        """
        Ask the LLM to extract entities and relationships from *content*.

        Returns
        -------
        dict with keys "entities" and "relationships" (both lists),
        or {"entities": [], "relationships": []} on any failure.
        """
        prompt = _build_prompt(content, heading_path)
        result = self.llm.complete_json(prompt, system=_EXTRACT_SYSTEM)

        # Normalise: ensure both keys exist as lists
        entities      = result.get("entities", [])
        relationships = result.get("relationships", [])

        if not isinstance(entities, list):
            entities = []
        if not isinstance(relationships, list):
            relationships = []

        # Filter out malformed items
        clean_entities = []
        for e in entities:
            if isinstance(e, dict) and e.get("name"):
                clean_entities.append({
                    "name":        str(e["name"]).strip(),
                    "type":        str(e.get("type", "concept")).strip(),
                    "description": str(e.get("description", "")).strip(),
                })

        clean_rels = []
        for r in relationships:
            if isinstance(r, dict) and r.get("source") and r.get("target"):
                clean_rels.append({
                    "source":  str(r["source"]).strip(),
                    "target":  str(r["target"]).strip(),
                    "type":    str(r.get("type", "RELATED_TO")).strip(),
                    "context": str(r.get("context", "")).strip(),
                })

        return {"entities": clean_entities, "relationships": clean_rels}

    # -----------------------------------------------------------------------
    # DB upsert helpers
    # -----------------------------------------------------------------------

    def upsert_entity(
        self,
        name: str,
        entity_type: str,
        description: str,
    ) -> int:
        """
        Insert or update an entity in rag.entities.

        On conflict (collection, name) the description is updated only when
        the new description is longer/more informative than the stored one.

        Returns
        -------
        entity_id (int)
        """
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO rag.entities (collection, name, entity_type, description)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (collection, name) DO UPDATE
                    SET entity_type   = EXCLUDED.entity_type,
                        description   = CASE
                            WHEN length(EXCLUDED.description) > length(rag.entities.description)
                            THEN EXCLUDED.description
                            ELSE rag.entities.description
                        END
                RETURNING id
            """, (self.collection, name, entity_type, description))
            return cur.fetchone()[0]

    def upsert_relationship(
        self,
        source_id: int,
        target_id: int,
        rel_type: str,
        context: str,
    ) -> None:
        """
        Insert a directed relationship edge; silently ignores duplicates
        (same source, target, and rel_type).
        """
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO rag.relationships
                    (collection, source_id, target_id, rel_type, context)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
            """, (self.collection, source_id, target_id, rel_type, context))

    def link_chunk_entity(
        self,
        chunk_id: int,
        entity_id: int,
        mention: str,
    ) -> None:
        """Record that *entity_id* was mentioned in *chunk_id*."""
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO rag.chunk_entities (chunk_id, entity_id, mention_text)
                VALUES (%s, %s, %s)
                ON CONFLICT (chunk_id, entity_id) DO NOTHING
            """, (chunk_id, entity_id, mention))

    # -----------------------------------------------------------------------
    # Entity embedding
    # -----------------------------------------------------------------------

    def _embed_entities(self, entity_ids: list) -> None:
        """
        Embed a list of entities (by id) and write results back to rag.entities.
        Uses batched Ollama calls, same as embedder.py.
        """
        if not entity_ids:
            return

        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT id, name, description
                FROM rag.entities
                WHERE id = ANY(%s) AND embedding IS NULL
            """, (entity_ids,))
            rows = cur.fetchall()

        if not rows:
            return

        texts = [
            f"{r[1]}: {r[2]}" if r[2] else r[1]
            for r in rows
        ]
        ids   = [r[0] for r in rows]

        for batch_start in range(0, len(texts), self._emb_batch):
            batch_texts = texts[batch_start: batch_start + self._emb_batch]
            batch_ids   = ids[batch_start: batch_start + self._emb_batch]
            try:
                embeddings = _embed_batch(batch_texts, self._emb_url, self._emb_model)
            except Exception as exc:
                log.warning("Entity embed batch failed at %d: %s", batch_start, exc)
                continue

            if len(embeddings) != len(batch_ids):
                log.warning(
                    "Entity embed: expected %d results, got %d",
                    len(batch_ids), len(embeddings),
                )
                continue

            with self.conn.cursor() as cur:
                psycopg2.extras.execute_batch(cur, """
                    UPDATE rag.entities
                    SET embedding = %s::vector
                    WHERE id = %s
                """, [(emb, eid) for emb, eid in zip(embeddings, batch_ids)])
            self.conn.commit()

    # -----------------------------------------------------------------------
    # Batch processing
    # -----------------------------------------------------------------------

    def process_collection(self, batch_size: int = 50) -> dict:
        """
        Process all chunks in the collection that do not yet have entity links.

        Fetches chunks via LEFT JOIN on rag.chunk_entities so only un-processed
        chunks are touched.  Commits after each batch.

        Returns
        -------
        dict: {chunks_processed, entities_created, relationships_created}
        """
        # Fetch IDs + content of un-processed chunks
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT c.id, c.content, c.heading_path
                FROM rag.chunks c
                LEFT JOIN rag.chunk_entities ce ON ce.chunk_id = c.id
                WHERE c.collection = %s
                  AND ce.chunk_id IS NULL
                ORDER BY c.id
            """, (self.collection,))
            pending = cur.fetchall()

        if not pending:
            log.info("No un-processed chunks for collection '%s'.", self.collection)
            return {"chunks_processed": 0, "entities_created": 0, "relationships_created": 0}

        log.info(
            "KGExtractor: processing %d chunks for collection '%s'.",
            len(pending), self.collection,
        )

        chunks_processed      = 0
        entities_created      = 0
        relationships_created = 0
        t0                    = time.time()

        for batch_start in range(0, len(pending), batch_size):
            batch = pending[batch_start: batch_start + batch_size]
            new_entity_ids: list[int] = []

            for chunk_id, content, heading_path in batch:
                try:
                    extracted = self.extract_chunk(chunk_id, content, heading_path)
                except Exception as exc:
                    log.warning("extract_chunk(%d) failed: %s", chunk_id, exc)
                    continue

                # Build a name → entity_id map for relationship linking
                name_to_id: dict[str, int] = {}

                for ent in extracted["entities"]:
                    try:
                        eid = self.upsert_entity(
                            ent["name"], ent["type"], ent["description"]
                        )
                        name_to_id[ent["name"]] = eid
                        new_entity_ids.append(eid)
                        entities_created += 1
                        self.link_chunk_entity(chunk_id, eid, ent["name"])
                    except Exception as exc:
                        log.warning("upsert_entity failed for '%s': %s", ent["name"], exc)
                        self.conn.rollback()
                        continue

                for rel in extracted["relationships"]:
                    src_id = name_to_id.get(rel["source"])
                    tgt_id = name_to_id.get(rel["target"])
                    if src_id is None or tgt_id is None:
                        # Try a DB lookup for entities from earlier chunks
                        if src_id is None:
                            src_id = self._lookup_entity_id(rel["source"])
                        if tgt_id is None:
                            tgt_id = self._lookup_entity_id(rel["target"])
                    if src_id and tgt_id:
                        try:
                            self.upsert_relationship(
                                src_id, tgt_id, rel["type"], rel["context"]
                            )
                            relationships_created += 1
                        except Exception as exc:
                            log.warning("upsert_relationship failed: %s", exc)
                            self.conn.rollback()
                            continue

                chunks_processed += 1

            # Commit after each batch, then embed the new entities
            try:
                self.conn.commit()
            except Exception as exc:
                log.error("Commit failed after batch at %d: %s", batch_start, exc)
                self.conn.rollback()

            self._embed_entities(list(set(new_entity_ids)))

            elapsed = time.time() - t0
            log.info(
                "  batch %d/%d — %d chunks | %d entities | %d rels | %.1fs",
                batch_start // batch_size + 1,
                (len(pending) + batch_size - 1) // batch_size,
                chunks_processed, entities_created, relationships_created, elapsed,
            )

        log.info(
            "KGExtractor done: %d chunks, %d entities, %d relationships.",
            chunks_processed, entities_created, relationships_created,
        )
        return {
            "chunks_processed":      chunks_processed,
            "entities_created":      entities_created,
            "relationships_created": relationships_created,
        }

    def _lookup_entity_id(self, name: str) -> Optional[int]:
        """Fast exact-match lookup of an entity by name in this collection."""
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM rag.entities WHERE collection=%s AND name=%s LIMIT 1",
                (self.collection, name),
            )
            row = cur.fetchone()
            return row[0] if row else None

    # -----------------------------------------------------------------------
    # Post-processing: synonymy edges
    # -----------------------------------------------------------------------

    def build_synonymy_edges(self, threshold: float = 0.80) -> int:
        """
        Detect near-duplicate entities by embedding cosine similarity and
        record them in rag.synonymy_edges.

        Uses numpy for efficient pairwise computation in blocks of 1 000 to
        avoid OOM on large collections.

        Returns
        -------
        Number of new edges inserted.
        """
        # Load all embedded entities for this collection
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT id, embedding::text
                FROM rag.entities
                WHERE collection = %s
                  AND embedding IS NOT NULL
                ORDER BY id
            """, (self.collection,))
            rows = cur.fetchall()

        if len(rows) < 2:
            log.info("build_synonymy_edges: fewer than 2 embedded entities — skipping.")
            return 0

        entity_ids = [r[0] for r in rows]
        # Parse pgvector text representation "[f1,f2,...]"
        matrix = np.array(
            [json.loads(r[1]) for r in rows],
            dtype=np.float32,
        )

        # L2-normalise rows
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        matrix = matrix / norms

        n         = len(entity_ids)
        block     = 1_000
        new_edges = 0

        for i in range(0, n, block):
            block_vecs = matrix[i: i + block]  # (B, D)
            # cosine similarity with ALL entities: (B, n)
            sims = block_vecs @ matrix.T

            for bi, global_i in enumerate(range(i, min(i + block, n))):
                for global_j in range(global_i + 1, n):
                    sim = float(sims[bi, global_j])
                    if sim >= threshold:
                        eid_a = entity_ids[global_i]
                        eid_b = entity_ids[global_j]
                        try:
                            with self.conn.cursor() as cur:
                                cur.execute("""
                                    INSERT INTO rag.synonymy_edges
                                        (entity_a_id, entity_b_id, cosine_sim)
                                    VALUES (%s, %s, %s)
                                    ON CONFLICT (entity_a_id, entity_b_id) DO UPDATE
                                        SET cosine_sim = EXCLUDED.cosine_sim
                                """, (eid_a, eid_b, sim))
                            new_edges += 1
                        except Exception as exc:
                            log.warning(
                                "synonymy_edges insert (%d,%d) failed: %s",
                                eid_a, eid_b, exc,
                            )
                            self.conn.rollback()

        try:
            self.conn.commit()
        except Exception as exc:
            log.error("build_synonymy_edges commit failed: %s", exc)
            self.conn.rollback()

        log.info(
            "build_synonymy_edges: %d new edges (threshold=%.2f) for collection '%s'.",
            new_edges, threshold, self.collection,
        )
        return new_edges

    # -----------------------------------------------------------------------
    # Post-processing: specificity
    # -----------------------------------------------------------------------

    def update_specificity(self) -> int:
        """
        Recompute specificity = 1.0 / max(1, degree) for every entity in the
        collection, where degree = number of incident relationship edges.

        Returns
        -------
        Number of entities updated.
        """
        with self.conn.cursor() as cur:
            cur.execute("""
                WITH degree_cte AS (
                    SELECT e.id,
                           COUNT(r.id) AS degree
                    FROM rag.entities e
                    LEFT JOIN rag.relationships r
                        ON (r.source_id = e.id OR r.target_id = e.id)
                       AND r.collection = e.collection
                    WHERE e.collection = %s
                    GROUP BY e.id
                )
                UPDATE rag.entities e
                SET specificity = 1.0 / GREATEST(1, d.degree)
                FROM degree_cte d
                WHERE e.id = d.id
                RETURNING e.id
            """, (self.collection,))
            updated = cur.rowcount

        self.conn.commit()
        log.info(
            "update_specificity: updated %d entities for collection '%s'.",
            updated, self.collection,
        )
        return updated
