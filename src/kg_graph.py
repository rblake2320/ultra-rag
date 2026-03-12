"""
KG graph operations:
- Personalized PageRank (PPR) for multi-hop retrieval
- Subgraph extraction for local context window building
- Specificity-weighted node scoring
- Cross-collection bridge detection

Typical usage:
    from src.kg_graph import KGGraph
    g = KGGraph(conn, "imds")

    seeds = g.get_seed_entities("Work Order status codes", query_embedding=emb)
    top   = g.ppr([s["id"] for s in seeds], top_k=20)
    chunks = g.get_entity_chunks([e["id"] for e in top])
"""
import json
import logging
from typing import Optional

import numpy as np
import psycopg2.extras

log = logging.getLogger(__name__)


class KGGraph:
    """
    In-memory graph helper built on top of rag.entities / rag.relationships.

    Parameters
    ----------
    conn       : psycopg2 connection with Ultra RAG schema present.
    collection : Collection name, e.g. "imds".
    """

    def __init__(self, conn, collection: str) -> None:
        self.conn       = conn
        self.collection = collection

    # -----------------------------------------------------------------------
    # Seed entity discovery
    # -----------------------------------------------------------------------

    def get_seed_entities(
        self,
        query: str,
        top_k: int = 5,
        query_embedding: Optional[list] = None,
    ) -> list:
        """
        Find seed entities relevant to *query*.

        If *query_embedding* is provided, uses cosine similarity against
        rag.entities.embedding (requires pgvector).
        Otherwise falls back to ILIKE text match on name and description.

        Returns
        -------
        list of dicts: {id, name, entity_type, description, specificity}
        """
        if not query_embedding:
            return self._seed_by_text(query, top_k)

        # Vector path
        emb_str = "[" + ",".join(str(float(x)) for x in query_embedding) + "]"
        try:
            with self.conn.cursor(
                cursor_factory=psycopg2.extras.RealDictCursor
            ) as cur:
                cur.execute("""
                    SELECT id, name, entity_type, description, specificity,
                           1 - (embedding <=> %s::vector) AS similarity
                    FROM rag.entities
                    WHERE collection = %s
                      AND embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (emb_str, self.collection, emb_str, top_k))
                rows = cur.fetchall()
            return [dict(r) for r in rows]
        except Exception as exc:
            log.warning(
                "Vector seed lookup failed (%s); falling back to text search.", exc
            )
            return self._seed_by_text(query, top_k)

    def _seed_by_text(self, query: str, top_k: int) -> list:
        """ILIKE fallback seed discovery."""
        pattern = f"%{query}%"
        with self.conn.cursor(
            cursor_factory=psycopg2.extras.RealDictCursor
        ) as cur:
            cur.execute("""
                SELECT id, name, entity_type, description, specificity
                FROM rag.entities
                WHERE collection = %s
                  AND (name ILIKE %s OR description ILIKE %s)
                ORDER BY specificity DESC
                LIMIT %s
            """, (self.collection, pattern, pattern, top_k))
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    # -----------------------------------------------------------------------
    # Personalized PageRank
    # -----------------------------------------------------------------------

    def ppr(
        self,
        seed_entity_ids: list,
        alpha: float = 0.85,
        max_iter: int = 30,
        top_k: int = 20,
    ) -> list:
        """
        Run Personalized PageRank from *seed_entity_ids* over the entity graph.

        Uses scipy.sparse for the adjacency matrix.  The personalization vector
        is a uniform distribution over the seed nodes.
        Specificity boost is applied to the raw PPR scores before ranking.

        Parameters
        ----------
        seed_entity_ids : List of entity IDs to personalise toward.
        alpha           : Damping factor (0.85 is standard PageRank default).
        max_iter        : Maximum power-iteration steps.
        top_k           : Number of top entities to return.

        Returns
        -------
        list of dicts: {id, name, entity_type, description, specificity, score}
        Sorted descending by score.  Returns [] if the collection has no edges.
        """
        # ── Load adjacency ───────────────────────────────────────────────
        entity_idx, n, weights = self._load_adjacency()
        if n == 0 or not entity_idx:
            log.info("ppr: no entities in collection '%s'.", self.collection)
            return []

        if not seed_entity_ids:
            log.warning("ppr called with empty seed list.")
            return []

        # ── Build sparse row-stochastic matrix ──────────────────────────
        try:
            from scipy.sparse import csr_matrix
        except ImportError as exc:
            raise ImportError(
                "scipy is required for PPR. Install with: pip install scipy"
            ) from exc

        idx_map: dict[int, int] = {eid: i for i, eid in enumerate(entity_idx)}
        rows, cols, data = [], [], []

        for (src_eid, tgt_eid), w in weights.items():
            src_i = idx_map.get(src_eid)
            tgt_i = idx_map.get(tgt_eid)
            if src_i is None or tgt_i is None:
                continue
            # Undirected: add both directions
            rows.append(src_i); cols.append(tgt_i); data.append(w)
            rows.append(tgt_i); cols.append(src_i); data.append(w)

        if not rows:
            log.info("ppr: no edges found for collection '%s'.", self.collection)
            return []

        A = csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)

        # Row-normalise to make it a transition matrix
        row_sums = np.array(A.sum(axis=1)).flatten()
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        # Divide each row by its sum using sparse diagonal multiplication
        from scipy.sparse import diags
        D_inv = diags(1.0 / row_sums)
        A_norm = D_inv @ A  # row-stochastic

        # ── Personalization vector ───────────────────────────────────────
        p = np.zeros(n, dtype=np.float64)
        valid_seeds = [idx_map[eid] for eid in seed_entity_ids if eid in idx_map]
        if not valid_seeds:
            log.warning(
                "ppr: none of the seed IDs (%s) found in graph index.", seed_entity_ids
            )
            return []

        p[valid_seeds] = 1.0 / len(valid_seeds)

        # ── Power iteration ──────────────────────────────────────────────
        r = p.copy()
        for _ in range(max_iter):
            r_new = alpha * A_norm.T.dot(r) + (1.0 - alpha) * p
            if np.linalg.norm(r_new - r, 1) < 1e-6:
                break
            r = r_new

        # ── Specificity boost & top-k ────────────────────────────────────
        # Fetch specificity for all entities
        spec = self._load_specificities(entity_idx)

        scored = [
            (entity_idx[i], float(r[i]) * (1.0 + spec.get(entity_idx[i], 0.0)))
            for i in range(n)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        top_ids = [eid for eid, _ in scored[:top_k]]

        # Fetch entity details
        id_to_score = {eid: s for eid, s in scored[:top_k]}
        entities    = self._fetch_entities_by_ids(top_ids)
        for e in entities:
            e["score"] = round(id_to_score.get(e["id"], 0.0), 8)

        entities.sort(key=lambda x: x["score"], reverse=True)
        return entities

    # -----------------------------------------------------------------------
    # Entity-to-chunk mapping
    # -----------------------------------------------------------------------

    def get_entity_chunks(
        self,
        entity_ids: list,
        top_k: int = 10,
    ) -> list:
        """
        Return the most relevant chunks for a set of entity IDs.

        If rag.chunk_utility exists and has data, the utility EMA is used as a
        ranking boost.  Falls back to plain chunk ordering by chunk_index.

        Returns
        -------
        list of dicts with chunk fields + optional utility_ema.
        """
        if not entity_ids:
            return []

        with self.conn.cursor(
            cursor_factory=psycopg2.extras.RealDictCursor
        ) as cur:
            cur.execute("""
                SELECT DISTINCT ON (c.id)
                    c.id,
                    c.content,
                    c.content_type,
                    c.context_prefix,
                    c.heading_path,
                    c.chunk_metadata,
                    c.token_count,
                    COALESCE(cu.utility_ema, 0.5) AS utility_ema
                FROM rag.chunk_entities ce
                JOIN rag.chunks c ON c.id = ce.chunk_id
                LEFT JOIN rag.chunk_utility cu ON cu.chunk_id = c.id
                WHERE ce.entity_id = ANY(%s)
                  AND c.collection = %s
                ORDER BY c.id, cu.utility_ema DESC NULLS LAST
                LIMIT %s
            """, (entity_ids, self.collection, top_k * 3))
            rows = cur.fetchall()

        results = [dict(r) for r in rows]
        # Sort by utility then truncate
        results.sort(key=lambda r: r.get("utility_ema", 0.5), reverse=True)
        return results[:top_k]

    # -----------------------------------------------------------------------
    # Subgraph extraction (for visualization / context window building)
    # -----------------------------------------------------------------------

    def extract_subgraph(
        self,
        entity_ids: list,
        hops: int = 2,
    ) -> dict:
        """
        BFS-expand *entity_ids* up to *hops* hops over rag.relationships.

        Returns
        -------
        dict: {
            entities:      list of entity dicts,
            relationships: list of relationship dicts,
        }
        Both lists de-duplicated by id.
        """
        if not entity_ids:
            return {"entities": [], "relationships": []}

        visited_entities: set[int] = set(entity_ids)
        frontier: set[int]         = set(entity_ids)
        all_rels: list[dict]       = []

        for _ in range(hops):
            if not frontier:
                break
            with self.conn.cursor(
                cursor_factory=psycopg2.extras.RealDictCursor
            ) as cur:
                cur.execute("""
                    SELECT id, source_id, target_id, rel_type, weight, context
                    FROM rag.relationships
                    WHERE collection = %s
                      AND (source_id = ANY(%s) OR target_id = ANY(%s))
                """, (self.collection, list(frontier), list(frontier)))
                edges = cur.fetchall()

            next_frontier: set[int] = set()
            for e in edges:
                all_rels.append(dict(e))
                for nid in (e["source_id"], e["target_id"]):
                    if nid not in visited_entities:
                        next_frontier.add(nid)
                        visited_entities.add(nid)

            frontier = next_frontier

        # De-duplicate relationships by id
        seen_rel_ids: set[int] = set()
        unique_rels = []
        for r in all_rels:
            if r["id"] not in seen_rel_ids:
                seen_rel_ids.add(r["id"])
                unique_rels.append(r)

        entities = self._fetch_entities_by_ids(list(visited_entities))

        return {"entities": entities, "relationships": unique_rels}

    # -----------------------------------------------------------------------
    # Cross-collection bridges
    # -----------------------------------------------------------------------

    def find_cross_collection_bridges(
        self,
        entity_ids: list,
        other_collections: list,
        similarity_threshold: float = 0.75,
    ) -> list:
        """
        Find entities in *other_collections* whose embeddings are cosine-similar
        (>= *similarity_threshold*) to any of the given entities.

        Returns
        -------
        list of dicts: {
            local_entity_id, local_entity_name,
            remote_entity_id, remote_entity_name,
            remote_collection, cosine_similarity
        }
        """
        if not entity_ids or not other_collections:
            return []

        # Load source entity embeddings
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT id, name, embedding::text
                FROM rag.entities
                WHERE id = ANY(%s)
                  AND embedding IS NOT NULL
            """, (entity_ids,))
            src_rows = cur.fetchall()

        if not src_rows:
            log.info("find_cross_collection_bridges: no embedded source entities.")
            return []

        src_ids   = [r[0] for r in src_rows]
        src_names = [r[1] for r in src_rows]
        src_matrix = np.array(
            [json.loads(r[2]) for r in src_rows], dtype=np.float32
        )
        src_norms = np.linalg.norm(src_matrix, axis=1, keepdims=True)
        src_norms = np.where(src_norms == 0, 1.0, src_norms)
        src_matrix = src_matrix / src_norms

        # Load target entity embeddings
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT id, name, collection, embedding::text
                FROM rag.entities
                WHERE collection = ANY(%s)
                  AND embedding IS NOT NULL
            """, (other_collections,))
            tgt_rows = cur.fetchall()

        if not tgt_rows:
            return []

        tgt_ids   = [r[0] for r in tgt_rows]
        tgt_names = [r[1] for r in tgt_rows]
        tgt_colls = [r[2] for r in tgt_rows]
        tgt_matrix = np.array(
            [json.loads(r[3]) for r in tgt_rows], dtype=np.float32
        )
        tgt_norms = np.linalg.norm(tgt_matrix, axis=1, keepdims=True)
        tgt_norms = np.where(tgt_norms == 0, 1.0, tgt_norms)
        tgt_matrix = tgt_matrix / tgt_norms

        # Compute all pairwise cosines: (n_src, n_tgt)
        sims = src_matrix @ tgt_matrix.T

        bridges = []
        n_src, n_tgt = sims.shape
        for i in range(n_src):
            for j in range(n_tgt):
                sim = float(sims[i, j])
                if sim >= similarity_threshold:
                    bridges.append({
                        "local_entity_id":    src_ids[i],
                        "local_entity_name":  src_names[i],
                        "remote_entity_id":   tgt_ids[j],
                        "remote_entity_name": tgt_names[j],
                        "remote_collection":  tgt_colls[j],
                        "cosine_similarity":  round(sim, 6),
                    })

        bridges.sort(key=lambda b: b["cosine_similarity"], reverse=True)
        return bridges

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _load_adjacency(self) -> tuple:
        """
        Load all relationship edges for this collection.

        Returns
        -------
        (entity_idx, n, weights) where:
            entity_idx : list of entity IDs in graph order
            n          : number of nodes
            weights    : dict {(src_id, tgt_id): weight}
        """
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT source_id, target_id, weight
                FROM rag.relationships
                WHERE collection = %s
            """, (self.collection,))
            edges = cur.fetchall()

        if not edges:
            return [], 0, {}

        node_set: set[int] = set()
        weights: dict      = {}
        for src, tgt, w in edges:
            node_set.add(src)
            node_set.add(tgt)
            key = (src, tgt)
            # Accumulate parallel edges
            weights[key] = weights.get(key, 0.0) + float(w if w else 1.0)

        entity_idx = sorted(node_set)
        return entity_idx, len(entity_idx), weights

    def _load_specificities(self, entity_ids: list) -> dict:
        """Return {entity_id: specificity} for a list of IDs."""
        if not entity_ids:
            return {}
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT id, specificity
                FROM rag.entities
                WHERE id = ANY(%s)
            """, (entity_ids,))
            return {r[0]: float(r[1]) for r in cur.fetchall()}

    def _fetch_entities_by_ids(self, entity_ids: list) -> list:
        """Fetch full entity rows for a list of IDs as list of dicts."""
        if not entity_ids:
            return []
        with self.conn.cursor(
            cursor_factory=psycopg2.extras.RealDictCursor
        ) as cur:
            cur.execute("""
                SELECT id, name, entity_type, description, specificity
                FROM rag.entities
                WHERE id = ANY(%s)
                ORDER BY id
            """, (entity_ids,))
            return [dict(r) for r in cur.fetchall()]
