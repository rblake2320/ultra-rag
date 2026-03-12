"""
Community detection: Leiden algorithm on entity graph → hierarchical communities.
Generates LLM summaries per community for global thematic search.

The pipeline is:
  1. Load rag.entities + rag.relationships for the collection.
  2. Build an undirected igraph.Graph (one vertex per entity, weighted edges).
  3. Run Leiden at three resolution levels (0.5, 1.0, 2.0) → coarse → fine.
  4. Persist communities to rag.communities + rag.community_members.
  5. For each community, ask the LLM for a 2-3 sentence thematic summary,
     embed it, and store back into rag.communities.summary / summary_embedding.

Dependencies: ``igraph``, ``leidenalg`` — both are pip-installable.
If either is missing the module degrades gracefully with a WARNING.
"""
import logging
import time
from typing import Optional

import psycopg2.extras

from .config import get_config
from .embedder import _embed_batch

log = logging.getLogger(__name__)

# Leiden resolution levels to run (low → coarse communities, high → fine)
_RESOLUTIONS = [0.5, 1.0, 2.0]

# How many top entities to fetch per community for summarisation
_TOP_ENTITIES_PER_COMMUNITY = 10

# How many sample relationships to include in the summarisation prompt
_SAMPLE_RELS_PER_COMMUNITY  = 15


def _load_llm_client():
    from .llm import LLMClient   # noqa: PLC0415
    from .config import get_config  # noqa: PLC0415
    _fast = get_config().get("llm", {}).get("fast_model", "qwen2.5:7b")
    return LLMClient(model=_fast)


# ---------------------------------------------------------------------------
# Availability guards
# ---------------------------------------------------------------------------

def _import_igraph():
    try:
        import igraph  # noqa: PLC0415
        return igraph
    except ImportError:
        log.warning(
            "igraph not installed — community detection disabled. "
            "Install with: pip install igraph leidenalg"
        )
        return None


def _import_leidenalg():
    try:
        import leidenalg  # noqa: PLC0415
        return leidenalg
    except ImportError:
        log.warning(
            "leidenalg not installed — community detection disabled. "
            "Install with: pip install leidenalg"
        )
        return None


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class CommunityDetector:
    """
    Detect knowledge communities in the entity graph and generate LLM summaries.

    Usage::

        conn = get_conn()
        cd   = CommunityDetector(conn, "imds")
        stats = cd.process_collection()
        print(stats)
    """

    def __init__(self, conn, collection: str, llm_client=None):
        self.conn       = conn
        self.collection = collection
        self._llm       = llm_client
        cfg             = get_config()["embedding"]
        self._emb_model = cfg["model"]
        self._emb_url   = cfg["ollama_url"]

    # ------------------------------------------------------------------
    # LLM accessor
    # ------------------------------------------------------------------

    @property
    def llm(self):
        if self._llm is None:
            self._llm = _load_llm_client()
        return self._llm

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def build_igraph(self):
        """
        Load entities + relationships for the collection and build an
        undirected weighted igraph.Graph.

        Returns
        -------
        igraph.Graph | None
            ``None`` if igraph is unavailable or fewer than 3 entities exist.
        """
        ig = _import_igraph()
        if ig is None:
            return None

        with self.conn.cursor() as cur:
            # Entities
            cur.execute("""
                SELECT id, name, specificity
                FROM   rag.entities
                WHERE  collection = %s
                ORDER  BY id
            """, (self.collection,))
            entity_rows = cur.fetchall()

            if len(entity_rows) < 3:
                log.info(
                    "Only %d entities in '%s' — skipping community detection.",
                    len(entity_rows), self.collection,
                )
                return None

            # Relationships
            cur.execute("""
                SELECT source_id, target_id, weight
                FROM   rag.relationships
                WHERE  collection = %s
            """, (self.collection,))
            rel_rows = cur.fetchall()

        # igraph requires 0-based integer vertex IDs.
        # Build a mapping: entity DB id → igraph vertex index.
        entity_ids   = [r[0] for r in entity_rows]
        id_to_vertex = {eid: vi for vi, eid in enumerate(entity_ids)}

        # Vertex attributes
        n = len(entity_ids)
        g = ig.Graph(n=n, directed=False)
        g.vs["db_id"]      = entity_ids
        g.vs["name"]       = [r[1] for r in entity_rows]
        g.vs["specificity"] = [r[2] or 0.0 for r in entity_rows]

        # Edges — skip if either endpoint not in the vertex set
        edge_list  = []
        weights    = []
        seen_edges: set[tuple[int, int]] = set()

        for src_id, tgt_id, weight in rel_rows:
            vi_src = id_to_vertex.get(src_id)
            vi_tgt = id_to_vertex.get(tgt_id)
            if vi_src is None or vi_tgt is None or vi_src == vi_tgt:
                continue
            key = (min(vi_src, vi_tgt), max(vi_src, vi_tgt))
            if key in seen_edges:
                continue
            seen_edges.add(key)
            edge_list.append(key)
            weights.append(float(weight or 1.0))

        if edge_list:
            g.add_edges(edge_list)
            g.es["weight"] = weights

        log.info(
            "Built igraph: %d vertices, %d edges for collection '%s'",
            g.vcount(), g.ecount(), self.collection,
        )
        return g

    # ------------------------------------------------------------------
    # Community detection
    # ------------------------------------------------------------------

    def detect_communities(self, n_levels: int = 3) -> dict:
        """
        Run Leiden at multiple resolutions and persist the results.

        Parameters
        ----------
        n_levels:
            Number of resolution levels (capped at ``len(_RESOLUTIONS)``).

        Returns
        -------
        dict
            ``{level: [community_db_ids…]}``
        """
        la = _import_leidenalg()
        if la is None:
            return {}

        g = self.build_igraph()
        if g is None:
            return {}

        resolutions = _RESOLUTIONS[: max(1, min(n_levels, len(_RESOLUTIONS)))]
        result: dict[int, list[int]] = {}

        for level, resolution in enumerate(resolutions):
            log.info(
                "Leiden level %d (resolution=%.1f)…", level, resolution
            )
            try:
                partition = la.find_partition(
                    g,
                    la.RBConfigurationVertexPartition,
                    weights="weight" if g.ecount() > 0 else None,
                    resolution_parameter=resolution,
                    seed=42,
                )
            except Exception as exc:
                log.warning("Leiden failed at level %d: %s", level, exc)
                continue

            community_ids = []
            with self.conn.cursor() as cur:
                for comm_idx, member_vertices in enumerate(partition):
                    db_ids = [g.vs[vi]["db_id"] for vi in member_vertices]
                    entity_count = len(db_ids)

                    cur.execute("""
                        INSERT INTO rag.communities
                            (collection, level, title, entity_count)
                        VALUES (%s, %s, %s, %s)
                        RETURNING id
                    """, (
                        self.collection,
                        level,
                        f"Community {comm_idx}",
                        entity_count,
                    ))
                    comm_db_id = cur.fetchone()[0]
                    community_ids.append(comm_db_id)

                    # Insert members
                    psycopg2.extras.execute_batch(cur, """
                        INSERT INTO rag.community_members
                            (community_id, entity_id)
                        VALUES (%s, %s)
                        ON CONFLICT DO NOTHING
                    """, [(comm_db_id, eid) for eid in db_ids])

            self.conn.commit()
            result[level] = community_ids
            log.info(
                "  Level %d: %d communities detected.",
                level, len(community_ids),
            )

        return result

    # ------------------------------------------------------------------
    # Community summarisation
    # ------------------------------------------------------------------

    def summarize_community(self, community_id: int) -> str:
        """
        Generate and persist an LLM summary for *community_id*.

        Steps:
          1. Fetch top-10 entities by specificity.
          2. Fetch up to 15 relationships among them.
          3. Call LLM for a 2-3 sentence thematic summary.
          4. Update ``rag.communities.summary``, ``title``, and
             ``summary_embedding``.

        Returns
        -------
        str
            The generated summary (empty string on error).
        """
        with self.conn.cursor() as cur:
            # Top entities in this community
            cur.execute("""
                SELECT e.id, e.name, e.entity_type, e.specificity
                FROM   rag.community_members cm
                JOIN   rag.entities          e  ON e.id = cm.entity_id
                WHERE  cm.community_id = %s
                ORDER  BY e.specificity DESC
                LIMIT  %s
            """, (community_id, _TOP_ENTITIES_PER_COMMUNITY))
            entity_rows = cur.fetchall()

        if not entity_rows:
            log.debug("Community %d has no entities.", community_id)
            return ""

        entity_ids  = [r[0] for r in entity_rows]
        entity_names = [r[1] for r in entity_rows]

        # Sample relationships among top entities
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT e1.name, r.rel_type, e2.name, r.context
                FROM   rag.relationships r
                JOIN   rag.entities      e1 ON e1.id = r.source_id
                JOIN   rag.entities      e2 ON e2.id = r.target_id
                WHERE  r.source_id = ANY(%s)
                  AND  r.target_id = ANY(%s)
                LIMIT  %s
            """, (entity_ids, entity_ids, _SAMPLE_RELS_PER_COMMUNITY))
            rel_rows = cur.fetchall()

        # Build prompt
        rel_lines = [
            f"  • {src} –[{rtype}]→ {tgt}"
            + (f" ({ctx[:80]})" if ctx else "")
            for src, rtype, tgt, ctx in rel_rows
        ] or ["  (no direct relationships found)"]

        prompt = (
            "Summarize this knowledge cluster. "
            f"Entities: {', '.join(entity_names)}.\n"
            f"Key relationships:\n" + "\n".join(rel_lines) + "\n\n"
            "Write a 2-3 sentence thematic summary that describes what these "
            "entities collectively represent and how they relate."
        )
        system = (
            "You are a knowledge graph analyst. Output only the summary "
            "— no headings, no bullets, no preamble."
        )

        try:
            summary = self.llm.complete(prompt, system=system, max_tokens=200).strip()
        except Exception as exc:
            log.warning("LLM summary failed for community %d: %s", community_id, exc)
            return ""

        if not summary:
            return ""

        # Derive title from first sentence
        first_sentence = summary.split(".")[0].strip()
        title = first_sentence[:200] if first_sentence else f"Community {community_id}"

        # Embed
        try:
            embs = _embed_batch([summary], self._emb_url, self._emb_model)
            embedding = embs[0] if embs else None
        except Exception as exc:
            log.warning("Embedding community summary failed: %s", exc)
            embedding = None

        with self.conn.cursor() as cur:
            cur.execute("""
                UPDATE rag.communities
                SET    summary           = %s,
                       title             = %s,
                       summary_embedding = %s::vector
                WHERE  id = %s
            """, (summary, title, embedding, community_id))
        self.conn.commit()

        return summary

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def process_collection(self) -> dict:
        """
        Run the full community-detection pipeline for ``self.collection``.

        Returns
        -------
        dict
            Stats: ``{levels, communities, summarised}``
        """
        t0 = time.time()

        community_map = self.detect_communities()

        total_communities = sum(len(v) for v in community_map.values())
        summarised = 0

        for level, comm_ids in community_map.items():
            log.info("Summarising %d communities at level %d…", len(comm_ids), level)
            for comm_id in comm_ids:
                try:
                    text = self.summarize_community(comm_id)
                    if text:
                        summarised += 1
                except Exception as exc:
                    log.error("summarize_community(%d) failed: %s", comm_id, exc)

        elapsed = time.time() - t0
        stats = {
            "levels":       len(community_map),
            "communities":  total_communities,
            "summarised":   summarised,
            "elapsed_s":    round(elapsed, 2),
        }
        log.info("Community detection complete: %s", stats)
        return stats
