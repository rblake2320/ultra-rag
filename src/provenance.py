"""
Provenance chains: records decomposed confidence scores for every retrieval result.

Shows exactly why each chunk was selected:
  keyword_rrf + vector_rrf + kg_diffusion + utility_boost + rerank_score

Each query answer can have a provenance chain attached showing the full scoring
breakdown for every retrieved chunk, making retrieval decisions auditable and
debuggable.

Usage::

    from src.provenance import ProvenanceBuilder, build_score_components

    builder = ProvenanceBuilder(conn)
    chain_id = builder.start_chain(query_log_id=42, answer_text="The answer is...")

    for rank, result in enumerate(results, 1):
        components = build_score_components(
            keyword_rank=result.get("kw_rank", 99),
            vector_rank=result.get("vec_rank", 99),
            kg_score=result.get("kg_score", 0.0),
            utility_ema=result.get("utility_ema", 0.5),
            rerank_score=result.get("rerank_score", 0.5),
        )
        builder.add_step(chain_id, chunk_id=result["id"], score_components=components, rank=rank)

    builder.finalize_chain(chain_id, overall_confidence=0.87)
    chain = builder.get_chain(chain_id)
    print(builder.format_provenance(chain))
"""
from __future__ import annotations

import json
import logging
from typing import Optional

import psycopg2.extras

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# RRF constant (must match the value used in rrf_merge)
# ---------------------------------------------------------------------------

_RRF_K = 60


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------

def build_score_components(
    keyword_rank: int,
    vector_rank: int,
    kg_score: float,
    utility_ema: float,
    rerank_score: float,
) -> dict:
    """
    Compute the decomposed score components for a single retrieval result.

    Uses Reciprocal Rank Fusion (RRF) for the keyword and vector sub-scores
    so the individual contributions are comparable across result lists.

    Parameters
    ----------
    keyword_rank : int
        1-based rank of this result in the BM25/tsvector keyword list.
        Pass a large value (e.g. 999) if the result was not in that list.
    vector_rank : int
        1-based rank of this result in the vector similarity list.
        Pass a large value (e.g. 999) if not present.
    kg_score : float
        Knowledge-graph PPR/diffusion score (0–1).  0.0 if not retrieved via KG.
    utility_ema : float
        Exponential moving average utility score from rag.chunk_utility (0–1).
        Default 0.5 for unseen chunks.
    rerank_score : float
        Cross-encoder reranker score (0–1).  0.5 if reranking was not applied.

    Returns
    -------
    dict with keys: keyword_rrf, vector_rrf, kg_diffusion, utility_boost,
    rerank_score, composite.
    """
    keyword_rrf = round(1.0 / (_RRF_K + max(1, keyword_rank)), 8)
    vector_rrf  = round(1.0 / (_RRF_K + max(1, vector_rank)),  8)
    kg_diffusion  = round(float(max(0.0, min(1.0, kg_score))),   6)
    utility_boost = round(float(max(0.0, min(1.0, utility_ema))), 6)
    rr_score      = round(float(max(0.0, min(1.0, rerank_score))), 6)

    # Composite: weighted sum (rerank dominates when available)
    composite = round(
        0.15 * keyword_rrf * 1000   # scale RRF to [0, ~15] then weight
        + 0.15 * vector_rrf  * 1000
        + 0.10 * kg_diffusion
        + 0.10 * utility_boost
        + 0.50 * rr_score,
        6,
    )

    return {
        "keyword_rrf":    keyword_rrf,
        "vector_rrf":     vector_rrf,
        "kg_diffusion":   kg_diffusion,
        "utility_boost":  utility_boost,
        "rerank_score":   rr_score,
        "composite":      composite,
    }


# ---------------------------------------------------------------------------
# ProvenanceBuilder
# ---------------------------------------------------------------------------

class ProvenanceBuilder:
    """
    Build, persist, and format retrieval provenance chains.

    One chain corresponds to one query/answer pair.  Each step in the chain
    records why a specific chunk (or entity) was selected, storing the full
    decomposed score vector.

    Parameters
    ----------
    conn : psycopg2 connection
        Active database connection.  The Ultra RAG schema (rag.provenance_chains,
        rag.provenance_steps) must already exist.
    """

    def __init__(self, conn) -> None:
        self._conn = conn

    # ------------------------------------------------------------------
    # Chain lifecycle
    # ------------------------------------------------------------------

    def start_chain(
        self,
        query_log_id: int,
        answer_text: Optional[str] = None,
    ) -> int:
        """
        Create a new provenance chain and return its id.

        Parameters
        ----------
        query_log_id : int
            Foreign key into rag.query_log.  May be None for ad-hoc chains
            that are not tied to a logged query (use 0 as a sentinel).
        answer_text : str, optional
            The generated answer text for this chain (for audit purposes).

        Returns
        -------
        chain_id (int)
        """
        # Allow query_log_id=0 as a sentinel for un-logged queries
        qlog_id: Optional[int] = query_log_id if query_log_id and query_log_id > 0 else None

        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO rag.provenance_chains (query_log_id, answer_text)
                VALUES (%s, %s)
                RETURNING id
                """,
                (qlog_id, answer_text),
            )
            chain_id = cur.fetchone()[0]
        self._conn.commit()
        log.debug("ProvenanceBuilder: started chain %d (query_log=%s)", chain_id, qlog_id)
        return chain_id

    def add_step(
        self,
        chain_id: int,
        chunk_id: Optional[int] = None,
        entity_id: Optional[int] = None,
        score_components: Optional[dict] = None,
        rank: Optional[int] = None,
    ) -> None:
        """
        Append a provenance step to an existing chain.

        Parameters
        ----------
        chain_id : int
            The chain to append to (returned by :meth:`start_chain`).
        chunk_id : int, optional
            rag.chunks.id of the retrieved chunk.  May be None for entity steps.
        entity_id : int, optional
            rag.entities.id for KG-sourced steps.  May be None for chunk steps.
        score_components : dict, optional
            Decomposed score dict (see :func:`build_score_components`).
            Stored as JSONB.
        rank : int, optional
            1-based position of this result in the final merged list.
        """
        components_json = json.dumps(score_components or {})

        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO rag.provenance_steps
                    (chain_id, chunk_id, entity_id, score_components, rank_position)
                VALUES (%s, %s, %s, %s::jsonb, %s)
                """,
                (chain_id, chunk_id, entity_id, components_json, rank),
            )
        self._conn.commit()

    def finalize_chain(self, chain_id: int, overall_confidence: float) -> None:
        """
        Set the overall_confidence on a completed chain.

        Parameters
        ----------
        chain_id : int
            Chain to finalise.
        overall_confidence : float
            Aggregate confidence score for the complete answer (0–1).
        """
        confidence = round(max(0.0, min(1.0, float(overall_confidence))), 6)
        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE rag.provenance_chains SET overall_confidence = %s WHERE id = %s",
                (confidence, chain_id),
            )
        self._conn.commit()
        log.debug(
            "ProvenanceBuilder: finalized chain %d — confidence=%.4f",
            chain_id, confidence,
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_chain(self, chain_id: int) -> dict:
        """
        Retrieve a complete provenance chain with all steps.

        Joins rag.provenance_chains → rag.provenance_steps → rag.chunks so
        that the returned dict contains the original chunk content for each step.

        Parameters
        ----------
        chain_id : int
            Chain to retrieve.

        Returns
        -------
        dict with keys:
            id, query_log_id, answer_text, overall_confidence, created_at,
            steps: list of step dicts (rank, chunk_id, content snippet,
                   entity_id, score_components)
        """
        # Fetch chain header
        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, query_log_id, answer_text, overall_confidence, created_at
                FROM rag.provenance_chains
                WHERE id = %s
                """,
                (chain_id,),
            )
            chain_row = cur.fetchone()

        if not chain_row:
            log.warning("get_chain: chain %d not found", chain_id)
            return {}

        chain = dict(chain_row)

        # Fetch steps with chunk content
        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    ps.id           AS step_id,
                    ps.rank_position,
                    ps.chunk_id,
                    ps.entity_id,
                    ps.score_components,
                    LEFT(c.content, 300)  AS content_snippet,
                    c.content_type,
                    c.stable_id
                FROM rag.provenance_steps ps
                LEFT JOIN rag.chunks c ON c.id = ps.chunk_id
                WHERE ps.chain_id = %s
                ORDER BY ps.rank_position ASC NULLS LAST, ps.id ASC
                """,
                (chain_id,),
            )
            step_rows = cur.fetchall()

        steps = []
        for row in step_rows:
            step = dict(row)
            # score_components is already a dict from JSONB (psycopg2 returns it as dict)
            if isinstance(step.get("score_components"), str):
                try:
                    step["score_components"] = json.loads(step["score_components"])
                except (json.JSONDecodeError, TypeError):
                    step["score_components"] = {}
            steps.append(step)

        chain["steps"] = steps
        return chain

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def format_provenance(self, chain: dict) -> str:
        """
        Render a human-readable provenance summary.

        Parameters
        ----------
        chain : dict
            A chain dict as returned by :meth:`get_chain`.

        Returns
        -------
        Multi-line string with one line per result step, showing the
        decomposed score components.

        Example output::

            Provenance chain #7 — confidence: 0.8700
            Answer: The approval threshold is $50,000 per section 4.2...

            Result 1 (chunk #4512): score=0.87 [keyword:0.0161 + vector:0.0159 + kg:0.0500 + utility:0.1200 + rerank:0.8700]
            Content: Purchase orders above threshold require secondary review per policy...

            Result 2 (chunk #3890): score=0.73 [keyword:0.0156 + vector:0.0154 + kg:0.0000 + utility:0.5000 + rerank:0.7300]
            Content: When submitted the system validates the vendor registration...
        """
        if not chain:
            return "(empty provenance chain)"

        lines = [
            f"Provenance chain #{chain.get('id', '?')} "
            f"— confidence: {chain.get('overall_confidence', 0.0):.4f}",
        ]

        answer = chain.get("answer_text")
        if answer:
            preview = answer[:120].replace("\n", " ")
            lines.append(f"Answer: {preview}{'...' if len(answer) > 120 else ''}")

        steps = chain.get("steps", [])
        if not steps:
            lines.append("(no steps recorded)")
            return "\n".join(lines)

        lines.append("")  # blank line before steps

        for step in steps:
            rank    = step.get("rank_position") or "?"
            cid     = step.get("chunk_id")
            eid     = step.get("entity_id")
            sc      = step.get("score_components") or {}

            # Subject identifier
            if cid:
                subject = f"chunk #{cid}"
            elif eid:
                subject = f"entity #{eid}"
            else:
                subject = "unknown"

            # Composite score — prefer composite key, else sum keyword+vector+rerank
            composite = sc.get("composite")
            if composite is None:
                composite = (
                    sc.get("keyword_rrf", 0.0)
                    + sc.get("vector_rrf", 0.0)
                    + sc.get("rerank_score", 0.0)
                )

            score_detail = (
                f"keyword:{sc.get('keyword_rrf', 0.0):.4f} + "
                f"vector:{sc.get('vector_rrf', 0.0):.4f} + "
                f"kg:{sc.get('kg_diffusion', 0.0):.4f} + "
                f"utility:{sc.get('utility_boost', 0.0):.4f} + "
                f"rerank:{sc.get('rerank_score', 0.0):.4f}"
            )

            lines.append(
                f"Result {rank} ({subject}): score={composite:.4f} "
                f"[{score_detail}]"
            )

            snippet = step.get("content_snippet", "")
            if snippet:
                lines.append(f"Content: {snippet.strip()[:200]}")

            lines.append("")  # blank line between steps

        return "\n".join(lines).rstrip()
