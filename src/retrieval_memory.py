"""
Retrieval Memory: tracks which chunks are actually useful for queries.
Uses exponential moving average (EMA) to bias future retrieval.
Novel: bandit algorithm for RAG — reinforcement learning for chunk selection.

Every time a chunk is retrieved it accumulates a retrieve_count.
When a caller signals that a chunk was actually *used* (contributed to an
answer), the chunk's utility EMA is updated toward the contribution score.
Chunks that haven't been used recently are gently decayed back toward the
neutral 0.5 prior.

apply_utility_boost() can re-rank any result list by blending the original
retrieval score with the learned utility EMA, effectively implementing an
upper-confidence-bound (UCB) style bandit on chunk selection.
"""
import logging
from typing import Optional

import psycopg2
import psycopg2.extras

from .config import get_config
from .db import get_conn

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALPHA = 0.15  # EMA learning rate — higher = faster adaptation

# Default utility when a chunk has never been seen
_DEFAULT_UTILITY = 0.5


# ---------------------------------------------------------------------------
# RetrievalMemory
# ---------------------------------------------------------------------------

class RetrievalMemory:
    """
    Bandit-style retrieval memory based on exponential moving averages.

    Persists chunk utility to ``rag.chunk_utility`` so learning carries over
    across sessions.  Integrates with the search pipeline via
    :meth:`apply_utility_boost` to bias result re-ranking toward proven chunks.

    Usage::

        mem = RetrievalMemory(conn)
        mem.record_retrieval(chunk_ids)          # after every search
        mem.record_use(chunk_ids, scores)         # after answer generation
        results = mem.apply_utility_boost(results)
    """

    def __init__(self, conn, alpha: float = ALPHA) -> None:
        self.conn  = conn
        self.alpha = alpha

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_retrieval(
        self,
        chunk_ids: list,
        collection: Optional[str] = None,
    ) -> None:
        """
        Increment retrieve_count for every chunk that was returned in a search.

        Upserts a row into ``rag.chunk_utility`` with a neutral starting EMA
        of 0.5 for previously unseen chunks.

        Parameters
        ----------
        chunk_ids:
            List of chunk IDs (int) that were retrieved.
        collection:
            Optional collection label for logging.
        """
        if not chunk_ids:
            return

        try:
            with self.conn.cursor() as cur:
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO rag.chunk_utility
                        (chunk_id, utility_ema, retrieve_count, updated_at)
                    VALUES %s
                    ON CONFLICT (chunk_id) DO UPDATE
                        SET retrieve_count = rag.chunk_utility.retrieve_count + 1,
                            updated_at     = now()
                    """,
                    [(cid, _DEFAULT_UTILITY, 1) for cid in chunk_ids],
                    template="(%s, %s, %s, now())",
                )
            self.conn.commit()
            log.debug(
                "record_retrieval: %d chunks (collection=%s)",
                len(chunk_ids), collection,
            )
        except Exception as exc:
            self.conn.rollback()
            log.error("record_retrieval failed: %s", exc)

    def record_use(
        self,
        chunk_ids: list,
        contribution_scores: Optional[list] = None,
    ) -> None:
        """
        Update utility EMA for chunks that were actually used in an answer.

        EMA update formula:
            utility_ema = (1 - alpha) * utility_ema + alpha * contribution

        Parameters
        ----------
        chunk_ids:
            List of chunk IDs that contributed to the answer.
        contribution_scores:
            Optional list of floats (0-1) indicating how much each chunk
            contributed.  Must be the same length as chunk_ids when provided.
            Defaults to 1.0 for each chunk.
        """
        if not chunk_ids:
            return

        if contribution_scores is None:
            contribution_scores = [1.0] * len(chunk_ids)

        # Pad or truncate to match chunk_ids length
        while len(contribution_scores) < len(chunk_ids):
            contribution_scores.append(1.0)

        try:
            with self.conn.cursor() as cur:
                for chunk_id, contribution in zip(chunk_ids, contribution_scores):
                    contribution = max(0.0, min(1.0, float(contribution)))
                    cur.execute(
                        """
                        INSERT INTO rag.chunk_utility
                            (chunk_id, utility_ema, use_count, last_used, updated_at)
                        VALUES (%s, %s, 1, now(), now())
                        ON CONFLICT (chunk_id) DO UPDATE
                            SET utility_ema = (1 - %s) * rag.chunk_utility.utility_ema
                                              + %s * %s,
                                use_count   = rag.chunk_utility.use_count + 1,
                                last_used   = now(),
                                updated_at  = now()
                        """,
                        (
                            chunk_id,
                            (1 - self.alpha) * _DEFAULT_UTILITY + self.alpha * contribution,
                            self.alpha,
                            self.alpha,
                            contribution,
                        ),
                    )
            self.conn.commit()
            log.debug("record_use: %d chunks updated", len(chunk_ids))
        except Exception as exc:
            self.conn.rollback()
            log.error("record_use failed: %s", exc)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get_utility_scores(self, chunk_ids: list) -> dict:
        """
        Fetch utility EMA scores for the given chunk IDs.

        Parameters
        ----------
        chunk_ids:
            List of chunk IDs to look up.

        Returns
        -------
        dict mapping chunk_id (int) → utility_ema (float).
        Chunks not yet in the table default to ``_DEFAULT_UTILITY`` (0.5).
        """
        if not chunk_ids:
            return {}

        scores: dict[int, float] = {cid: _DEFAULT_UTILITY for cid in chunk_ids}

        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT chunk_id, utility_ema
                    FROM   rag.chunk_utility
                    WHERE  chunk_id = ANY(%s)
                    """,
                    (list(chunk_ids),),
                )
                for row in cur.fetchall():
                    scores[row[0]] = float(row[1])
        except Exception as exc:
            log.error("get_utility_scores failed: %s", exc)

        return scores

    def apply_utility_boost(
        self,
        results: list,
        boost_weight: float = 0.3,
    ) -> list:
        """
        Re-rank results by blending retrieval score with learned utility EMA.

        Boosted score formula:
            boosted = score * (1 + boost_weight * (utility_ema - 0.5))

        Chunks with utility > 0.5 are up-ranked; chunks with utility < 0.5
        are slightly down-ranked.  A utility of exactly 0.5 leaves the
        original score unchanged.

        Parameters
        ----------
        results:
            List of result dicts (must have 'id' and 'score').
        boost_weight:
            Strength of the utility adjustment (0 = no boost, 1 = max boost).

        Returns
        -------
        New list sorted by boosted_score descending, with 'utility_boost'
        and 'utility_ema' fields added to each dict.
        """
        if not results:
            return results

        chunk_ids      = [r["id"] for r in results]
        utility_scores = self.get_utility_scores(chunk_ids)

        boosted: list[dict] = []
        for result in results:
            cid         = result["id"]
            utility_ema = utility_scores.get(cid, _DEFAULT_UTILITY)
            base_score  = float(result.get("score", 0.0))
            adjustment  = boost_weight * (utility_ema - 0.5)
            boosted_score = base_score * (1.0 + adjustment)

            enriched                = dict(result)
            enriched["utility_ema"]    = round(utility_ema, 4)
            enriched["utility_boost"]  = round(adjustment, 4)
            enriched["boosted_score"]  = round(boosted_score, 6)
            boosted.append(enriched)

        boosted.sort(key=lambda r: r["boosted_score"], reverse=True)
        log.debug("apply_utility_boost: re-ranked %d results", len(boosted))
        return boosted

    def get_top_utility_chunks(
        self,
        collection: Optional[str] = None,
        top_k: int = 100,
    ) -> list:
        """
        Return the highest-utility chunks (for warmup, eval, or inspection).

        Parameters
        ----------
        collection:
            Filter by collection when provided.
        top_k:
            Maximum number of chunks to return.

        Returns
        -------
        List of dicts with fields: chunk_id, utility_ema, retrieve_count,
        use_count, last_used, content, collection.
        """
        try:
            with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                if collection:
                    cur.execute(
                        """
                        SELECT cu.chunk_id, cu.utility_ema, cu.retrieve_count,
                               cu.use_count, cu.last_used,
                               c.content, c.collection
                        FROM   rag.chunk_utility cu
                        JOIN   rag.chunks        c  ON c.id = cu.chunk_id
                        WHERE  c.collection = %s
                        ORDER  BY cu.utility_ema DESC
                        LIMIT  %s
                        """,
                        (collection, top_k),
                    )
                else:
                    cur.execute(
                        """
                        SELECT cu.chunk_id, cu.utility_ema, cu.retrieve_count,
                               cu.use_count, cu.last_used,
                               c.content, c.collection
                        FROM   rag.chunk_utility cu
                        JOIN   rag.chunks        c  ON c.id = cu.chunk_id
                        ORDER  BY cu.utility_ema DESC
                        LIMIT  %s
                        """,
                        (top_k,),
                    )
                rows = cur.fetchall()
            return [dict(r) for r in rows]
        except Exception as exc:
            log.error("get_top_utility_chunks failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def decay_old_utility(self, days_threshold: int = 30) -> int:
        """
        Gently decay utility for chunks that haven't been used recently.

        Unused chunks drift back toward the neutral prior of 0.5 using:
            utility_ema = 0.9 * utility_ema + 0.1 * 0.5

        This prevents stale high-utility scores from perpetually biasing
        retrieval for chunks that may no longer be relevant.

        Parameters
        ----------
        days_threshold:
            Number of days of inactivity before decay is applied.

        Returns
        -------
        int — number of rows updated.
        """
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE rag.chunk_utility
                    SET    utility_ema = 0.9 * utility_ema + 0.1 * 0.5,
                           updated_at  = now()
                    WHERE  last_used < now() - INTERVAL '%s days'
                       OR  (last_used IS NULL AND updated_at < now() - INTERVAL '%s days')
                    """,
                    (days_threshold, days_threshold),
                )
                count = cur.rowcount
            self.conn.commit()
            log.info(
                "decay_old_utility: decayed %d chunks (threshold=%d days)",
                count, days_threshold,
            )
            return count
        except Exception as exc:
            self.conn.rollback()
            log.error("decay_old_utility failed: %s", exc)
            return 0

    def stats(self) -> dict:
        """
        Return summary statistics for the chunk utility table.

        Returns
        -------
        dict with keys: total_chunks, avg_utility, max_utility, min_utility,
        total_retrievals, total_uses.
        """
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        COUNT(*)          AS total_chunks,
                        AVG(utility_ema)  AS avg_utility,
                        MAX(utility_ema)  AS max_utility,
                        MIN(utility_ema)  AS min_utility,
                        SUM(retrieve_count) AS total_retrievals,
                        SUM(use_count)    AS total_uses
                    FROM rag.chunk_utility
                    """
                )
                row = cur.fetchone()
            if not row or row[0] == 0:
                return {
                    "total_chunks": 0,
                    "avg_utility":  0.5,
                    "max_utility":  0.5,
                    "min_utility":  0.5,
                    "total_retrievals": 0,
                    "total_uses":   0,
                }
            return {
                "total_chunks":     int(row[0]),
                "avg_utility":      round(float(row[1] or 0.5), 4),
                "max_utility":      round(float(row[2] or 0.5), 4),
                "min_utility":      round(float(row[3] or 0.5), 4),
                "total_retrievals": int(row[4] or 0),
                "total_uses":       int(row[5] or 0),
            }
        except Exception as exc:
            log.error("stats() failed: %s", exc)
            return {}
