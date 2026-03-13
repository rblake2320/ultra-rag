"""
CRAG: Corrective Retrieval Augmented Generation.
Evaluates retrieved context quality, triggers rewrite/re-retrieve/fallback.
Also implements NVIDIA-style dual-loop reflection (context relevance + groundedness).

Quality thresholds:
    correct   >= 0.8  → use results as-is
    ambiguous >= 0.5  → supplement with rewritten query
    incorrect  < 0.5  → full query rewrite / fallback
"""
import logging
import time
from typing import Callable, Optional

from .config import get_config
from .db import get_conn

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Quality thresholds
# ---------------------------------------------------------------------------

QUALITY_LEVELS = {
    "correct":   0.8,   # threshold for "good enough"
    "ambiguous": 0.5,   # needs correction / supplementation
    "incorrect": 0.3,   # needs full rewrite / fallback
}


def _quality_level(score: float) -> str:
    """Map a 0-1 quality score to a named quality level."""
    if score >= QUALITY_LEVELS["correct"]:
        return "correct"
    if score >= QUALITY_LEVELS["ambiguous"]:
        return "ambiguous"
    return "incorrect"


# ---------------------------------------------------------------------------
# CRAGEvaluator
# ---------------------------------------------------------------------------

class CRAGEvaluator:
    """
    Corrective RAG evaluator.

    Evaluates the quality of retrieved chunks for a given query, decides
    whether to use, supplement, or rewrite/replace the results, and
    orchestrates a full corrective pipeline when needed.

    Usage::

        conn = get_conn()
        evaluator = CRAGEvaluator(conn, "my-docs")
        result = evaluator.corrective_pipeline(query, initial_results, search_fn)
    """

    def __init__(self, conn, collection: str, llm_client=None) -> None:
        self.conn       = conn
        self.collection = collection
        self._llm       = llm_client  # injected or lazily created

    # ------------------------------------------------------------------
    # LLM accessor (lazy)
    # ------------------------------------------------------------------

    @property
    def llm(self):
        if self._llm is None:
            from .llm import LLMClient   # noqa: PLC0415
            from .config import get_config  # noqa: PLC0415
            _fast = get_config().get("llm", {}).get("fast_model", "qwen2.5:7b")
            self._llm = LLMClient(model=_fast)
        return self._llm

    # ------------------------------------------------------------------
    # Context relevance evaluation
    # ------------------------------------------------------------------

    def evaluate_context_relevance(
        self,
        query: str,
        retrieved_chunks: list,
    ) -> dict:
        """
        Score the relevance of each retrieved chunk to the query.

        For each chunk the LLM returns a 0-10 integer score plus a brief
        reason.  The overall quality is the average score normalised to [0, 1].

        Parameters
        ----------
        query:
            The original user query.
        retrieved_chunks:
            List of result dicts (must have at least 'id' and 'content').

        Returns
        -------
        dict with keys:
            overall_score  float  0-1
            chunk_scores   list of {chunk_id, score, reason}
            quality_level  str    "correct" | "ambiguous" | "incorrect"
        """
        if not retrieved_chunks:
            log.debug("evaluate_context_relevance: no chunks provided")
            return {
                "overall_score": 0.0,
                "chunk_scores":  [],
                "quality_level": "incorrect",
            }

        # Fast path: if reranker ran, use its best score as quality signal (no LLM needed)
        rerank_scores = [
            float(c["rerank_score"])
            for c in retrieved_chunks
            if c.get("rerank_score") is not None
        ]
        if rerank_scores:
            # Use top-3 average (not all — outliers drag it down)
            top3 = sorted(rerank_scores, reverse=True)[:3]
            top3_avg = sum(top3) / len(top3)
            # Cross-encoder scores are typically in -10..10; map to 0-1
            # A score of 0+ means the model thinks the passage is relevant
            normalized = max(0.0, min(1.0, (top3_avg + 3.0) / 6.0))
            log.debug("CRAG shortcut: top3_rerank_avg=%.3f → quality=%.3f (skipping LLM)", top3_avg, normalized)
            return {
                "overall_score": round(normalized, 4),
                "chunk_scores":  [],
                "quality_level": _quality_level(normalized),
            }

        chunk_scores: list[dict] = []
        raw_scores: list[float]  = []

        # Only evaluate top 5 chunks with LLM to limit latency
        for chunk in retrieved_chunks[:5]:
            chunk_id = chunk.get("id", "?")
            snippet  = (chunk.get("content") or "")[:500]

            prompt = (
                f"On a scale of 0-10, how relevant is this passage to the query?\n"
                f"Query: {query}\n"
                f"Passage: {snippet}\n"
                f"Return JSON: {{\"score\": <int 0-10>, \"reason\": <str>}}"
            )

            try:
                data = self.llm.complete_json(prompt)
                score  = float(data.get("score", 5)) / 10.0
                reason = str(data.get("reason", ""))
            except Exception as exc:
                log.warning(
                    "CRAG relevance scoring failed for chunk %s: %s", chunk_id, exc
                )
                score  = 0.5
                reason = "scoring error"

            # Clamp to [0, 1]
            score = max(0.0, min(1.0, score))
            chunk_scores.append({
                "chunk_id": chunk_id,
                "score":    score,
                "reason":   reason,
            })
            raw_scores.append(score)

        overall = sum(raw_scores) / len(raw_scores) if raw_scores else 0.0

        log.debug(
            "Context relevance: overall=%.3f level=%s chunks=%d",
            overall, _quality_level(overall), len(chunk_scores),
        )
        return {
            "overall_score": round(overall, 4),
            "chunk_scores":  chunk_scores,
            "quality_level": _quality_level(overall),
        }

    # ------------------------------------------------------------------
    # Groundedness evaluation
    # ------------------------------------------------------------------

    def evaluate_groundedness(
        self,
        query: str,
        answer: str,
        retrieved_chunks: list,
    ) -> dict:
        """
        Check whether an LLM-generated answer is grounded in the retrieved context.

        Implements NVIDIA-style dual-loop reflection: first scoring context
        relevance, then checking whether the answer is actually supported by
        that context.

        Parameters
        ----------
        query:
            The original user query (used for context framing).
        answer:
            The generated answer text to be checked.
        retrieved_chunks:
            List of result dicts.

        Returns
        -------
        dict with keys:
            grounded           bool
            score              float 0-1
            unsupported_claims list[str]
        """
        combined = "\n\n".join(
            (c.get("content") or "")[:400] for c in retrieved_chunks
        )[:2000]

        prompt = (
            f"Is this answer grounded in the provided context?\n"
            f"Answer: {answer}\n"
            f"Context:\n{combined}\n"
            f"Return JSON: {{"
            f"\"grounded\": <true|false>, "
            f"\"score\": <float 0-1>, "
            f"\"unsupported_claims\": [<str>, ...]}}"
        )

        try:
            data = self.llm.complete_json(prompt)
            grounded   = bool(data.get("grounded", False))
            score      = float(data.get("score", 0.0))
            claims     = list(data.get("unsupported_claims") or [])
        except Exception as exc:
            log.warning("Groundedness evaluation failed: %s", exc)
            grounded = False
            score     = 0.0
            claims    = []

        score = max(0.0, min(1.0, score))
        log.debug(
            "Groundedness: grounded=%s score=%.3f unsupported=%d",
            grounded, score, len(claims),
        )
        return {
            "grounded":           grounded,
            "score":              round(score, 4),
            "unsupported_claims": claims,
        }

    # ------------------------------------------------------------------
    # Action determination
    # ------------------------------------------------------------------

    def determine_action(self, quality_score: float) -> str:
        """
        Decide which corrective action to take based on quality score.

        Returns
        -------
        str
            "use_as_is"     — quality >= 0.8, results are good
            "supplement"    — quality >= 0.5, add results from rewritten query
            "rewrite_query" — quality  < 0.5, discard and retry with new query
        """
        if quality_score >= QUALITY_LEVELS["correct"]:
            return "use_as_is"
        if quality_score >= QUALITY_LEVELS["ambiguous"]:
            return "supplement"
        return "rewrite_query"

    # ------------------------------------------------------------------
    # Query rewriting
    # ------------------------------------------------------------------

    def rewrite_query(
        self,
        original_query: str,
        failed_chunks: list,
    ) -> str:
        """
        Ask the LLM to rewrite a query that produced poor results.

        The failed chunks are summarised and included to give the LLM context
        on *why* the original query didn't work.

        Parameters
        ----------
        original_query:
            The query that produced low-quality results.
        failed_chunks:
            The low-scoring chunks returned by the original query.

        Returns
        -------
        str
            Rewritten query string, or the original query on error.
        """
        snippets = "; ".join(
            (c.get("content") or "")[:150] for c in failed_chunks[:3]
        )

        prompt = (
            f"The following search query didn't find good results. "
            f"Rewrite it to be more specific and use different vocabulary.\n"
            f"Original: {original_query}\n"
            f"Context of poor results: {snippets}\n"
            f"Rewritten query:"
        )

        try:
            rewritten = self.llm.complete(
                prompt,
                system=(
                    "You are an expert query optimizer. "
                    "Output only the rewritten query — no explanations."
                ),
                max_tokens=100,
            ).strip()
            if rewritten:
                log.debug(
                    "Query rewritten: %r → %r", original_query, rewritten
                )
                return rewritten
        except Exception as exc:
            log.warning("Query rewrite failed: %s", exc)

        # Safe fallback: return original
        return original_query

    # ------------------------------------------------------------------
    # Full corrective pipeline
    # ------------------------------------------------------------------

    def corrective_pipeline(
        self,
        query: str,
        initial_results: list,
        search_fn: Callable,
    ) -> dict:
        """
        Run the full CRAG corrective loop.

        Steps:
        1. Evaluate quality of initial_results.
        2. Decide action: use_as_is / supplement / rewrite_query.
        3. Execute corrective action if needed.
        4. Return enriched result dict.

        Parameters
        ----------
        query:
            Original user query.
        initial_results:
            Results from the first retrieval pass.
        search_fn:
            Callable(query: str, top_k: int) -> list[dict]  used for
            re-retrieval.  Signature must match search.search() results.

        Returns
        -------
        dict with keys:
            results         list    final (possibly corrected) result list
            action          str     corrective action taken
            quality         float   quality score of initial retrieval
            original_query  str
            final_query     str
        """
        t0 = time.time()

        # Step 1: evaluate
        eval_result = self.evaluate_context_relevance(query, initial_results)
        quality     = eval_result["overall_score"]
        action      = self.determine_action(quality)

        log.info(
            "CRAG: query=%r quality=%.3f action=%s",
            query[:80], quality, action,
        )

        final_query = query

        if action == "use_as_is":
            results = initial_results

        elif action == "supplement":
            # Rewrite query and merge new results with initial
            rewritten   = self.rewrite_query(query, initial_results)
            final_query = rewritten
            try:
                new_results = search_fn(rewritten, top_k=len(initial_results))
            except Exception as exc:
                log.warning("Supplement search failed: %s", exc)
                new_results = []

            # Merge: deduplicate by chunk id, keep original order first
            seen    = {r["id"] for r in initial_results}
            extra   = [r for r in new_results if r.get("id") not in seen]
            results = initial_results + extra
            log.debug(
                "CRAG supplement: original=%d new=%d extra=%d merged=%d",
                len(initial_results), len(new_results), len(extra), len(results),
            )

        else:  # rewrite_query
            rewritten   = self.rewrite_query(query, initial_results)
            final_query = rewritten
            try:
                results = search_fn(rewritten, top_k=len(initial_results) or 5)
            except Exception as exc:
                log.warning("Rewrite search failed: %s", exc)
                results = initial_results  # fall back to original if re-search fails

        # Log to rag.query_log
        latency_ms = int((time.time() - t0) * 1000)
        try:
            self._log_query(
                query=query,
                quality=quality,
                action=action,
                result_count=len(results),
                latency_ms=latency_ms,
            )
        except Exception as exc:
            log.debug("Failed to log corrective query: %s", exc)

        return {
            "results":        results,
            "action":         action,
            "quality":        quality,
            "original_query": query,
            "final_query":    final_query,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log_query(
        self,
        query: str,
        quality: float,
        action: str,
        result_count: int,
        latency_ms: int,
    ) -> None:
        """Persist a row to rag.query_log for analytics / debugging."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO rag.query_log
                    (query_text, collection, strategy, quality_score,
                     latency_ms, corrective_action, result_count)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    query[:2000],
                    self.collection,
                    "crag",
                    quality,
                    latency_ms,
                    action,
                    result_count,
                ),
            )
        self.conn.commit()
