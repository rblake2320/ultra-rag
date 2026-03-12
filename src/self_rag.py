"""
Self-RAG: adaptive retrieval with passage-level relevance filtering.
Decides whether retrieval is needed, which passages are relevant,
and whether the answer is supported.

Self-RAG adds fine-grained reflection tokens at generation time:
    [Retrieve]            → external retrieval needed
    [No Retrieve]         → model knowledge sufficient
    [Relevant]            → passage supports the query
    [Irrelevant]          → passage should be discarded
    [Supported]           → answer claim is grounded
    [Partially Supported] → some claims grounded
    [No Support]          → answer is not grounded
"""
import logging
import re
from typing import Callable, Optional

from .config import get_config
from .db import get_conn

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reflection token constants
# ---------------------------------------------------------------------------

REFLECTION_TOKENS = {
    "retrieve":           "[Retrieve]",
    "no_retrieve":        "[No Retrieve]",
    "relevant":           "[Relevant]",
    "irrelevant":         "[Irrelevant]",
    "supported":          "[Supported]",
    "partially_supported":"[Partially Supported]",
    "no_support":         "[No Support]",
}

# Patterns that suggest a query needs external facts
_FACTOID_PATTERNS = re.compile(
    r"\b(\d{4}|\d{1,2}/\d{1,2}|\bwhen\b|\bwhere\b|\bwho\b|\bhow many\b"
    r"|\bwhat is\b|\bwhat are\b|[A-Z]{2,6}-\d+|error code\b|fault code\b"
    r"|\bscreen\b|\bstatus\b|\bversion\b|\bidentifier\b)",
    re.IGNORECASE,
)

# Number of passages processed in a single relevance-scoring LLM call
_BATCH_SIZE = 5


# ---------------------------------------------------------------------------
# SelfRAG
# ---------------------------------------------------------------------------

class SelfRAG:
    """
    Self-RAG: decides whether to retrieve, filters irrelevant passages, and
    assesses whether generated claims are supported by the evidence.

    Usage::

        rag = SelfRAG()
        passages = rag.adaptive_retrieve_and_filter(query, search_fn)
    """

    def __init__(
        self,
        llm_client=None,
        retrieve_threshold: float = 0.6,
    ) -> None:
        self._llm               = llm_client
        self.retrieve_threshold = retrieve_threshold

    # ------------------------------------------------------------------
    # LLM accessor (lazy)
    # ------------------------------------------------------------------

    @property
    def llm(self):
        if self._llm is None:
            from .llm import LLMClient  # noqa: PLC0415
            self._llm = LLMClient()
        return self._llm

    # ------------------------------------------------------------------
    # Retrieval decision
    # ------------------------------------------------------------------

    def should_retrieve(
        self,
        query: str,
        context: Optional[str] = None,
    ) -> bool:
        """
        Decide whether external retrieval is necessary for this query.

        Factoid queries (those containing dates, military document codes,
        error codes, screen names, or specific entity names) always trigger
        retrieval.  For other queries the LLM is asked directly.

        Parameters
        ----------
        query:
            User query string.
        context:
            Optional existing context (e.g. conversation history).  When
            present and substantial, may reduce the need for retrieval.

        Returns
        -------
        bool — True means "retrieve external documents".
        """
        # Hard rule: factoid patterns always need retrieval
        if _FACTOID_PATTERNS.search(query):
            log.debug("should_retrieve=True (factoid pattern matched): %r", query[:80])
            return True

        prompt_parts = [
            f"Does answering this query require retrieving external information?\n"
            f"Query: {query}"
        ]
        if context:
            prompt_parts.append(f"Existing context: {context[:500]}")
        prompt_parts.append('Return JSON: {"retrieve": <true|false>, "reason": <str>}')

        prompt = "\n".join(prompt_parts)

        try:
            data     = self.llm.complete_json(prompt)
            decision = bool(data.get("retrieve", True))
            log.debug(
                "should_retrieve=%s reason=%s",
                decision, data.get("reason", ""),
            )
            return decision
        except Exception as exc:
            log.warning("should_retrieve LLM call failed (%s); defaulting to True", exc)
            return True  # safe default: always retrieve

    # ------------------------------------------------------------------
    # Passage relevance filtering
    # ------------------------------------------------------------------

    def filter_relevant_passages(
        self,
        query: str,
        passages: list,
        threshold: float = 0.5,
    ) -> list:
        """
        Score each passage for relevance to the query and drop irrelevant ones.

        Passages are scored in batches of ``_BATCH_SIZE`` to keep individual
        LLM prompts manageable.  Each passage receives a 0-10 score; those
        below ``threshold * 10`` are discarded.

        A ``relevance_score`` field (0-1) and ``reflection_token`` field are
        added to every retained passage dict.

        Parameters
        ----------
        query:
            User query string.
        passages:
            List of result dicts (must have 'id' and 'content').
        threshold:
            Minimum score fraction (0-1) to retain a passage.

        Returns
        -------
        Filtered list of passage dicts with 'relevance_score' added.
        """
        if not passages:
            return []

        filtered: list[dict] = []
        cutoff = threshold * 10.0  # convert to 0-10 scale

        for batch_start in range(0, len(passages), _BATCH_SIZE):
            batch = passages[batch_start : batch_start + _BATCH_SIZE]

            # Build numbered list for the LLM
            numbered = "\n".join(
                f"{i+1}. {(p.get('content') or '')[:300]}"
                for i, p in enumerate(batch)
            )
            prompt = (
                f"Rate the relevance of each passage to the query (0-10).\n"
                f"Query: {query}\n"
                f"Passages:\n{numbered}\n"
                f"Return a JSON array of integer scores, one per passage, "
                f"in the same order. Example: [7, 3, 9]"
            )

            scores: list[float] = []
            try:
                raw = self.llm.complete_json(prompt)
                # The LLM may return a list directly or {"scores": [...]}
                if isinstance(raw, list):
                    scores = [float(s) for s in raw]
                elif isinstance(raw, dict):
                    for key in ("scores", "relevance", "ratings"):
                        if key in raw and isinstance(raw[key], list):
                            scores = [float(s) for s in raw[key]]
                            break
            except Exception as exc:
                log.warning("Batch relevance scoring failed: %s", exc)

            # Pad scores to batch length if the LLM returned fewer
            while len(scores) < len(batch):
                scores.append(5.0)  # neutral default

            for passage, raw_score in zip(batch, scores):
                raw_score   = max(0.0, min(10.0, raw_score))
                norm_score  = raw_score / 10.0

                if raw_score >= cutoff:
                    token = REFLECTION_TOKENS["relevant"]
                else:
                    token = REFLECTION_TOKENS["irrelevant"]

                enriched = dict(passage)
                enriched["relevance_score"]   = round(norm_score, 4)
                enriched["reflection_token"]  = token

                if raw_score >= cutoff:
                    filtered.append(enriched)
                else:
                    log.debug(
                        "Dropped irrelevant passage id=%s score=%.1f",
                        passage.get("id"), raw_score,
                    )

        log.info(
            "filter_relevant_passages: kept %d/%d (threshold=%.2f)",
            len(filtered), len(passages), threshold,
        )
        return filtered

    # ------------------------------------------------------------------
    # Support assessment
    # ------------------------------------------------------------------

    def assess_support(
        self,
        claim: str,
        passages: list,
    ) -> dict:
        """
        Determine whether a specific claim is supported by the retrieved passages.

        Parameters
        ----------
        claim:
            A single claim or sentence from a generated answer.
        passages:
            List of result dicts to check against.

        Returns
        -------
        dict with keys:
            support_level     str    "supported" | "partial" | "unsupported"
            reflection_token  str    one of REFLECTION_TOKENS values
            evidence_snippets list[str]
        """
        context_parts = [
            (p.get("content") or "")[:400]
            for p in passages[:3]
        ]
        context_text = "\n\n".join(context_parts)

        prompt = (
            f"Is this claim supported by these passages?\n"
            f"Claim: {claim}\n"
            f"Passages:\n{context_text}\n"
            f"Return JSON: {{"
            f"\"support_level\": <\"supported\"|\"partial\"|\"unsupported\">, "
            f"\"evidence_snippets\": [<str>, ...]}}"
        )

        try:
            data = self.llm.complete_json(prompt)
            level     = str(data.get("support_level", "unsupported")).lower()
            snippets  = list(data.get("evidence_snippets") or [])
        except Exception as exc:
            log.warning("assess_support LLM call failed: %s", exc)
            level    = "unsupported"
            snippets = []

        token_map = {
            "supported":   REFLECTION_TOKENS["supported"],
            "partial":     REFLECTION_TOKENS["partially_supported"],
            "unsupported": REFLECTION_TOKENS["no_support"],
        }
        reflection_token = token_map.get(level, REFLECTION_TOKENS["no_support"])

        return {
            "support_level":     level,
            "reflection_token":  reflection_token,
            "evidence_snippets": snippets,
        }

    # ------------------------------------------------------------------
    # Adaptive retrieve-and-filter
    # ------------------------------------------------------------------

    def adaptive_retrieve_and_filter(
        self,
        query: str,
        search_fn: Callable,
        initial_results: Optional[list] = None,
    ) -> list:
        """
        Full Self-RAG pipeline: decide, retrieve, filter, re-retrieve if thin.

        Workflow:
        1. Call ``should_retrieve`` — if False and initial_results given, return top 3.
        2. Run ``search_fn`` if no initial_results (or retrieval is needed).
        3. ``filter_relevant_passages`` to drop irrelevant results.
        4. If < 3 relevant passages remain, re-retrieve with a simplified query
           and filter again.
        5. Return final passage list with reflection tokens attached.

        Parameters
        ----------
        query:
            User query string.
        search_fn:
            Callable(query: str, top_k: int) -> list[dict].
        initial_results:
            Optional pre-fetched results; skips first search_fn call if provided.

        Returns
        -------
        list of passage dicts with 'relevance_score' and 'reflection_token' fields.
        """
        # Step 1: check whether retrieval is warranted
        retrieve = self.should_retrieve(query)

        if not retrieve and initial_results:
            log.info(
                "Self-RAG: no retrieval needed — returning top 3 of %d initial results",
                len(initial_results),
            )
            top3 = initial_results[:3]
            for r in top3:
                r.setdefault("reflection_token", REFLECTION_TOKENS["no_retrieve"])
            return top3

        # Step 2: retrieve if we don't have results yet
        if initial_results is None:
            log.debug("Self-RAG: calling search_fn for initial retrieval")
            try:
                initial_results = search_fn(query, top_k=10)
            except Exception as exc:
                log.error("Self-RAG initial search failed: %s", exc)
                initial_results = []

        # Step 3: filter by relevance
        relevant = self.filter_relevant_passages(
            query,
            initial_results,
            threshold=self.retrieve_threshold,
        )

        # Step 4: if too few relevant results, attempt a simplified re-retrieval
        if len(relevant) < 3:
            log.info(
                "Self-RAG: only %d relevant passages; attempting simplified re-retrieve",
                len(relevant),
            )
            # Simplify the query: take first 5 meaningful words
            simple_query = " ".join(query.split()[:5])
            try:
                extra_results = search_fn(simple_query, top_k=10)
            except Exception as exc:
                log.warning("Self-RAG re-retrieval failed: %s", exc)
                extra_results = []

            # Avoid duplicates
            seen_ids    = {r.get("id") for r in relevant}
            extra_novel = [r for r in extra_results if r.get("id") not in seen_ids]

            extra_filtered = self.filter_relevant_passages(
                query,
                extra_novel,
                threshold=self.retrieve_threshold * 0.8,  # slightly lower bar
            )
            relevant.extend(extra_filtered)
            log.info(
                "Self-RAG: after re-retrieve total relevant=%d",
                len(relevant),
            )

        log.info(
            "Self-RAG: returning %d relevant passages for query %r",
            len(relevant), query[:80],
        )
        return relevant
