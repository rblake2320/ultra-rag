"""
Query Decomposer: breaks compound and multi-hop queries into sub-queries.

For compound queries (multiple distinct questions in one), it decomposes
them into independent parallel sub-queries.  For multi-hop queries it
identifies dependency chains — sub-queries whose answer depends on the
result of an earlier sub-query.

After parallel/sequential retrieval, :meth:`QueryDecomposer.merge_results`
fuses all sub-result lists via Reciprocal Rank Fusion (RRF) into a single
ranked result set.

Usage
-----
    from src.query_decomposer import QueryDecomposer
    decomposer = QueryDecomposer()

    sub_queries = decomposer.decompose(
        "What fields are on screen 207 and also how do I close a job card?"
    )
    # → [{'query': 'What fields are on screen 207?', 'type': 'entity_focused',
    #      'depends_on': None},
    #    {'query': 'How do I close a job card?', 'type': 'specific_factoid',
    #      'depends_on': None}]
"""
from __future__ import annotations

import logging
from typing import Optional

from .llm import LLMClient

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM prompt templates
# ---------------------------------------------------------------------------

_DECOMPOSE_SYSTEM = (
    "You are a query decomposition assistant for a technical document retrieval system. "
    "Your job is to split a complex user query into simpler, independent sub-queries "
    "that together fully answer the original.\n\n"
    "Rules:\n"
    "  1. Each sub-query must be self-contained and retrievable on its own.\n"
    "  2. If a sub-query requires the answer to a prior sub-query, set "
    "     'depends_on' to the 0-based index of that prerequisite; otherwise null.\n"
    "  3. Use at most {max_subqueries} sub-queries.\n"
    "  4. For each sub-query assign a 'type' from: specific_factoid, entity_focused, "
    "     global_thematic, multi_hop, vague_exploratory, compound.\n"
    "  5. Return ONLY valid JSON — no prose, no markdown fences.\n\n"
    "Return JSON: {{\"sub_queries\": ["
    "{{\"query\": str, \"type\": str, \"depends_on\": int | null}}, ..."
    "]}}"
)

_DECOMPOSE_PROMPT_TEMPLATE = (
    "Original query: \"{query}\"\n\n"
    "Decompose into {max_subqueries} or fewer sub-queries."
)

_VALID_QUERY_TYPES = frozenset({
    "specific_factoid",
    "entity_focused",
    "global_thematic",
    "multi_hop",
    "vague_exploratory",
    "compound",
})

# ---------------------------------------------------------------------------
# QueryDecomposer
# ---------------------------------------------------------------------------

class QueryDecomposer:
    """
    Splits compound / multi-hop queries into ranked sub-queries for retrieval.

    Parameters
    ----------
    llm_client : LLMClient, optional
        Client used for decomposition inference.  Defaults to a new
        LLMClient using the fast model (gemma3:latest) for low latency.
    """

    def __init__(self, llm_client: Optional[LLMClient] = None) -> None:
        self._llm = llm_client or LLMClient(model="gemma3:latest")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decompose(self, query: str, max_subqueries: int = 4) -> list:
        """
        Break *query* into sub-queries.

        Parameters
        ----------
        query : str
            Original user query.
        max_subqueries : int
            Upper bound on the number of sub-queries returned (default 4).

        Returns
        -------
        list of dicts, each with:
          query      : str   — the sub-query text
          type       : str   — one of the QUERY_TYPES keys
          depends_on : int | None — 0-based index of prerequisite, or None
        """
        if not query or not query.strip():
            return self._fallback(query)

        max_subqueries = max(1, min(max_subqueries, 8))  # guard extremes

        system = _DECOMPOSE_SYSTEM.format(max_subqueries=max_subqueries)
        prompt = _DECOMPOSE_PROMPT_TEMPLATE.format(
            query=query.strip(),
            max_subqueries=max_subqueries,
        )

        try:
            data = self._llm.complete_json(prompt, system=system)
        except Exception as exc:
            log.warning("QueryDecomposer LLM call failed: %s", exc)
            return self._fallback(query)

        sub_queries = self._parse_response(data, query)
        return sub_queries

    def merge_results(self, sub_results: list, original_query: str) -> list:  # noqa: ARG002
        """
        Merge multiple sub-query result lists into a single ranked list.

        Parameters
        ----------
        sub_results : list of lists
            One inner list per sub-query, each containing result dicts with
            at least an 'id' key.  Empty sub-result lists are silently skipped.
        original_query : str
            Retained for logging / future re-ranking hooks (not used directly).

        Returns
        -------
        Merged, deduplicated list sorted by cross-sub-query RRF score.
        Each result dict gains an 'rrf_score' key.
        """
        # Filter out empty result lists
        non_empty = [rl for rl in sub_results if rl]
        if not non_empty:
            return []

        from .hyde import rrf_merge  # avoid circular import at module level
        return rrf_merge(non_empty, k=60, top_k=50)  # caller can slice further

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_response(self, data: dict, original_query: str) -> list:
        """
        Validate and normalise the LLM JSON response.

        Falls back to :meth:`_fallback` if the response is malformed or
        contains only one trivially passthrough sub-query.
        """
        if not data or not isinstance(data, dict):
            log.debug("QueryDecomposer: empty/non-dict response, using fallback")
            return self._fallback(original_query)

        raw_subs = data.get("sub_queries", [])
        if not isinstance(raw_subs, list) or len(raw_subs) == 0:
            return self._fallback(original_query)

        cleaned: list = []
        for i, item in enumerate(raw_subs):
            if not isinstance(item, dict):
                continue

            q = str(item.get("query", "")).strip()
            if not q:
                continue

            q_type = str(item.get("type", "specific_factoid")).strip()
            if q_type not in _VALID_QUERY_TYPES:
                log.debug(
                    "QueryDecomposer: unknown sub-query type %r, defaulting to specific_factoid",
                    q_type,
                )
                q_type = "specific_factoid"

            depends_on = item.get("depends_on")
            if depends_on is not None:
                try:
                    depends_on = int(depends_on)
                    # Validate the dependency index is in range
                    if depends_on < 0 or depends_on >= len(raw_subs):
                        log.debug(
                            "QueryDecomposer: sub-query %d has out-of-range depends_on=%d; "
                            "clearing dependency.",
                            i, depends_on,
                        )
                        depends_on = None
                    elif depends_on >= i:
                        # Forward dependency — not valid; clear it
                        log.debug(
                            "QueryDecomposer: sub-query %d has forward/self depends_on=%d; "
                            "clearing dependency.",
                            i, depends_on,
                        )
                        depends_on = None
                except (TypeError, ValueError):
                    depends_on = None

            cleaned.append({
                "query":      q,
                "type":       q_type,
                "depends_on": depends_on,
            })

        # If we ended up with 0 or 1 passthrough, use fallback
        if len(cleaned) == 0:
            return self._fallback(original_query)

        if len(cleaned) == 1 and cleaned[0]["query"].lower() == original_query.lower().strip():
            # LLM just returned the original query unchanged — use fallback
            return self._fallback(original_query)

        return cleaned

    @staticmethod
    def _fallback(query: str) -> list:
        """Return a single-item list wrapping the original query unchanged."""
        return [{
            "query":      query.strip() if query else "",
            "type":       "specific_factoid",
            "depends_on": None,
        }]
