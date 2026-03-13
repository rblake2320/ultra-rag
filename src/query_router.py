"""
Query Router: classifies queries and routes to the optimal retrieval strategy.

Strategies:
  hybrid       — standard pgvector + tsvector RRF (specific factoids, default)
  kg_local     — entity-centric KG neighbourhood traversal
  kg_global    — Personalised PageRank across the full KG
  multi_hop    — decomposed sub-queries with dependency chaining
  hyde         — Hypothetical Document Embedding for vague/vocabulary-gap queries
  compound     — parallel sub-query decomposition + RRF merge

The router first attempts an LLM-based few-shot classification.  If the LLM
is unavailable or returns an unrecognisable result, a fast heuristic fallback
is applied so retrieval is never blocked.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from .llm import LLMClient

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------

QUERY_TYPES: dict[str, str] = {
    "specific_factoid":  "Single specific fact, identifier, procedure step, or field name",
    "entity_focused":    "Question about a specific entity, component, or concept",
    "global_thematic":   "High-level theme, overview, or 'what are the main…' questions",
    "multi_hop":         "Requires connecting multiple facts, e.g. 'how does X relate to Y'",
    "vague_exploratory": "Vague, open-ended, or using unfamiliar / domain-specific vocabulary",
    "compound":          "Multiple distinct sub-questions bundled in one query",
}

STRATEGY_MAP: dict[str, str] = {
    "specific_factoid":  "hybrid",
    "entity_focused":    "kg_local",
    "global_thematic":   "kg_global",
    "multi_hop":         "multi_hop",
    "vague_exploratory": "hyde",
    "compound":          "compound",
}

# ---------------------------------------------------------------------------
# Few-shot examples (one per query type) — kept compact to save tokens
# ---------------------------------------------------------------------------

_FEW_SHOT_EXAMPLES = """
EXAMPLES:
Q: "What is the maximum contract value allowed without a review board?"
A: {"query_type": "specific_factoid", "confidence": 0.95, "reasoning": "Asks for a single threshold value."}

Q: "Tell me everything about the vendor onboarding process"
A: {"query_type": "entity_focused", "confidence": 0.90, "reasoning": "Centres on a specific named process/entity."}

Q: "What are the main compliance themes across all policy documents?"
A: {"query_type": "global_thematic", "confidence": 0.88, "reasoning": "Broad overview across the whole corpus."}

Q: "How does the procurement policy relate to the budget approval workflow?"
A: {"query_type": "multi_hop", "confidence": 0.92, "reasoning": "Requires connecting two distinct entities."}

Q: "Tell me about the reporting requirements"
A: {"query_type": "vague_exploratory", "confidence": 0.80, "reasoning": "Vague; vocabulary gap between query and docs."}

Q: "What are the approval steps and also what happens when a vendor is rejected?"
A: {"query_type": "compound", "confidence": 0.93, "reasoning": "Two unrelated sub-questions in one query."}
""".strip()

_CLASSIFICATION_SYSTEM = (
    "You are a query classification assistant for a technical document retrieval system. "
    "Classify the user query into EXACTLY ONE of these types: "
    + ", ".join(QUERY_TYPES.keys())
    + ".\n\n"
    + _FEW_SHOT_EXAMPLES
    + "\n\nReturn a JSON object with keys: "
    "'query_type' (string), 'confidence' (float 0-1), 'reasoning' (brief string). "
    "No prose outside the JSON."
)


# ---------------------------------------------------------------------------
# Heuristic fallback
# ---------------------------------------------------------------------------

def _heuristic_classify(query: str) -> dict:
    """
    Fast rule-based classification used when the LLM is unavailable.

    Precedence (first match wins):
      1. multi_hop   — relational phrasing
      2. global_thematic — overview / summary phrasing
      3. compound    — multiple question marks or "and also"
      4. specific_factoid — "what is", "define", field codes
      5. hyde        — default fallback (vague_exploratory → hyde)
    """
    q = query.lower().strip()

    _MULTI_HOP_PATTERNS = [
        r"how does .+ relate",
        r"connection between",
        r"relationship between",
        r"how .+ affect",
        r"difference between .+ and",
        r"what happens when .+ and",
        r"link between",
    ]
    for pattern in _MULTI_HOP_PATTERNS:
        if re.search(pattern, q):
            return {
                "query_type": "multi_hop",
                "confidence":  0.70,
                "strategy":    STRATEGY_MAP["multi_hop"],
                "reasoning":   "heuristic: relational phrasing detected",
            }

    _GLOBAL_PATTERNS = [
        r"^overview",
        r"summary of",
        r"what are (all|the main|the key|the primary)",
        r"give me an overview",
        r"explain the (concept|system|process) of",
        r"describe the (workflow|process|stages)",
    ]
    for pattern in _GLOBAL_PATTERNS:
        if re.search(pattern, q):
            return {
                "query_type": "global_thematic",
                "confidence":  0.70,
                "strategy":    STRATEGY_MAP["global_thematic"],
                "reasoning":   "heuristic: overview/summary phrasing detected",
            }

    # Compound: multiple question marks or explicit "and also"
    if query.count("?") >= 2 or "and also" in q or " and how " in q:
        return {
            "query_type": "compound",
            "confidence":  0.72,
            "strategy":    STRATEGY_MAP["compound"],
            "reasoning":   "heuristic: multiple questions detected",
        }

    _FACTOID_PATTERNS = [
        r"^what is\b",
        r"^what are\b",
        r"^define\b",
        r"^what does .+ stand for",
        r"^which field",
        r"^what field",
        r"^what fields",
        r"^list the",
        r"^what error",
        r"^what code",
        r"^how do (i|you)\b",          # "how do I enter...", "how do you..."
        r"^how to\b",                   # "how to change..."
        r"^where is\b",                 # "where is the field..."
        r"^show me\b",                  # "show me screen..."
        r"\bsection\s+\d+",             # "section 4", "section 12"
        r"\b[A-Z]{2,5}-\d{3,}\b",      # identifier codes e.g. VND-2024
        r"\b(section|field|error code|procedure|step|form|requirement)\b",
    ]
    for pattern in _FACTOID_PATTERNS:
        if re.search(pattern, q):
            return {
                "query_type": "specific_factoid",
                "confidence":  0.68,
                "strategy":    STRATEGY_MAP["specific_factoid"],
                "reasoning":   "heuristic: factoid phrasing detected",
            }

    # Default fallback → hybrid with confidence just above LLM threshold
    # Most technical document queries are best served by hybrid search
    return {
        "query_type": "specific_factoid",
        "confidence":  0.66,
        "strategy":    STRATEGY_MAP["specific_factoid"],
        "reasoning":   "heuristic: default fallback to hybrid",
    }


# ---------------------------------------------------------------------------
# QueryRouter
# ---------------------------------------------------------------------------

class QueryRouter:
    """
    Classifies a retrieval query and maps it to the best retrieval strategy.

    Parameters
    ----------
    llm_client : LLMClient, optional
        If None a new LLMClient is instantiated with the fast model so that
        classification latency stays low (gemma3:latest by default).
    """

    def __init__(self, llm_client: Optional[LLMClient] = None) -> None:
        # Use the fast model for classification — it's latency-sensitive
        from .config import get_config  # noqa: PLC0415
        _fast = get_config().get("llm", {}).get("fast_model", "qwen2.5:7b")
        self._llm = llm_client or LLMClient(model=_fast)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, query: str) -> dict:
        """
        Classify *query* using LLM few-shot inference with heuristic fallback.

        Returns
        -------
        dict with keys:
          query_type : str   — one of the QUERY_TYPES keys
          confidence : float — 0–1
          strategy   : str   — one of the STRATEGY_MAP values
          reasoning  : str   — brief explanation
        """
        if not query or not query.strip():
            return {
                "query_type": "specific_factoid",
                "confidence":  1.0,
                "strategy":    "hybrid",
                "reasoning":   "empty query defaulted to hybrid",
            }

        # Heuristic first (sub-ms, no LLM cost)
        heuristic = _heuristic_classify(query)

        # Only call LLM when heuristic is uncertain (confidence < 0.65)
        if heuristic["confidence"] >= 0.65:
            log.debug("QueryRouter: heuristic confident (%.2f) → %s", heuristic["confidence"], heuristic["strategy"])
            return heuristic

        result = self._llm_classify(query)
        if result:
            return result

        return heuristic

    def get_strategy(self, query: str) -> str:
        """Convenience wrapper — returns only the strategy string."""
        return self.classify(query)["strategy"]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _llm_classify(self, query: str) -> Optional[dict]:
        """
        Ask the LLM to classify *query*.  Returns a complete classification
        dict on success, None on any failure.
        """
        prompt = f'Query to classify: "{query}"'
        try:
            data = self._llm.complete_json(prompt, system=_CLASSIFICATION_SYSTEM)
        except Exception as exc:
            log.warning("QueryRouter LLM call error: %s", exc)
            return None

        if not data:
            return None

        query_type = data.get("query_type", "").strip()
        if query_type not in QUERY_TYPES:
            log.debug(
                "QueryRouter: LLM returned unknown query_type=%r; falling back.", query_type
            )
            return None

        confidence = float(data.get("confidence", 0.5))
        strategy   = STRATEGY_MAP.get(query_type, "hybrid")
        reasoning  = str(data.get("reasoning", "")).strip()

        return {
            "query_type": query_type,
            "confidence":  round(min(max(confidence, 0.0), 1.0), 4),
            "strategy":    strategy,
            "reasoning":   reasoning,
        }
