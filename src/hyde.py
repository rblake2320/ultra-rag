"""
HyDE — Hypothetical Document Embedding.

Generates a plausible hypothetical answer passage for a query, embeds it,
and fuses the hypothesis-based vector search with a direct query vector
search via Reciprocal Rank Fusion (RRF).

This bridges the vocabulary gap between a user's vague/exploratory query
and the precise technical language used in the document corpus.

Typical use
-----------
    from src.hyde import HyDERetriever
    retriever = HyDERetriever(conn, collection="my-docs")
    results = retriever.search("tell me about discrepancy documentation", top_k=5)
"""
from __future__ import annotations

import logging
from typing import Optional

import psycopg2.extras

from .config import get_config
from .embedder import _embed_batch
from .llm import LLMClient

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_HYPOTHESIS_SYSTEM = (
    "You are a subject-matter expert writing technical documentation. "
    "Write a detailed, factual passage that directly answers the question. "
    "Write as if you are the authoritative documentation source. "
    "Do NOT hedge, say you don't know, or refuse. "
    "Write 200–400 words of plausible, specific, domain-accurate content. "
    "No preamble, no meta-commentary — just the passage itself."
)

_HYPOTHESIS_PROMPT_TEMPLATE = (
    "{domain_context}"
    "Question: {query}\n\n"
    "Passage:"
)


# ---------------------------------------------------------------------------
# Standalone RRF helper (also used by query_decomposer and reranker)
# ---------------------------------------------------------------------------

def rrf_merge(result_lists: list, k: int = 60, top_k: int = 10) -> list:
    """
    Reciprocal Rank Fusion across multiple result lists.

    Parameters
    ----------
    result_lists : list of lists
        Each inner list is a sequence of result dicts that must contain an
        'id' key.  Lists need not be the same length.
    k : int
        RRF constant (default 60, following the original paper).
    top_k : int
        Maximum number of results to return.

    Returns
    -------
    Merged, deduplicated list of result dicts, sorted by descending RRF score.
    Each dict gains a 'rrf_score' key.
    """
    scores: dict = {}       # id → cumulative RRF score
    rows:   dict = {}       # id → dict (first-seen wins for metadata)

    for result_list in result_lists:
        for rank, row in enumerate(result_list, start=1):
            rid = row["id"]
            scores[rid] = scores.get(rid, 0.0) + 1.0 / (k + rank)
            if rid not in rows:
                rows[rid] = dict(row)

    merged = sorted(rows.values(), key=lambda r: scores[r["id"]], reverse=True)
    for row in merged:
        row["rrf_score"] = round(scores[row["id"]], 8)

    return merged[:top_k]


# ---------------------------------------------------------------------------
# HyDERetriever
# ---------------------------------------------------------------------------

class HyDERetriever:
    """
    Hypothetical Document Embedding retriever.

    Parameters
    ----------
    conn : psycopg2 connection
        Active database connection (read-only queries only).
    collection : str
        Collection name used to scope chunk lookups.
    llm_client : LLMClient, optional
        Client used to generate hypothesis passages.  Defaults to a new
        LLMClient using the extraction model (deepseek-r1:32b) for quality.
    """

    def __init__(
        self,
        conn,
        collection: str,
        llm_client: Optional[LLMClient] = None,
    ) -> None:
        self._conn       = conn
        self._collection = collection
        self._llm        = llm_client or LLMClient()  # extraction-quality model

        cfg_emb          = get_config()["embedding"]
        self._ollama_url = cfg_emb["ollama_url"]
        self._emb_model  = cfg_emb["model"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_hypothesis(self, query: str, domain_hint: Optional[str] = None) -> str:
        """
        Generate a hypothetical passage that plausibly answers *query*.

        Parameters
        ----------
        query : str
            The user's natural language query.
        domain_hint : str, optional
            Short domain context injected before the question, e.g.
            "IMDS Air Force maintenance management system".

        Returns
        -------
        The generated passage as a string, or an empty string on failure.
        """
        domain_context = ""
        if domain_hint:
            domain_context = f"Domain context: {domain_hint}\n\n"

        prompt = _HYPOTHESIS_PROMPT_TEMPLATE.format(
            domain_context=domain_context,
            query=query.strip(),
        )

        try:
            hypothesis = self._llm.complete(
                prompt,
                system=_HYPOTHESIS_SYSTEM,
                max_tokens=512,
            )
        except Exception as exc:
            log.warning("HyDE hypothesis generation failed: %s", exc)
            return ""

        return hypothesis.strip()

    def search(
        self,
        query: str,
        top_k: int = 5,
        domain_hint: Optional[str] = None,
    ) -> list:
        """
        HyDE retrieval: fuses hypothesis-vector search with query-vector search.

        Steps
        -----
        1. Generate a hypothetical passage for *query*.
        2. Embed both the hypothesis and the original query.
        3. Run pgvector searches for each embedding.
        4. RRF-fuse the two result sets.

        Parameters
        ----------
        query : str
            User's natural language query.
        top_k : int
            Number of results to return after fusion.
        domain_hint : str, optional
            Forwarded to :meth:`generate_hypothesis`.

        Returns
        -------
        List of result dicts (see rag.chunks schema), each with a 'rrf_score'.
        Falls back to query-only vector search if hypothesis generation fails.
        """
        n_fetch = top_k * 4   # over-fetch before fusion

        # ── Embed query directly ─────────────────────────────────────
        try:
            query_embs = _embed_batch([query], self._ollama_url, self._emb_model)
            query_embedding = query_embs[0] if query_embs else None
        except Exception as exc:
            log.warning("HyDE: query embedding failed: %s", exc)
            query_embedding = None

        # ── Generate & embed hypothesis ──────────────────────────────
        hypothesis = self.generate_hypothesis(query, domain_hint=domain_hint)
        hypothesis_embedding = None

        if hypothesis:
            try:
                hyp_embs = _embed_batch([hypothesis], self._ollama_url, self._emb_model)
                hypothesis_embedding = hyp_embs[0] if hyp_embs else None
            except Exception as exc:
                log.warning("HyDE: hypothesis embedding failed: %s", exc)

        # ── Vector searches ──────────────────────────────────────────
        result_lists = []

        if hypothesis_embedding:
            hyp_results = self._vector_search_by_embedding(hypothesis_embedding, n_fetch)
            if hyp_results:
                result_lists.append(hyp_results)
                log.debug("HyDE: hypothesis search returned %d results", len(hyp_results))

        if query_embedding:
            q_results = self._vector_search_by_embedding(query_embedding, n_fetch)
            if q_results:
                result_lists.append(q_results)
                log.debug("HyDE: query-direct search returned %d results", len(q_results))

        if not result_lists:
            log.warning("HyDE: no vector results — returning empty list")
            return []

        # ── RRF fusion ───────────────────────────────────────────────
        merged = rrf_merge(result_lists, k=60, top_k=top_k)

        # Normalise score key to 'score' for consistency with search.py output
        for row in merged:
            row.setdefault("score", row.get("rrf_score", 0.0))

        return merged

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _vector_search_by_embedding(
        self, embedding: list, top_k: int
    ) -> list:
        """
        Run a pgvector cosine similarity search using a pre-computed embedding.

        Parameters
        ----------
        embedding : list of float
            Query (or hypothesis) embedding vector.
        top_k : int
            Maximum rows to return.

        Returns
        -------
        List of result dicts, or [] on error.
        """
        emb_str = "[" + ",".join(str(x) for x in embedding) + "]"
        try:
            with self._conn.cursor(
                cursor_factory=psycopg2.extras.RealDictCursor
            ) as cur:
                cur.execute(
                    """
                    SELECT
                        id,
                        content,
                        content_type,
                        context_prefix,
                        chunk_metadata,
                        token_count,
                        1 - (embedding <=> %s::vector) AS score
                    FROM rag.chunks
                    WHERE collection = %s
                      AND embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (emb_str, self._collection, emb_str, top_k),
                )
                rows = cur.fetchall()
            return [dict(r) for r in rows]
        except Exception as exc:
            log.error("HyDE _vector_search_by_embedding failed: %s", exc)
            return []
