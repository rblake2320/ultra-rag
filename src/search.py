"""
Hybrid search: pgvector cosine + tsvector full-text → RRF fusion → NIM re-ranker.
"""
import hashlib
import logging
from functools import lru_cache
from typing import Optional

import httpx
import psycopg2.extras

from .config import get_config
from .embedder import _embed_batch

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Query embedding cache — avoids repeated 2.2s Ollama calls for same query
# ---------------------------------------------------------------------------

@lru_cache(maxsize=512)
def _cached_embed(query: str, model: str, ollama_url: str) -> tuple:
    """Embed a query string, caching results by (query, model) key."""
    emb = _embed_batch([query], ollama_url, model)
    return tuple(emb[0]) if emb else ()


def _get_query_embedding(query: str) -> list:
    """Return embedding for query, using cache when available."""
    cfg_emb = get_config()["embedding"]
    result = _cached_embed(query.strip(), cfg_emb["model"], cfg_emb["ollama_url"])
    return list(result) if result else []


def _rrf_score(rank: int, k: int) -> float:
    return 1.0 / (k + rank)


def search(conn, query: str, collection: str,
           top_k: int = 5,
           content_type: Optional[str] = None,
           metadata_filter: Optional[dict] = None,
           force_tier: Optional[int] = None) -> list:
    """
    Hybrid search returning top_k results.

    Args:
        conn:            psycopg2 connection
        query:           natural language query
        collection:      collection name (e.g. 'imds')
        top_k:           final results to return
        content_type:    filter by content_type (optional)
        metadata_filter: JSONB containment filter e.g. {'imds_screens': ['207']}
        force_tier:      1=keyword only, 2=vector only, 3=both+rerank

    Returns:
        list of dicts: {id, content, content_type, context_prefix,
                        chunk_metadata, token_count, score, source}
    """
    cfg     = get_config()["search"]
    k_vec   = cfg.get("rrf_k_vector", 60)
    k_kw    = cfg.get("rrf_k_keyword", 40)
    n_fetch = top_k * 4  # fetch more before RRF

    # Build WHERE clause
    where_parts = ["collection = %s"]
    where_vals  = [collection]
    if content_type:
        where_parts.append("content_type = %s")
        where_vals.append(content_type)
    if metadata_filter:
        import json
        where_parts.append("chunk_metadata @> %s::jsonb")
        where_vals.append(json.dumps(metadata_filter))
    where_clause = " AND ".join(where_parts)

    # ── Keyword search (tsvector) ─────────────────────────────────────
    def _keyword_search() -> list:
        query_tsq = " | ".join(query.split())  # simple OR of words
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(f"""
                SELECT id, content, content_type, context_prefix,
                       chunk_metadata, token_count,
                       ts_rank(content_tsv, plainto_tsquery('english', %s)) AS kw_score
                FROM rag.chunks
                WHERE {where_clause}
                  AND content_tsv @@ plainto_tsquery('english', %s)
                ORDER BY kw_score DESC
                LIMIT %s
            """, [query] + where_vals + [query, n_fetch])
            return cur.fetchall()

    # ── Vector search (pgvector cosine) ──────────────────────────────
    def _vector_search(query_embedding: list) -> list:
        emb_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(f"""
                SELECT id, content, content_type, context_prefix,
                       chunk_metadata, token_count,
                       1 - (embedding <=> %s::vector) AS vec_score
                FROM rag.chunks
                WHERE {where_clause}
                  AND embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, [emb_str] + where_vals + [emb_str, n_fetch])
            return cur.fetchall()

    # ── Run search tiers ─────────────────────────────────────────────
    kw_results  = []
    vec_results = []

    if force_tier == 1:
        kw_results = _keyword_search()
    elif force_tier == 2:
        emb = _get_query_embedding(query)
        if emb:
            vec_results = _vector_search(emb)
    else:
        # Tier 3 (default): both
        kw_results = _keyword_search()
        try:
            emb = _get_query_embedding(query)
            if emb:
                vec_results = _vector_search(emb)
        except Exception as e:
            log.warning(f"Vector search failed: {e}; falling back to keyword only")

    # ── RRF Fusion ───────────────────────────────────────────────────
    rrf_scores: dict[int, float] = {}

    for rank, row in enumerate(kw_results, 1):
        rrf_scores[row["id"]] = rrf_scores.get(row["id"], 0) + _rrf_score(rank, k_kw)
    for rank, row in enumerate(vec_results, 1):
        rrf_scores[row["id"]] = rrf_scores.get(row["id"], 0) + _rrf_score(rank, k_vec)

    # Build merged result set
    all_rows: dict[int, dict] = {}
    for row in kw_results + vec_results:
        if row["id"] not in all_rows:
            all_rows[row["id"]] = dict(row)

    merged = sorted(all_rows.values(), key=lambda r: rrf_scores.get(r["id"], 0), reverse=True)
    merged = merged[:n_fetch]

    # ── NIM Re-ranker (optional) ──────────────────────────────────────
    if len(merged) > top_k and force_tier != 1 and force_tier != 2:
        try:
            merged = _nim_rerank(query, merged, top_k, cfg)
        except Exception as e:
            log.warning(f"NIM re-ranker unavailable: {e}; using RRF order")
            merged = merged[:top_k]
    else:
        merged = merged[:top_k]

    # Attach scores
    for row in merged:
        row["score"] = round(rrf_scores.get(row["id"], 0.0), 6)

    return merged


def _nim_rerank(query: str, candidates: list, top_k: int, cfg: dict) -> list:
    """Call NVIDIA NIM re-ranker, return top_k re-ordered results."""
    api_key  = cfg.get("nvidia_api_key", "")
    endpoint = cfg.get("reranker_endpoint", "")
    model    = cfg.get("reranker_model", "nvidia/nv-rerankqa-mistral-4b-v3")

    if not api_key or not endpoint:
        raise ValueError("NVIDIA_API_KEY or reranker_endpoint not configured")

    passages = [{"text": r["content"][:2000]} for r in candidates]
    resp = httpx.post(
        endpoint,
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": model, "query": {"text": query}, "passages": passages},
        timeout=30.0,
    )
    resp.raise_for_status()
    rankings = resp.json().get("rankings", [])
    # rankings: [{index, logit}, ...] sorted by relevance
    top_indices = [r["index"] for r in rankings[:top_k]]
    return [candidates[i] for i in top_indices if i < len(candidates)]
