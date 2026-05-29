"""
Semantic query cache backed by Redis.

Two-tier strategy:
  Tier 1 (fast, ~0ms): exact normalized-text match — catches repeated identical queries.
  Tier 2 (~2ms):       cosine-similarity match against recent embedded queries stored in
                       Redis — catches paraphrases and near-duplicates above a threshold.

Configuration (via get_config()["cache"]):
  enabled:   true/false (default true; disable with ULTRA_RAG_CACHE=false)
  redis_url: redis://localhost:6379 (default)
  ttl:       3600 seconds (default)
  threshold: 0.96 (semantic similarity hit threshold)
  max_candidates: 100 (max recent queries checked for semantic similarity)

Usage:
    from src.semantic_cache import SemanticCache
    cache = SemanticCache()
    hit = cache.get(query, collection, embedding)
    if hit:
        return hit
    result = run_pipeline(...)
    cache.set(query, collection, embedding, result)
"""
import hashlib
import json
import logging
import os
from typing import List, Optional

log = logging.getLogger(__name__)

_CACHE_ENABLED = os.environ.get("ULTRA_RAG_CACHE", "true").lower() == "true"


def _cosine(a: List[float], b: List[float]) -> float:
    """Fast pure-Python cosine similarity for short float lists."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = sum(x * x for x in a) ** 0.5
    mag_b = sum(x * x for x in b) ** 0.5
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _norm_key(query: str, collection: str) -> str:
    """Deterministic cache key from normalized query text."""
    text = f"{collection}:{query.strip().lower()}"
    return "rag:cache:exact:" + hashlib.sha256(text.encode()).hexdigest()


def _emb_index_key(collection: str) -> str:
    return f"rag:cache:emb_index:{collection}"


def _result_key(hit_hash: str) -> str:
    return f"rag:cache:result:{hit_hash}"


def _keyset_key(collection: str) -> str:
    """Set tracking every cache key written for a collection (for invalidation).

    Exact-tier keys hash the collection name inside the key, so they cannot be
    matched by a glob scan. We record every key we write here so a collection can
    be invalidated exactly after an ingest.
    """
    return f"rag:cache:keyset:{collection}"


class SemanticCache:
    """
    Redis-backed two-tier semantic cache for RAG search results.

    Not thread-safe for the same Redis client object across threads;
    each thread should instantiate its own or use thread-local storage.
    """

    def __init__(self):
        if not _CACHE_ENABLED:
            self._client = None
            return
        try:
            import redis  # noqa: PLC0415
            from .config import get_config  # noqa: PLC0415
            cfg = get_config().get("cache", {})
            url = cfg.get("redis_url", os.environ.get("REDIS_URL", "redis://localhost:6379"))
            self._ttl = int(cfg.get("ttl", 3600))
            self._threshold = float(cfg.get("threshold", 0.96))
            self._max_candidates = int(cfg.get("max_candidates", 100))
            self._client = redis.from_url(url, decode_responses=False, socket_connect_timeout=1)
            self._client.ping()
            log.debug("SemanticCache connected to Redis at %s", url)
        except Exception as exc:
            log.warning("SemanticCache: Redis unavailable (%s) — caching disabled.", exc)
            self._client = None

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def get(self, query: str, collection: str,
            embedding: Optional[List[float]] = None) -> Optional[dict]:
        """
        Return cached result dict if a cache hit is found, else None.

        Tier 1: exact text hash lookup (O(1)).
        Tier 2: cosine similarity against stored embeddings (O(max_candidates)).
        """
        if not self.enabled:
            return None
        try:
            # Tier 1: exact text match
            exact_key = _norm_key(query, collection)
            raw = self._client.get(exact_key)
            if raw:
                log.debug("SemanticCache: tier-1 HIT for query=%.50s", query)
                return json.loads(raw)

            # Tier 2: semantic similarity (only if embedding provided)
            if embedding is None:
                return None

            index_key = _emb_index_key(collection)
            # Stored as list of JSON: [{hash, emb}]
            entries_raw = self._client.lrange(index_key, 0, self._max_candidates - 1)
            for entry_raw in entries_raw:
                try:
                    entry = json.loads(entry_raw)
                    sim = _cosine(embedding, entry["emb"])
                    if sim >= self._threshold:
                        result_raw = self._client.get(_result_key(entry["hash"]))
                        if result_raw:
                            log.debug(
                                "SemanticCache: tier-2 HIT (sim=%.3f) for query=%.50s",
                                sim, query,
                            )
                            return json.loads(result_raw)
                except Exception:
                    continue
        except Exception as exc:
            log.debug("SemanticCache.get failed (non-fatal): %s", exc)
        return None

    def set(self, query: str, collection: str, result: dict,
            embedding: Optional[List[float]] = None) -> None:
        """Store a result under the given query + optional embedding."""
        if not self.enabled:
            return
        try:
            keyset_key = _keyset_key(collection)
            # Tier 1: exact text key
            exact_key = _norm_key(query, collection)
            self._client.setex(exact_key, self._ttl, json.dumps(result))
            self._client.sadd(keyset_key, exact_key)

            # Tier 2: store embedding index entry
            if embedding is not None:
                emb_hash = hashlib.sha256(
                    json.dumps(embedding[:8], separators=(",", ":")).encode()
                ).hexdigest()[:16]
                entry = json.dumps({"hash": emb_hash, "emb": embedding})
                index_key = _emb_index_key(collection)
                result_key = _result_key(emb_hash)
                self._client.lpush(index_key, entry)
                self._client.ltrim(index_key, 0, self._max_candidates - 1)
                self._client.expire(index_key, self._ttl * 2)
                self._client.setex(result_key, self._ttl, json.dumps(result))
                self._client.sadd(keyset_key, index_key, result_key)

            # Keep the keyset alive at least as long as the longest-lived entry.
            self._client.expire(keyset_key, self._ttl * 2)
        except Exception as exc:
            log.debug("SemanticCache.set failed (non-fatal): %s", exc)

    def invalidate_collection(self, collection: str) -> None:
        """Clear all cached results for a collection (call after ingest)."""
        if not self.enabled:
            return
        try:
            keyset_key = _keyset_key(collection)
            # Tracked keys (exact-tier, emb-index, result) — exact-tier keys hash the
            # collection inside the key, so a glob scan alone would miss them.
            tracked = self._client.smembers(keyset_key)
            keys = {k for k in tracked}
            # Belt-and-suspenders: also sweep glob-matchable keys (emb_index/result
            # written before keyset tracking existed).
            keys.update(self._client.scan_iter(f"rag:cache:emb_index:{collection}", count=100))
            keys.add(keyset_key)
            if keys:
                self._client.delete(*keys)
                log.info("SemanticCache: invalidated %d key(s) for collection=%s", len(keys), collection)
        except Exception as exc:
            log.debug("SemanticCache.invalidate_collection failed (non-fatal): %s", exc)
