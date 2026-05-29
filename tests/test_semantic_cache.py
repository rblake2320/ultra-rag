"""
Unit tests for src.semantic_cache.SemanticCache.

Hermetic: uses fakeredis (no live Redis server required). Covers:
  - tier-1 exact-text hit / miss
  - tier-2 semantic (cosine) hit above threshold, miss below
  - TTL expiry
  - invalidate_collection clears exact-tier entries (regression test for the
    bug where exact keys hash the collection inside the key and so were never
    matched by a glob scan)
  - collection isolation (one collection's invalidation never touches another)
"""
import importlib

import fakeredis
import pytest

import src.semantic_cache as sc


@pytest.fixture
def cache(monkeypatch):
    """A SemanticCache wired to an isolated fakeredis instance."""
    server = fakeredis.FakeServer()

    def _fake_from_url(url, **kwargs):
        kwargs.pop("decode_responses", None)
        return fakeredis.FakeStrictRedis(server=server, decode_responses=False)

    # Force the cache on regardless of the ULTRA_RAG_CACHE env var, and make it
    # build its client from our fake.
    monkeypatch.setattr(sc, "_CACHE_ENABLED", True)
    monkeypatch.setattr("redis.from_url", _fake_from_url, raising=False)
    # config may be absent in a bare checkout; default to {} so __init__ uses defaults.
    monkeypatch.setattr(sc, "get_config", lambda: {"cache": {}}, raising=False)

    c = sc.SemanticCache()
    assert c.enabled, "fakeredis-backed cache should report enabled"
    return c


RESULT = {"answer": "42", "results": [{"id": 1}], "quality_score": 0.9}


def test_tier1_exact_hit_and_miss(cache):
    cache.set("What is the answer?", "col", RESULT)
    assert cache.get("What is the answer?", "col") == RESULT
    # Normalization: case/whitespace-insensitive exact match.
    assert cache.get("  what is THE answer?  ", "col") == RESULT
    # Different collection must not hit.
    assert cache.get("What is the answer?", "other") is None
    # Different query must not hit.
    assert cache.get("Unrelated question", "col") is None


def test_tier2_semantic_hit_and_miss(cache):
    emb = [1.0, 0.0, 0.0, 0.0]
    cache.set("colors of the sky", "col", RESULT, embedding=emb)
    # A near-identical vector (cosine ~0.9998) should hit at default threshold 0.96.
    near = [0.98, 0.02, 0.0, 0.0]
    hit = cache.get("totally different wording", "col", embedding=near)
    assert hit == RESULT
    # An orthogonal vector should miss.
    far = [0.0, 1.0, 0.0, 0.0]
    assert cache.get("totally different wording", "col", embedding=far) is None


def test_ttl_expiry(cache):
    import time as _t
    cache.set("ephemeral", "col", RESULT)
    assert cache.get("ephemeral", "col") == RESULT
    # Force the entry to lapse; fakeredis honors key TTL.
    cache._client.pexpire(sc._norm_key("ephemeral", "col"), 1)
    _t.sleep(0.05)
    assert cache.get("ephemeral", "col") is None


def test_invalidate_clears_exact_tier(cache):
    """Regression: invalidate_collection must clear exact-tier keys."""
    emb = [0.1, 0.2, 0.3, 0.4]
    cache.set("q1", "docs", RESULT, embedding=emb)
    cache.set("q2", "docs", RESULT)  # exact-tier only, no embedding
    assert cache.get("q1", "docs") == RESULT
    assert cache.get("q2", "docs") == RESULT

    cache.invalidate_collection("docs")

    # Both exact-tier entries gone (this is what the old glob-only code missed).
    assert cache.get("q1", "docs") is None
    assert cache.get("q2", "docs") is None
    # Semantic tier gone too.
    assert cache.get("anything", "docs", embedding=emb) is None


def test_invalidate_is_collection_scoped(cache):
    cache.set("shared question", "alpha", RESULT)
    cache.set("shared question", "beta", RESULT)
    cache.invalidate_collection("alpha")
    assert cache.get("shared question", "alpha") is None
    assert cache.get("shared question", "beta") == RESULT


def test_disabled_cache_is_noop(monkeypatch):
    monkeypatch.setattr(sc, "_CACHE_ENABLED", False)
    c = sc.SemanticCache()
    assert not c.enabled
    c.set("q", "col", RESULT)            # no-op, must not raise
    assert c.get("q", "col") is None
    c.invalidate_collection("col")       # no-op, must not raise


def test_cosine_helper():
    assert sc._cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert sc._cosine([1, 0], [0, 1]) == pytest.approx(0.0)
    assert sc._cosine([0, 0], [1, 1]) == 0.0  # zero-magnitude guard
