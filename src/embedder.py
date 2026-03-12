"""
Synchronous batch embedder via Ollama nomic-embed-text.
Embeds context_prefix + content for richer retrieval vectors.
Updates rag.chunks.embedding in-place.
"""
import logging
import time
from typing import Optional

import httpx
import psycopg2.extras

from .config import get_config

log = logging.getLogger(__name__)


def _embed_batch(texts: list, ollama_url: str, model: str) -> list:
    """Call Ollama /api/embed for a batch of texts. Returns list of float lists."""
    resp = httpx.post(
        f"{ollama_url}/api/embed",
        json={"model": model, "input": texts},
        timeout=120.0,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("embeddings", [])


def embed_collection(conn, collection: str) -> int:
    """
    Embed all un-embedded chunks for a collection.
    At embed time: concatenates context_prefix + content for the vector.
    Stores ONLY the embedding — content and context_prefix stay separate.
    Returns number of chunks embedded.
    """
    cfg      = get_config()["embedding"]
    model    = cfg["model"]
    url      = cfg["ollama_url"]
    batch_sz = cfg.get("batch_size", 32)
    dims     = cfg.get("dimensions", 768)

    # Fetch un-embedded chunks
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, content, context_prefix
            FROM rag.chunks
            WHERE collection = %s AND embedding IS NULL
            ORDER BY id
        """, (collection,))
        rows = cur.fetchall()

    if not rows:
        log.info(f"No un-embedded chunks for collection '{collection}'")
        return 0

    log.info(f"Embedding {len(rows)} chunks for collection '{collection}'...")
    t0 = time.time()
    total = 0

    for batch_start in range(0, len(rows), batch_sz):
        batch = rows[batch_start: batch_start + batch_sz]
        ids   = [r[0] for r in batch]

        # Build embed inputs: context_prefix + "\n\n" + content
        texts = []
        for _, content, ctx in batch:
            if ctx:
                texts.append(f"{ctx}\n\n{content}")
            else:
                texts.append(content)

        try:
            embeddings = _embed_batch(texts, url, model)
        except Exception as e:
            # Batch too large — fall back to one-at-a-time
            log.warning(f"Batch at index {batch_start} failed ({e}), retrying one-by-one...")
            embeddings = []
            for i, text in enumerate(texts):
                try:
                    emb = _embed_batch([text], url, model)
                    embeddings.append(emb[0] if emb else None)
                except Exception as e2:
                    log.error(f"  Single embed failed for chunk id={ids[i]}: {e2}")
                    embeddings.append(None)

        if len(embeddings) != len(batch):
            log.warning(f"Expected {len(batch)} embeddings, got {len(embeddings)}")
            continue

        # Write back (skip any None embeddings from fallback failures)
        updates = [
            (emb, model, row_id)
            for emb, row_id in zip(embeddings, ids)
            if emb is not None
        ]
        if updates:
            with conn.cursor() as cur:
                psycopg2.extras.execute_batch(cur, """
                    UPDATE rag.chunks
                    SET embedding = %s::vector, embed_model = %s
                    WHERE id = %s
                """, updates)
            conn.commit()
            total += len(updates)

        if (batch_start // batch_sz) % 10 == 0:
            elapsed = time.time() - t0
            log.info(f"  {total}/{len(rows)} embedded ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    log.info(f"Embedded {total} chunks in {elapsed:.1f}s")
    return total
