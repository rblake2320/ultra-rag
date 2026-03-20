"""
Batch embedder with two providers:
  - local: sentence-transformers SentenceTransformer (NV-Embed-v2, GPU)
  - ollama: Ollama HTTP API (nomic-embed-text, fallback)

Provider is selected from config.yaml embedding.provider.
embed_collection() auto-routes based on config.

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

# Module-level singleton for the local sentence-transformers model
_model_cache: dict = {}


def _sanitize(text: str, max_chars: int = 6000) -> str:
    """Strip non-printable / problematic chars and truncate for embedding."""
    import unicodedata
    # Normalize unicode and replace non-printable chars with space
    text = unicodedata.normalize("NFKC", text)
    text = "".join(c if c.isprintable() or c in ("\n", "\t") else " " for c in text)
    # Collapse runs of spaces/underscores that appear from PDF noise
    import re
    text = re.sub(r"[_]{4,}", " ", text)
    text = re.sub(r" {3,}", "  ", text)
    return text[:max_chars].strip()


# ---------------------------------------------------------------------------
# Ollama provider
# ---------------------------------------------------------------------------

def _embed_batch(texts: list, ollama_url: str, model: str) -> list:
    """Call Ollama /api/embed for a batch of texts. Returns list of float lists."""
    clean = [_sanitize(t) for t in texts]
    resp = httpx.post(
        f"{ollama_url}/api/embed",
        json={"model": model, "input": clean},
        timeout=120.0,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("embeddings", [])


# ---------------------------------------------------------------------------
# Local sentence-transformers provider (NV-Embed-v2 and similar)
# ---------------------------------------------------------------------------

def _get_local_model(model_name: str):
    """Return cached SentenceTransformer instance for *model_name*."""
    if model_name not in _model_cache:
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
            import torch  # noqa: PLC0415

            device = "cuda" if torch.cuda.is_available() else "cpu"
            log.info("Embedder: loading local model '%s' on %s ...", model_name, device)

            # NV-Embed-v2 requires trust_remote_code for its custom pooling head
            model = SentenceTransformer(
                model_name,
                device=device,
                trust_remote_code=True,
            )
            _model_cache[model_name] = model
            log.info("Embedder: local model '%s' ready.", model_name)
        except Exception as exc:
            log.error("Embedder: failed to load local model '%s': %s", model_name, exc)
            raise

    return _model_cache[model_name]


def _embed_batch_local(texts: list, model_name: str, batch_size: int = 4) -> list:
    """
    Embed *texts* using a local sentence-transformers SentenceTransformer.

    NV-Embed-v2 uses a task prefix for asymmetric retrieval:
      query encoding : model.encode([query], prompt_name="query")
      passage encoding: model.encode([passage])  ← no prompt for documents

    At index time we always embed passages (no prompt).

    Returns list of float lists.
    """
    clean  = [_sanitize(t) for t in texts]
    model  = _get_local_model(model_name)

    # NV-Embed-v2 recommends normalize_embeddings=True for cosine similarity
    embeddings = model.encode(
        clean,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    # Returns numpy array (n, dim); convert to plain Python lists
    return [emb.tolist() for emb in embeddings]


# ---------------------------------------------------------------------------
# Provider-agnostic router
# ---------------------------------------------------------------------------

def get_embedder():
    """
    Return a callable (texts: list[str]) → list[list[float]] based on config.

    Reads config.yaml embedding.provider:
      'local'  → sentence-transformers (NV-Embed-v2, GPU)
      'ollama' → Ollama HTTP API (nomic-embed-text)
    """
    cfg      = get_config()["embedding"]
    provider = cfg.get("provider", "ollama").lower()
    model    = cfg["model"]
    batch_sz = cfg.get("batch_size", 4)

    if provider == "local":
        def _fn(texts: list) -> list:
            return _embed_batch_local(texts, model, batch_size=batch_sz)
    else:
        url = cfg["ollama_url"]
        def _fn(texts: list) -> list:
            return _embed_batch(texts, url, model)

    return _fn


# ---------------------------------------------------------------------------
# Main entry point: embed a whole collection
# ---------------------------------------------------------------------------

def embed_collection(conn, collection: str) -> int:
    """
    Embed all un-embedded chunks for a collection.
    At embed time: concatenates context_prefix + content for the vector.
    Stores ONLY the embedding — content and context_prefix stay separate.
    Returns number of chunks embedded.
    """
    cfg      = get_config()["embedding"]
    model    = cfg["model"]
    provider = cfg.get("provider", "ollama").lower()
    url      = cfg.get("ollama_url", "http://localhost:11434")
    batch_sz = cfg.get("batch_size", 4)
    dims     = cfg.get("dimensions", 768)

    log.info(
        "Embedder: provider=%s model=%s dims=%d batch_size=%d",
        provider, model, dims, batch_sz,
    )

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
    t0    = time.time()
    total = 0

    # Pre-load local model once (avoid repeated load per batch)
    if provider == "local":
        _get_local_model(model)

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
            if provider == "local":
                embeddings = _embed_batch_local(texts, model, batch_size=batch_sz)
            else:
                embeddings = _embed_batch(texts, url, model)
        except Exception as e:
            # Batch too large — fall back to one-at-a-time
            log.warning(f"Batch at index {batch_start} failed ({e}), retrying one-by-one...")
            embeddings = []
            for i, text in enumerate(texts):
                try:
                    if provider == "local":
                        emb = _embed_batch_local([text], model, batch_size=1)
                    else:
                        emb = _embed_batch([text], url, model)
                    embeddings.append(emb[0] if emb else None)
                except Exception as e2:
                    # Last resort: truncate hard to 3000 chars and retry
                    try:
                        short = _sanitize(text, max_chars=3000)
                        if provider == "local":
                            emb = _embed_batch_local([short], model, batch_size=1)
                        else:
                            emb = _embed_batch([short], url, model)
                        embeddings.append(emb[0] if emb else None)
                        log.warning(f"  Chunk id={ids[i]} embedded after truncation to 3000 chars")
                    except Exception as e3:
                        log.error(f"  Single embed failed for chunk id={ids[i]}: {e3}")
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
