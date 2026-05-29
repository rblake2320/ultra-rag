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
import os
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

def _patch_nv_embed_v2(st_model) -> None:
    """
    Monkey-patch NV-Embed-v2 custom classes for transformers >=4.47 + sentence-transformers 5.x.

    Issues fixed:
      1. BidirectionalMistralModel: get_usable_length() removed → force use_cache=False
      2. BidirectionalMistralModel: MistralDecoderLayer now needs position_embeddings kwarg
      3. BidirectionalMistralModel: MistralDecoderLayer now returns bare tensor (not tuple)
      4. NVEmbedModel: forward() returns {"sentence_embeddings": x} — incompatible with
         sentence-transformers Transformer.forward() which expects outputs[0] = last_hidden_state
         (a [batch, seq_len, dim] tensor before pooling).

    Applied at class level after model load so on-disk files (trust_remote_code hash-checked) remain unchanged.
    """
    import torch
    from transformers.modeling_outputs import BaseModelOutputWithPast
    from transformers.modeling_attn_mask_utils import (
        _prepare_4d_attention_mask,
        _prepare_4d_attention_mask_for_sdpa,
    )

    # Find class objects
    bidirectional_cls = None
    nv_embed_cls = None
    for module in st_model.modules():
        cls = type(module)
        if cls.__name__ == "BidirectionalMistralModel":
            bidirectional_cls = cls
        elif cls.__name__ == "NVEmbedModel":
            nv_embed_cls = cls
        if bidirectional_cls and nv_embed_cls:
            break

    # ── Patch 1: BidirectionalMistralModel.forward ──────────────────────────
    if bidirectional_cls is None:
        log.debug("_patch_nv_embed_v2: BidirectionalMistralModel not found; skip")
    else:
        def _bidirectional_forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        ):
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            use_cache = False  # embedding model; no generation cache needed
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            if input_ids is not None and inputs_embeds is not None:
                raise ValueError("Cannot specify both input_ids and inputs_embeds")
            elif input_ids is not None:
                batch_size, seq_length = input_ids.shape
            elif inputs_embeds is not None:
                batch_size, seq_length, _ = inputs_embeds.shape
            else:
                raise ValueError("Must specify either input_ids or inputs_embeds")

            if position_ids is None:
                device = input_ids.device if input_ids is not None else inputs_embeds.device
                position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
                position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
            else:
                position_ids = position_ids.view(-1, seq_length).long()

            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)

            if self._attn_implementation == "flash_attention_2":
                attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
            elif self._attn_implementation == "sdpa" and not output_attentions:
                attention_mask = _prepare_4d_attention_mask_for_sdpa(attention_mask, inputs_embeds.dtype)
            else:
                attention_mask = _prepare_4d_attention_mask(attention_mask, inputs_embeds.dtype)

            hidden_states = inputs_embeds
            all_hidden_states = () if output_hidden_states else None
            all_self_attns = () if output_attentions else None

            position_embeddings = self.rotary_emb(hidden_states, position_ids)

            for decoder_layer in self.layers:
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=None,
                    output_attentions=output_attentions,
                    use_cache=False,
                    position_embeddings=position_embeddings,
                )
                if isinstance(layer_outputs, tuple):
                    hidden_states = layer_outputs[0]
                    if output_attentions and len(layer_outputs) > 1:
                        all_self_attns += (layer_outputs[1],)
                else:
                    hidden_states = layer_outputs

            hidden_states = self.norm(hidden_states)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if not return_dict:
                return tuple(v for v in [hidden_states, None, all_hidden_states, all_self_attns] if v is not None)
            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=None,
                hidden_states=all_hidden_states,
                attentions=all_self_attns,
            )

        bidirectional_cls.forward = _bidirectional_forward
        log.info("Embedder: BidirectionalMistralModel.forward patched for transformers >=4.47.")

    # ── Patch 2: NVEmbedModel.forward → return BaseModelOutput ──────────────
    # sentence_transformers Transformer.forward() does outputs[0] expecting last_hidden_state.
    # NVEmbedModel.forward() returns {"sentence_embeddings": pooled} — incompatible.
    # Fix: return BaseModelOutputWithPast with last_hidden_state = unsqueezed pooled embeds
    # so that the subsequent Pooling module (mean over seq_len=1) is a no-op.
    if nv_embed_cls is None:
        log.debug("_patch_nv_embed_v2: NVEmbedModel not found; skip")
    else:
        def _nv_embed_forward(self, input_ids, attention_mask, pool_mask=None, return_dict=True):
            from contextlib import nullcontext
            autocast_ctx = torch.autocast if torch.cuda.is_available() else nullcontext
            with autocast_ctx("cuda"):
                outputs = self.embedding_model(input_ids=input_ids, attention_mask=attention_mask)
                embeds = self.latent_attention_model(outputs.last_hidden_state, pool_mask)
            # embeds shape: [batch, 1, dim] or [batch, dim]
            if embeds.dim() == 2:
                embeds = embeds.unsqueeze(1)  # → [batch, 1, dim] for Pooling mean
            if not return_dict:
                return (embeds,)
            return BaseModelOutputWithPast(last_hidden_state=embeds)

        nv_embed_cls.forward = _nv_embed_forward
        log.info("Embedder: NVEmbedModel.forward patched to return BaseModelOutputWithPast.")


def _get_local_model(model_name: str):
    """Return cached SentenceTransformer instance for *model_name*."""
    if model_name not in _model_cache:
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
            import torch  # noqa: PLC0415

            # Redirect HuggingFace downloads to D: to avoid filling C: (nearly full)
            os.environ.setdefault("HF_HOME", r"D:\huggingface_cache")

            device = "cuda" if torch.cuda.is_available() else "cpu"
            log.info("Embedder: loading local model '%s' on %s ...", model_name, device)

            # NV-Embed-v2 requires trust_remote_code for its custom pooling head.
            # Load in fp16 to keep ~15GB VRAM footprint (fp32 would be ~31GB, exceeds RTX 5090).
            # truncate_dim=1024: NV-Embed-v2 Matryoshka Representation Learning (MRL) truncation.
            # pgvector 0.8.0 HNSW index supports ≤2000 dims for vector type; 4096 would fail.
            # 1024-dim NV-Embed-v2 still outperforms nomic-embed-text 768 on MTEB benchmarks.
            model = SentenceTransformer(
                model_name,
                device=device,
                trust_remote_code=True,
                model_kwargs={"torch_dtype": torch.float16},
                truncate_dim=1024,
            )

            # Patch NV-Embed-v2 BidirectionalMistralModel for transformers >=4.47 compatibility.
            # Newer transformers changed: (1) DynamicCache no longer has get_usable_length,
            # (2) MistralDecoderLayer.forward() requires position_embeddings kwarg,
            # (3) MistralDecoderLayer.forward() now returns bare tensor (not tuple).
            # We patch BidirectionalMistralModel.forward at class level after loading.
            _patch_nv_embed_v2(model)

            _model_cache[model_name] = model
            log.info("Embedder: local model '%s' ready.", model_name)
        except Exception as exc:
            log.error("Embedder: failed to load local model '%s': %s", model_name, exc)
            raise

    return _model_cache[model_name]


def _embed_batch_local(texts: list, model_name: str, batch_size: int = 4,
                       is_query: bool = False) -> list:
    """
    Embed *texts* using a local sentence-transformers SentenceTransformer.

    NV-Embed-v2 is an asymmetric model:
      query encoding : model.encode([query], prompt_name="query")
      passage encoding: model.encode([passage])  ← no prompt for documents

    Pass is_query=True when encoding search queries so the model applies
    its "query" task prefix; leave False (default) for document/passage indexing.

    Returns list of float lists.
    """
    clean  = [_sanitize(t) for t in texts]
    model  = _get_local_model(model_name)

    encode_kwargs = dict(
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    if is_query:
        encode_kwargs["prompt_name"] = "query"

    # Returns numpy array (n, dim); convert to plain Python lists
    embeddings = model.encode(clean, **encode_kwargs)
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
