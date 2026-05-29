# Ultra RAG

A production-grade RAG system that combines knowledge-graph retrieval, RAPTOR hierarchical
summarisation, contextual chunking, hybrid search, late-interaction re-ranking, corrective
retrieval (CRAG), Self-RAG adaptive filtering, and grounded LLM synthesis into a single
measurable, benchmarked pipeline.

---

## Architecture

Every `/api/search` request passes through this ordered pipeline:

```
Query
  │
  ├─ [0] Semantic cache (Redis, 2-tier: exact hash + cosine similarity)
  │       ↳ cache hit → return immediately
  │
  ├─ [1] Query router  (LLM-based: hybrid / kg_local / kg_global / multi_hop / hyde / compound)
  │
  ├─ [2] Retrieval  ──────────────────────────────────────────────────────────
  │       ├─ Vector search    (NV-Embed-v2 1024-d, pgvector HNSW cosine)
  │       ├─ BM25 keyword     (PostgreSQL TSVector + ts_rank)
  │       ├─ ColBERT          (late-interaction token-level, optional)
  │       └─ RRF fusion       (Reciprocal Rank Fusion, k_vec=60 k_kw=40)
  │
  ├─ [3] Retrieval memory     (chunk utility tracking, EMA boost)
  │
  ├─ [4] Cross-encoder rerank (BAAI/bge-reranker-v2-m3 local GPU; NIM API fallback)
  │
  ├─ [5] CRAG quality gate    (evaluate → use_as_is / supplement / rewrite_query)
  │
  ├─ [6] Self-RAG filter      (adaptive relevance filtering)
  │
  ├─ [7] Utility score boost  (learned per-chunk helpfulness)
  │
  ├─ [8] Parent chunk expand  (return full section around each chunk)
  │
  ├─ [9] Grounded synthesis   (LLM answer citing sources [1]..[n], synthesize=False to skip)
  │
  └─ Answer + ranked chunks + provenance chain + quality score
```

### Strategy dispatch

| Strategy | When used | How it retrieves |
|---|---|---|
| `hybrid` | Default (and fallback) | Vector + BM25 + RRF |
| `kg_local` | Entity-centric questions | KG neighborhood + Personalized PageRank |
| `kg_global` | Broad thematic questions | Community summary search |
| `multi_hop` | Multi-step reasoning | Query decomposition → sub-retrieval → merge |
| `hyde` | Abstract / open-ended | Hypothetical document embedding |
| `compound` | Complex mixed | Decompose + per-sub-query routing + merge |

---

## Ingest pipeline

Documents → parse → contextual chunking → KG extraction → RAPTOR hierarchical summaries
→ community detection → NV-Embed-v2 embedding → ColBERT indexing → pgvector HNSW

Contextual chunking (Anthropic-style) prepends a document-level context prefix to each
chunk before embedding, so retrieved chunks carry surrounding context at query time.

---

## Evaluation

```bash
# Generate synthetic Q&A for a collection, then evaluate
python ultra_eval.py imds --generate 50 --run --run-name v1

# Standard public benchmark (HotpotQA distractor set, CC BY-SA 4.0)
python scripts/benchmark_dataset.py --setup --n 200 --eval
```

Four RAGAS-style metrics are computed end-to-end against the synthesized answer:

| Metric | Measures | Threshold |
|---|---|---|
| Faithfulness | Answer grounded in retrieved context? | ≥ 0.70 |
| Answer relevancy | Answer addresses the question? | higher is better |
| Context precision | Retrieved chunks relevant to ground truth? | higher is better |
| Context recall | Context covers all ground-truth statements? | ≥ 0.60 |

### Baseline → current results (HotpotQA distractor, 200 QA pairs)

| Run | Embedding | Synthesis | Faithfulness | Answer rel. | Ctx precision | Ctx recall |
|---|---|---|---|---|---|---|
| baseline-nomic-768 | nomic-embed-text 768-d | none | 0.37 | 0.64 | 0.35 | 0.50 |
| nv-embed-v2 (private IMDS) | NV-Embed-v2 1024-d | none | 0.46 | 0.45 | 0.60 | 0.66 |
| **current** | NV-Embed-v2 1024-d | grounded [n] citations | *run `benchmark_dataset.py --eval`* | | | |

The current configuration uses NV-Embed-v2 (1024-d MRL truncation, MTEB rank #1 at time of
integration), a cross-encoder reranker on GPU, and a grounded synthesis step that cites source
indices inline — directly targeting faithfulness.

---

## Stack

| Component | What |
|---|---|
| **PostgreSQL 16** + pgvector | Vector store, BM25, provenance, eval history. Port 5434 on Windows (native service). |
| **NV-Embed-v2** | 7.8B-param SOTA embedding model, 1024-d MRL truncation (fits pgvector HNSW ≤ 2000-d limit). Local inference on GPU. |
| **BAAI/bge-reranker-v2-m3** | Cross-encoder reranker, GPU local, with NVIDIA NIM API fallback. |
| **Ollama** | LLM backend for synthesis, routing, CRAG, Self-RAG (qwen2.5:7b fast / qwen2.5-coder:32b heavy). |
| **Redis** | Semantic query cache (exact hash tier + cosine similarity tier, per-collection TTL). |
| **FastAPI** | REST API on :8300. Multi-tenant RLS via `rag_app` Postgres role. |
| **ColBERT** (optional) | Late-interaction retrieval via RAGatouille. Set `colbert_enabled: true` after `ultra_ingest.py --stages colbert`. |

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements_ultra.txt

# 2. Copy and fill environment file
cp .env.example .env
# Set RAG_DB_PASSWORD (postgres@5434), NVIDIA_API_KEY, ULTRA_RAG_API_KEY

# 3. Start server
python ultra_server.py        # :8300

# 4. Ingest documents
python ultra_ingest.py my-collection --stages all

# 5. Query
python ultra_query.py "What is X?" --collection my-collection

# 6. Evaluate
python ultra_eval.py my-collection --generate 50 --run
```

### .env.example

```
RAG_DB_PASSWORD=your_postgres_password
NVIDIA_API_KEY=your_nvidia_nim_api_key
ULTRA_RAG_API_KEY=your_admin_api_key
ULTRA_RAG_KEY_<COLLECTION>=per_collection_key
ULTRA_RAG_CACHE=true       # set false to disable Redis semantic cache
REDIS_URL=redis://localhost:6379
```

---

## Tests

```bash
pytest tests/          # hermetic (fakeredis, no live DB required for cache tests)
```

---

## Adversarial blind-spot testing

On server startup, `AdversarialTester` (`src/adversarial.py`) is scheduled for each active
collection. Every 24 hours (configurable via `adversarial.schedule_hours` in `config.yaml`)
it runs synthetic adversarial queries, identifies low-recall blind spots, and auto-heals
affected chunks via re-embedding. Reports written to `logs/adversarial_*.json`.

---

## Roadmap (not yet implemented)

- **Late chunking** — token-level embedding then split (requires Jina-v3 or a long-context model)
- **IRCoT** — interleaved reasoning + retrieval (major architectural addition; current `query_decomposer.py` is decompose-then-retrieve, not interleaved)
- **Speculative RAG** — draft→verify generation pattern
- **Online DPO** — real-time preference learning from query logs (separate MLOps pipeline)
- **pgvector bit quantization** — 32× storage reduction; requires full re-embed + schema migration via `src/migrate_embedding_dim.py`
- **pgvectorscale DiskANN** — evaluate when corpus exceeds 100K chunks (HNSW is optimal at current scale)

---

## License

MIT. HotpotQA benchmark data is CC BY-SA 4.0 (Yang et al., 2018) — not bundled, fetched at runtime.
