# Ultra RAG

**The most comprehensive open-source RAG system** — combining the best techniques from every major research system into a single, production-ready pipeline.

Ultra RAG implements a complete 7-stage ingestion pipeline and a 10-step intelligent query pipeline, drawing from LightRAG, HippoRAG, GraphRAG, Microsoft's GraphRAG, RAPTOR, CRAG, Self-RAG, HyDE, Contextual Retrieval, and original innovations not found in any single system.

---

## What Makes Ultra RAG Different

- **7-stage ingestion pipeline** — parse, embed, contextualize, parent-chunk, KG extract, community detect, RAPTOR tree
- **Knowledge Graph with PPR** — entities + relationships extracted by LLM, then Personalized PageRank traversal for multi-hop reasoning (HippoRAG-inspired)
- **Leiden community detection** — Louvain/Leiden algorithm over the KG, LLM-generated community summaries for global thematic queries (GraphRAG-inspired)
- **RAPTOR hierarchical summaries** — recursive clustering + LLM summarization tree for multi-level abstraction (RAPTOR paper)
- **Contextual Retrieval** — LLM-generated situating context prepended to each chunk before embedding (Anthropic research)
- **Parent-chunk expansion** — small chunks for precision retrieval, large parent chunks for context-rich answers
- **HyDE** — Hypothetical Document Embedding: generate a fake answer, embed it, retrieve by similarity
- **CRAG** — Corrective RAG: evaluate retrieved chunk quality, re-query or web-search if quality is low
- **Self-RAG** — adaptive filtering: predict relevance, support, and utility for each retrieved chunk
- **Auto-routing query classifier** — LLM routes each query to the optimal retrieval strategy
- **Retrieval Memory** — tracks chunk retrieval frequency with novelty penalties to prevent repetitive responses
- **Adversarial self-testing** — automatically generates blind-spot queries and monitors for coverage gaps
- **RAGAS evaluation** — faithfulness, answer relevancy, context precision, context recall metrics
- **NIM reranker** — optional NVIDIA NIM cross-encoder reranking (nvidia/rerank-qa-mistral-4b)

---

## Architecture

```
INGESTION PIPELINE
==================
Documents (DOCX/PDF/TXT/MD)
    │
    ▼  Stage 1: PARSE
    ├── Structure-aware DOCX parser (headings, tables, definitions, procedures)
    ├── PDF parser with table extraction (PyMuPDF)
    └── Markdown/TXT parser (# heading sections)
    │
    ▼  Stage 2: EMBED
    ├── Ollama nomic-embed-text (768-dim)
    └── Stored in pgvector with full-text search index
    │
    ▼  Stage 3: CONTEXTUAL
    └── LLM generates situating context per chunk ("This document discusses...")
    │
    ▼  Stage 4: PARENTS
    └── Small chunks linked to medium parent chunks for context expansion
    │
    ▼  Stage 5: KG EXTRACT
    ├── LLM extracts entities + typed relationships from each chunk
    └── Synonymy edges built from embedding similarity
    │
    ▼  Stage 6: COMMUNITIES
    ├── Leiden community detection over entity graph
    └── LLM summarizes each community into a thematic description
    │
    ▼  Stage 7: RAPTOR
    ├── KMeans clustering of chunk embeddings
    ├── LLM summarizes each cluster
    └── Recursive until tree depth reached (default: 3 levels)


QUERY PIPELINE
==============
User Query
    │
    ▼  Step 1: ROUTE
    └── LLM classifies: factual | thematic | multi-hop | comparative | hyde
    │
    ▼  Step 2: RETRIEVE (strategy-dependent)
    ├── hybrid     — vector cosine (pgvector) + BM25 keyword + RRF fusion
    ├── kg_local   — entity seed + PPR graph traversal (multi-hop reasoning)
    ├── kg_global  — community summary search (thematic/global questions)
    ├── multi_hop  — query decomposition → parallel retrieval → merge
    ├── hyde       — generate hypothetical answer → embed → retrieve by similarity
    └── compound   — decompose + route each sub-query independently
    │
    ▼  Step 3: RETRIEVAL MEMORY
    └── Record chunk IDs; apply novelty penalty to over-retrieved chunks
    │
    ▼  Step 4: RERANK
    └── NVIDIA NIM cross-encoder reranking (or score-based fallback)
    │
    ▼  Step 5: CRAG QUALITY CHECK
    └── LLM evaluates relevance; triggers re-query or web search if needed
    │
    ▼  Step 6: SELF-RAG FILTERING
    └── Per-chunk: predict IsRel, IsSup, IsUse — drop low-utility chunks
    │
    ▼  Step 7: UTILITY BOOST
    └── Apply helpfulness signal boost from retrieval memory
    │
    ▼  Step 8: PARENT EXPANSION
    └── Replace small chunks with parent chunks for richer context
    │
    ▼  Step 9: LOG QUERY
    └── Persist query, strategy, quality score, latency in rag.query_log
    │
    ▼  Step 10: PROVENANCE
    └── Build provenance chain with per-chunk score decomposition
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- PostgreSQL 14+ with [pgvector](https://github.com/pgvector/pgvector) extension
- [Ollama](https://ollama.com/) running locally with at least one model

### 1. Clone and install

```bash
git clone https://github.com/rblake2320/ultra-rag.git
cd ultra-rag
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and set RAG_DB_PASSWORD (and optionally NVIDIA_API_KEY)
```

### 3. Pull Ollama models

```bash
# Embedding model (required)
ollama pull nomic-embed-text

# LLM for KG extraction, RAPTOR, communities (pick one)
ollama pull deepseek-r1:32b    # high quality, requires ~20GB VRAM
ollama pull gemma3:latest      # lighter option (~3GB)
ollama pull llama3.1:8b        # fastest option (~5GB)
```

### 4. Set up pgvector

```sql
-- Run in psql or pgAdmin
CREATE EXTENSION IF NOT EXISTS vector;
```

### 5. Add your documents and ingest

```bash
mkdir documents
# Copy your DOCX, PDF, TXT, or MD files into documents/

python ingest.py default --embed
```

### 6. Query

```bash
# Simple query
python query.py "your question here"

# Full pipeline with auto-routing and provenance
python ultra_query.py "your question here"
```

---

## Full Pipeline

Run all 7 stages at once with `ultra_ingest.py`:

```bash
# Full pipeline (parse → embed → contextual → parents → KG → communities → RAPTOR)
python ultra_ingest.py default --stages all

# Selective stages
python ultra_ingest.py default --stages parse,embed
python ultra_ingest.py default --stages kg,communities,raptor

# Skip LLM stages (parse + embed only — no API calls)
python ultra_ingest.py default --stages all --no-llm

# Re-process contextual retrieval (e.g. after changing the LLM)
python ultra_ingest.py default --stages contextual --reprocess
```

---

## Query Strategies

| Strategy | Use Case | Example Query |
|----------|----------|---------------|
| `hybrid` | Specific facts, definitions | "What is the definition of X?" |
| `kg_local` | Multi-hop, relationships | "How does A relate to B through C?" |
| `kg_global` | Themes, summaries | "What are the main topics in this corpus?" |
| `multi_hop` | Complex, multi-part | "Compare X and Y in terms of Z and W" |
| `hyde` | Vague or broad queries | "Tell me about the general approach to..." |
| `compound` | AND / OR queries | "X AND Y in context of Z" |

Force a strategy:
```bash
python ultra_query.py "query" --strategy kg_local
python ultra_query.py "query" --strategy kg_global
python ultra_query.py "query" --hyde
```

---

## Novel Innovations

### Retrieval Memory (`src/retrieval_memory.py`)

Every retrieved chunk is recorded with a timestamp. Future queries apply a novelty penalty to chunks retrieved too recently, ensuring diverse and non-repetitive answers. A helpfulness signal (`/api/memories/{id}/helpful`) boosts chunks that users found useful.

### Adversarial Self-Testing (`src/adversarial.py`)

The system automatically generates adversarial queries designed to expose knowledge gaps. It measures how many queries return zero or low-confidence results, identifies blind spots, and logs them for corpus improvement. Can be scheduled to run periodically.

### Content-Aware KG Diffusion

Unlike standard KG RAG that stops at immediate entity neighbors, Ultra RAG uses Personalized PageRank (PPR) initialized from query-relevant seed entities. This allows multi-hop reasoning across the entire knowledge graph, surfacing non-obvious connections that flat vector search misses.

### Hybrid RRF Fusion

Vector search (cosine similarity via pgvector) and keyword search (PostgreSQL full-text search with ts_rank) are independently ranked, then fused with Reciprocal Rank Fusion (RRF). RRF parameters `k_vector` and `k_keyword` are tunable in `config.yaml`.

### Two-Tier Parent Chunking

Chunks are stored at two granularities: small (50-600 tokens) for precise retrieval, and medium parents (1,500+ tokens) for context-rich passage reading. Retrieval uses small chunks; the response generation receives the parent context.

### RAPTOR Clustering

Embeddings are clustered with Gaussian Mixture Models (soft assignment), then each cluster is summarized by an LLM. The summaries are re-embedded and re-clustered recursively to build an abstraction tree. Querying at different tree levels answers questions at different abstraction levels.

---

## API Server

Start the REST API server:

```bash
python ultra_server.py
# or
python ultra_server.py --port 8300 --host 0.0.0.0
```

Interactive docs at `http://localhost:8300/docs`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/search` | POST | Full 10-step query pipeline |
| `/api/ingest` | POST | Trigger ingestion pipeline |
| `/api/entities` | POST | Browse KG entities |
| `/api/communities` | POST | Browse detected communities |
| `/api/provenance/{id}` | GET | Full score decomposition for a query |
| `/api/eval` | POST | Generate synthetic Q&A + run RAGAS |
| `/api/stats` | GET | Collection statistics |
| `/api/health` | GET | DB, Ollama, pgvector health check |

### Optional TensorRT-LLM Speculative Decoding

UltraRAG uses Ollama by default. For low-latency local generation, you can run a
TensorRT-LLM OpenAI-compatible server and switch `llm.provider` to `trtllm` in
`config.yaml`.

Start with NGram speculation because it supports all models and does not need a
separate draft model:

```bash
trtllm-serve /path/to/target_model \
  --host 0.0.0.0 \
  --port 8000 \
  --extra_llm_api_options D:\ultra-rag\configs\trtllm-speculative-ngram.yaml
```

Then set:

```yaml
llm:
  provider: trtllm
  trtllm_url: "http://localhost:8000"
```

Keep Ollama running as fallback while benchmarking. Speculative decoding is most
likely to help interactive low-batch queries; benchmark UltraRAG search and
synthesis prompts before making it the default.

Example search request:
```json
POST /api/search
{
  "query": "your question",
  "collection": "default",
  "top_k": 5,
  "strategy": null,
  "include_provenance": true
}
```

---

## Evaluation

Generate a synthetic Q&A dataset and measure RAG quality with RAGAS metrics:

```bash
# Generate 50 questions + run RAGAS evaluation
python ultra_eval.py default --generate 50 --run

# Just generate questions (save for later)
python ultra_eval.py default --generate 100

# Run eval on previously generated questions
python ultra_eval.py default --run --run-name "v1-baseline"

# Show past eval runs
python ultra_eval.py default --report

# Export questions to JSONL
python ultra_eval.py default --export ./eval_questions.jsonl
```

RAGAS metrics reported:
- **Faithfulness** — are answers grounded in retrieved context?
- **Answer Relevancy** — does the answer address the question?
- **Context Precision** — are retrieved chunks relevant to the question?
- **Context Recall** — does retrieved context cover the ground truth answer?

---

## Hardware Recommendations

| Setup | Minimum | Recommended |
|-------|---------|-------------|
| Embedding only | 8GB RAM, CPU | 16GB RAM, any GPU |
| + Contextual / routing | 16GB RAM, 8GB VRAM | 32GB RAM, 16GB VRAM |
| + KG + RAPTOR (32B LLM) | 64GB RAM, 24GB VRAM | 128GB RAM, RTX 4090/5090 |
| Production (full pipeline) | 32GB RAM, 16GB VRAM | 128GB RAM, A100/H100 |

The system degrades gracefully: each stage is independently optional. On CPU-only machines, use a small fast model (`gemma3:latest`) and skip RAPTOR.

---

## Configuration Reference

All configuration lives in `config.yaml`. Key sections:

```yaml
collections:
  my_collection:
    paths: ["/path/to/docs"]
    exclude_dirs: ["__pycache__", ".git"]
    skip_files: ["draft.docx"]

embedding:
  model: nomic-embed-text       # change to mxbai-embed-large for 1024-dim
  dimensions: 768

llm:
  model: deepseek-r1:32b        # main LLM for KG/RAPTOR/communities
  fast_model: gemma3:latest     # fast LLM for routing/contextual

chunking:
  max_tokens: 600               # reduce for more precise retrieval
  min_tokens: 50

kg:
  ppr_alpha: 0.85               # higher = stays closer to seed entities
```

---

## Project Structure

```
ultra-rag/
├── ingest.py              # Simple ingest CLI (parse + chunk + embed)
├── query.py               # Simple query CLI (hybrid search)
├── ultra_ingest.py        # Full 7-stage ingestion pipeline
├── ultra_query.py         # Full 10-step intelligent query pipeline
├── ultra_server.py        # FastAPI REST server
├── ultra_eval.py          # Synthetic evaluation + RAGAS metrics
├── config.yaml            # All configuration
├── requirements.txt
├── .env.example
├── documents/             # Put your docs here (git-ignored)
├── logs/                  # Auto-created log files (git-ignored)
├── src/
│   ├── config.py          # Config loader
│   ├── db.py              # Base PostgreSQL schema + CRUD
│   ├── db_ultra.py        # Extended schema (KG, RAPTOR, eval tables)
│   ├── parsers.py         # DOCX / PDF / TXT parsers
│   ├── chunker.py         # Token-aware chunker
│   ├── embedder.py        # Ollama embedding + batch processing
│   ├── llm.py             # LLM client (Ollama + Claude fallback)
│   ├── search.py          # Hybrid vector + keyword search + RRF
│   ├── reranker.py        # NIM cross-encoder reranking
│   ├── contextual.py      # Contextual Retrieval (LLM chunk context)
│   ├── parent_chunker.py  # Two-tier parent chunk hierarchy
│   ├── kg_extractor.py    # LLM entity + relationship extraction
│   ├── kg_graph.py        # igraph PPR traversal
│   ├── kg_communities.py  # Leiden community detection + LLM summaries
│   ├── raptor.py          # RAPTOR hierarchical summary tree
│   ├── query_router.py    # LLM query classifier + strategy dispatcher
│   ├── query_decomposer.py # Multi-hop query decomposition
│   ├── hyde.py            # Hypothetical Document Embedding
│   ├── corrective.py      # CRAG corrective retrieval
│   ├── self_rag.py        # Self-RAG adaptive filtering
│   ├── retrieval_memory.py # Retrieval frequency tracking + novelty
│   ├── adversarial.py     # Adversarial self-testing
│   ├── provenance.py      # Score decomposition + audit trail
│   ├── eval_generator.py  # Synthetic Q&A dataset generation
│   ├── eval_runner.py     # RAGAS evaluation runner
│   ├── multimodal.py      # Image/OCR support
│   └── watcher.py         # Directory watcher for auto-ingest
└── tests/
    ├── test_ultra_rag.py  # Full test suite
    └── test_chunker.py    # Chunker unit tests
```

---

## Contributing

Pull requests welcome. Key areas for contribution:
- Additional document parsers (PPTX, HTML, XML, Excel)
- New retrieval strategies
- Streaming API responses
- Web UI dashboard
- Additional embedding providers (OpenAI, Cohere, etc.)
- RAGAS metric extensions

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

Ultra RAG synthesizes ideas from:
- [HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG) — KG + PPR retrieval
- [Microsoft GraphRAG](https://github.com/microsoft/graphrag) — community detection for global queries
- [RAPTOR](https://arxiv.org/abs/2401.18059) — recursive abstractive processing for tree-organized retrieval
- [Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) — Anthropic
- [CRAG](https://arxiv.org/abs/2401.15884) — Corrective Retrieval Augmented Generation
- [Self-RAG](https://arxiv.org/abs/2310.11511) — adaptive retrieval with reflection tokens
- [HyDE](https://arxiv.org/abs/2212.10496) — Hypothetical Document Embeddings
- [LightRAG](https://github.com/HKUDS/LightRAG) — unified KG + vector retrieval
