# Launch Brief

## Positioning

Ultra RAG is a local-first advanced retrieval system that combines hybrid search,
knowledge graph reasoning, deterministic code graph extraction, RAPTOR summaries,
query routing, reranking, corrective retrieval, provenance, and evaluation gates.

The strongest launch message is not "another RAG app." It is:

> A production-oriented RAG stack that turns retrieval quality into an observable,
> testable system instead of a prompt-and-vector-store demo.

## Why It Matters

Most RAG systems fail quietly: retrieval drifts, prompts change, chunks get worse,
and no one knows until users complain. Ultra RAG now has deterministic graph
artifacts, golden retrieval checks, readiness gates, API health probes, and
agent instructions that keep future work aligned.

## Differentiators

- Deterministic code graph layer before LLM enrichment.
- Hybrid vector and keyword retrieval with graph-based reasoning options.
- Provenance and score decomposition for auditability.
- Corpus-specific retrieval evaluation designed for CI.
- Local-first deployment path with pgvector and Ollama.
- Built-in `/dashboard` for launch demos and operational inspection.
- Built-in `/app` for workspace upload, source-cited chat, and document inspection.
- Policy-aware reusable playbooks for repeated launch, eval, and external-research workflows.
- Agent-ready repo instructions for Codex, Claude, Gemini, Cursor, and Copilot.

## Launch Demo Flow

1. Run `python ultra_check.py` and show every gate green.
2. Build a code graph with `python ultra_code_graph.py default --no-db --json`.
3. Open `graph-out/default/graph.html` to show deterministic structure.
4. Run a golden retrieval eval and show precision, recall, MRR, and NDCG.
5. Start the API with Docker Compose and show `/api/live`, `/api/ready`, and `/docs`.
6. Open `/app`, create or select a workspace, upload documents, and run a source-cited chat.
7. Open `/dashboard` and show health, query latency, quality, and recent query history.
8. Query with provenance enabled and show why each result was selected plus per-stage trace timing.
9. Run `python ultra_benchmark.py` and discuss only measured results. Unconfigured competitors must remain labeled `not_configured`.
10. Run `python ultra_playbooks.py recommend "launch readiness" --json` to show the small skill index and then `show production_green_gate` to load the full workflow on demand.

## What To Avoid Claiming

- Do not claim universal production readiness without target-corpus evals.
- Do not claim hallucination elimination.
- Do not claim graph extraction replaces LLM-based KG extraction.
- Do not claim benchmark superiority until measured against public and private baselines.

## Launch Metrics

Track these from day one:

- Retrieval `recall@5` on corpus golden queries.
- Median and p95 search latency.
- Percentage of answers with usable provenance.
- Zero-result and low-confidence query rate.
- User-marked helpfulness rate.
- Failed readiness probes and model endpoint errors.
