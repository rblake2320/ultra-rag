# Architecture Assessment

Status: sound enough for a local-first product branch, with explicit remaining gaps.

## Shape

Ultra RAG is organized as a layered system:

- Product surface: `static/app.html`, `static/dashboard.html`, product API routes in `ultra_server.py`.
- API orchestration: FastAPI endpoints for chat, search, upload, ingest, stats, observability, health, playbooks, eval, provenance, entities, and communities.
- Retrieval engine: `ultra_query.py`, `src/search.py`, graph search, reranking, CRAG, Self-RAG, parent expansion, provenance.
- Ingestion engine: `ingest.py`, `ultra_ingest.py`, parsers, chunker, embedder, code graph, KG, communities, RAPTOR.
- Evaluation and readiness: `ultra_check.py`, `ultra_retrieval_eval.py`, `ultra_product_smoke.py`, `ultra_benchmark.py`.
- Durable agent execution: `ultra_agent_run.py`, `src/durable_agents.py`, persisted run state, resumable workflow YAML.
- Operations: production docs, launch docs, failure runbooks, evidence benchmark targets, playbooks, durable runbooks.

## Functional Wiring

The consumer workflow is wired:

1. `/app` loads the workspace UI.
2. `/api/collections` lists or creates workspaces.
3. `/api/upload` saves supported files and can trigger parse/chunk/embed.
4. `/api/chat` runs retrieval and returns a grounded extractive answer with sources.
5. `/api/stats`, `/api/documents`, and `/api/observability/summary` populate the product UI.
6. `/api/live`, `/api/ready`, and `/api/health` separate process liveness from dependency readiness.

The engineering workflow is wired:

1. `python ultra_check.py --json` is the release gate.
2. `python ultra_retrieval_eval.py ...` is the deterministic retrieval gate.
3. `python ultra_benchmark.py` reports only measured targets.
4. `python ultra_product_smoke.py --json` verifies product surface wiring.
5. `python ultra_agent_run.py run --workflow workflows/production_green_gate.yaml --store runs --json` executes a resumable launch workflow.
6. `docker compose config` validates deploy configuration.

## Soundness

The architecture is not just a bag of features. It has a clear spine:

- Documents become chunks, embeddings, graph entities, summaries, and provenance records.
- Queries become routed retrieval runs with logs, traces, sources, and quality scores.
- Product UI calls the same API routes that automated tests and readiness checks inspect.
- Agent workflows persist each step and can resume without replaying completed idempotent work.
- Competitive claims are blocked unless measured targets are configured and pass.

## Complexity Risk

The system is broad. That creates real risk:

- Many optional advanced retrieval stages can fail independently.
- API routes currently share a single-process DB connection helper.
- The product app is intentionally static and lightweight, not a full frontend build system.
- The durable runner is single-node file-backed; high-scale multi-worker agents need a production workflow backend.
- Live quality still depends on actual corpus, model, DB, and embedding service behavior.

Mitigation:

- Keep deterministic gates in `ultra_check.py`.
- Keep optional services degraded rather than fatal unless `ULTRARAG_STRICT_STARTUP=true`.
- Treat every advanced retrieval stage as optional unless explicitly required by a release.
- Move durable workflow storage to Temporal, DBOS, Restate, Step Functions, or equivalent before promising distributed agent execution.
- Add real corpus golden evals before production claims.

## Current Verdict

- Functional: yes.
- Bloated: broad, but not incoherent. The readiness gate, modular CLIs, and docs prevent it from being unmanaged sprawl.
- Sloppy: not by current repo-local evidence. Tests, compile, product smoke, evals, Docker config, and benchmark evidence pass.
- Production-complete: not until real deployment telemetry, corpus evals, authentication policy, and live competitor targets are configured.
