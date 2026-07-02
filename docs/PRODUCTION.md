# Production Readiness

Ultra RAG is designed to be run with explicit gates before it is exposed to users.
This checklist defines the operational baseline for a production deployment.

## Required Gates

Run these before every release:

```bash
python ultra_check.py
python ultra_retrieval_eval.py --golden eval/golden_retrieval.jsonl --fixture-results eval/fixture_results.json --json
docker compose config
python ultra_benchmark.py
python ultra_playbooks.py recommend "launch readiness" --json
python ultra_arch_audit.py --json
```

The release is not ready unless all gates pass.

## Environment

- Set `RAG_DB_PASSWORD` to a strong secret. Do not use the Docker Compose fallback outside local development.
- Set `ULTRARAG_API_KEY` for any shared, public, or network-exposed API.
- Set `ULTRARAG_CORS_ORIGINS` to an explicit allowlist. Do not use wildcard CORS in production.
- Keep optional provider keys blank unless the feature is enabled and tested.
- Run Ollama or another configured LLM endpoint close enough to the API to avoid avoidable latency.

## Health Probes

- `/api/live` checks that the API process is alive.
- `/api/ready` checks DB, pgvector data path, and Ollama reachability.
- `/api/health` returns detailed component state for dashboards and manual diagnosis.
- `/dashboard` provides a local operations view over health and recent query telemetry.
- `/app` provides the consumer workspace for upload, ingest, document inspection, and source-cited chat.
- `/api/observability/summary` returns query count, average latency, average quality, zero-result count, and recent query rows.

Use `/api/live` for container liveness and `/api/ready` for load-balancer readiness.

## Data Quality

Production quality depends on corpus-specific golden queries. The bundled eval fixture proves that the evaluation system works; it is not a substitute for your launch corpus.

Before launch:

- Add at least 50 high-value golden questions for the target corpus.
- Include exact relevant chunk IDs or document IDs.
- Track `precision@k`, `recall@k`, `mrr`, and `ndcg@k` in CI.
- Add difficult queries that should fail gracefully, not hallucinate.
- Re-run evals after chunking, embedding, reranking, or prompt changes.

## Security

- Keep `.env` out of git.
- Rotate any key that was ever pasted into logs, screenshots, issue threads, or chat.
- Require `X-API-Key` on the API outside local development.
- Restrict Docker volumes to documents and logs only.
- Review `/docs` exposure before public deployment; keep it private if the API is internet-facing.

## Operations

Minimum production telemetry should include:

- API request count, latency, status code, and route.
- Query strategy, result count, quality score, and latency.
- Per-query stage trace from provenance-enabled `/api/search` responses.
- Retrieval eval trend by release.
- Playbook recommendations for repeated launch, eval, and research workflows.
- Competitive claims only from `benchmarks/targets.json` entries that ran and produced measured evidence.
- DB size, connection count, slow queries, and pgvector index usage.
- Ollama/model endpoint latency and error rate.

## Launch Bar

A launch candidate should satisfy:

- All local readiness gates pass.
- Docker Compose validates.
- API requires authentication in the target environment.
- Readiness probes are wired into the runtime platform.
- Corpus-specific golden retrieval evals meet the agreed thresholds.
- A rollback path exists for config, image, and database migrations.
