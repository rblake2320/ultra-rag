# Gap Register

This file tracks gaps that are intentionally not claimed as fixed.

| Gap | Status | Reason Not Fully Fixed | Required Action To Close | Release Impact |
|---|---|---|---|---|
| Real corpus quality | Open | Requires the user's actual launch corpus and human-approved golden questions. | Add 50+ corpus-specific golden cases and pass retrieval thresholds. | Blocks quality superiority claims. |
| Real competitor benchmarks | Open | Competitor endpoints are not configured locally. | Configure AnythingLLM, Dify, RAGFlow, Open WebUI, etc. against the same corpus in `benchmarks/targets.json`. | Blocks competitor superiority claims. |
| Hosted production deployment | Open | No target cloud/runtime, domain, TLS, secrets manager, or live DB selected. | Deploy with real secrets, readiness probes, backups, monitoring, and rollback. | Blocks hosted production claim. |
| Full auth/RBAC | Partial | API key exists; workspace-level identity/RBAC is not implemented. | Add users, workspace ownership, object-level authorization, and tests. | Blocks multi-tenant SaaS claim. |
| Full generated-answer synthesis | Partial | `/api/chat` returns grounded extractive answers; LLM synthesis depends on configured models. | Add model-backed answer synthesis with citation enforcement and evals. | Blocks polished assistant claim. |
| OpenTelemetry export | Open | Internal traces exist; external OTLP export is not wired. | Add OTLP configuration and instrumentation if deployment requires it. | Blocks enterprise observability claim. |
| Browser visual regression | Open | Current environment did not expose a usable browser screenshot tool. | Add Playwright screenshot tests in CI or install callable browser tooling. | Blocks pixel-perfect UI claim. |
| Distributed durable agents | Partial | Local durable execution exists with persisted state and resume; no multi-worker Temporal/DBOS/Restate/Step Functions backend is configured. | Select production runtime, migrate `DurableRunStore` dispatch, add worker deployment and failure drills. | Blocks always-on distributed agent claim. |

Rules:

- A gap cannot be removed unless there is a test, command, endpoint, deployment record, or documented human approval proving closure.
- Marketing copy must not imply closed gaps are done unless this file says they are closed.
- If a gap blocks a user-facing claim, the claim must be downgraded or removed.
