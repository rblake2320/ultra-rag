# Failure Runbook

This runbook defines what happens when any Ultra RAG area fails.

## Failure Lifecycle

1. Detect
   - Source: `ultra_check.py`, `/api/ready`, `/api/health`, dashboard, user report, logs, eval failure, benchmark failure, or deployment alarm.
   - Record the failing command, endpoint, timestamp, environment, collection, and expected behavior.

2. Triage
   - Classify severity:
     - SEV1: data loss, security exposure, public outage, or broken chat/upload for all users.
     - SEV2: degraded retrieval quality, failed ingest, failed readiness, broken dashboard, or one workspace affected.
     - SEV3: docs drift, non-critical test gap, benchmark target not configured, or cosmetic UI issue.
   - Identify owner: API, ingestion, retrieval, product UI, deployment, eval, security, or docs.

3. Contain
   - Stop exposing broken paths if user-facing.
   - Roll back config or image if a recent change caused the failure.
   - Disable optional stages instead of breaking the full product when possible.
   - Keep `/api/live` available so orchestration can distinguish process health from dependency readiness.

4. Diagnose
   - Run `python ultra_check.py --json`.
   - Run the smallest failing command directly.
   - Check `logs/`, `/api/health`, `/api/ready`, query traces, and Docker health.
   - If retrieval quality failed, inspect golden cases and retrieved IDs.
   - If upload/ingest failed, inspect parser support, file path, collection, chunk count, and DB state.

5. Fix
   - Patch the smallest responsible module.
   - Add or update a regression test before declaring the failure fixed.
   - If the issue was process/documentation drift, update `docs/GAP_REGISTER.md` or the relevant runbook.

6. Verify
   - Run `python ultra_check.py --json`.
   - Run targeted tests for the affected component.
   - For product UI issues, run `python ultra_product_smoke.py --json`.
   - For deployment issues, run `docker compose config`.
   - For retrieval issues, run `python ultra_retrieval_eval.py --golden eval/golden_retrieval.jsonl --fixture-results eval/fixture_results.json --json`.

7. Return To Service
   - Re-enable disabled stage or route.
   - Restart API or redeploy image.
   - Confirm `/api/live` and `/api/ready`.
   - Confirm one real user path: upload or chat, depending on the failure.

8. Report
   - Summarize impact, cause, fix, verification, and follow-up.
   - Significant incidents require a blameless postmortem with action items.

## Failure Matrix

| Area | Detection | Log/Evidence | Immediate Action | Fix Validation | Return To Service |
|---|---|---|---|---|---|
| Product app | `/app`, `ultra_product_smoke.py` | HTTP status, static smoke output | Serve degraded UI, keep `/api/live` up | `python ultra_product_smoke.py --json` | Reload `/app` and confirm workspace controls |
| Upload | `/api/upload`, user report | API error, file name, parser suffix | Reject unsupported file safely | parser/product tests | Upload supported sample and list document |
| Ingest | `/api/ingest`, `ingest.py` | stage stats, parser errors | Disable optional stages if needed | targeted parser/chunker tests | Run parse/embed for affected collection |
| Retrieval | `/api/chat`, `ultra_retrieval_eval.py` | query trace, retrieved IDs, metrics | Fall back to hybrid or keyword | retrieval eval passes | Run representative query with sources |
| DB | `/api/ready`, `/api/health` | DB error, connect timeout | Keep `/app` and `/api/live`; mark degraded | schema check and `ultra_check.py` | Confirm ready returns ok |
| Ollama/model | `/api/health` | model endpoint error | Disable LLM-only stages; keep deterministic search | health and targeted query | Confirm model endpoint and query trace |
| Durable agent run | `ultra_agent_run.py show <run_id> --json` | run events, failed step output, attempt count | Stop new side-effect runs; keep completed steps from replaying | resume same run to `succeeded` | `python ultra_agent_run.py resume <run_id> --workflow workflows/production_green_gate.yaml --json` |
| Security | API key/CORS checks | request headers, config, logs | Require `ULTRARAG_API_KEY`, restrict CORS | config review and tests | Confirm unauthorized request rejected |
| Benchmark | `ultra_benchmark.py` | measured target report | Mark unconfigured, no claims | benchmark tests | Only publish measured results |
| Docker/deploy | `docker compose config` | compose output | Block release | compose config passes | Redeploy and confirm live/ready |

## Postmortem Template

- Date/time:
- Severity:
- Affected users/workspaces:
- Detection source:
- Timeline:
- Root cause:
- What worked:
- What failed:
- Fix:
- Verification commands:
- Follow-up actions:
- Owner and due date:
