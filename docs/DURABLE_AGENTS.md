# Durable Agent Execution

Ultra RAG agent workflows must be able to resume after process death, record each action, and fail closed when a step needs human approval. The local runner in `src/durable_agents.py` provides that baseline without pretending to be a full distributed workflow engine.

## What Is Durable Now

- Each workflow run is persisted under `runs/` as JSON.
- Every step transition is recorded as an event.
- Command output, return code, attempt count, duration, and status are saved after each step.
- Resuming a run skips previously succeeded steps and continues at the first pending or failed step.
- Steps have bounded retries and timeouts.
- Side-effecting steps can require explicit approval before execution.
- Workflow definitions are YAML files under `workflows/`.

## Run It

```bash
python ultra_agent_run.py run --workflow workflows/production_green_gate.yaml --store runs --json
python ultra_agent_run.py list --store runs
python ultra_agent_run.py show <run_id> --store runs --json
python ultra_agent_run.py resume <run_id> --workflow workflows/production_green_gate.yaml --store runs --json
```

## Production Upgrade Path

The current runner is appropriate for local launch gates, self-hosted single-node operations, and CI smoke workflows. For multi-worker production agents that take external action, replace `DurableRunStore` and the runner dispatch layer with Temporal, DBOS, Restate, Step Functions, or another durable execution backend.

That upgrade should preserve the same contract:

- Persist before and after every external action.
- Use idempotency keys for every side-effecting tool call.
- Keep deterministic workflow control separate from non-deterministic LLM calls.
- Store receipts for tool calls, approvals, model responses, and retries.
- Make failed runs resumable without replaying completed side effects.
- Block destructive or privileged actions until approval is recorded.

## Failure Handling

Failed durable runs follow `docs/FAILURE_RUNBOOK.md`: detect, triage, contain, diagnose, fix, verify, return to service, and report. A run is not considered recovered until the failed step is fixed, resumed, and the final run status is `succeeded`.
