"""Durable workflow execution for production agent tasks.

This module intentionally keeps the local runtime small: workflow state is
persisted to JSON after every transition so a process can crash and resume from
the last completed step. The storage interface is narrow enough to replace with
Temporal, DBOS, or another durable backend without changing workflow specs.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


TERMINAL_STATUSES = {"succeeded"}


@dataclass(frozen=True)
class WorkflowStep:
    id: str
    name: str
    kind: str
    command: tuple[str, ...] = ()
    timeout_seconds: int = 300
    retries: int = 0
    retry_backoff_seconds: float = 0.0
    requires_approval: bool = False
    idempotency_key: str | None = None


@dataclass(frozen=True)
class WorkflowSpec:
    id: str
    title: str
    version: str
    steps: tuple[WorkflowStep, ...]


class DurableRunStore:
    """Atomic JSON file store for durable workflow runs."""

    def __init__(self, root: str | Path = "runs") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, run_id: str) -> Path:
        safe = "".join(ch for ch in run_id if ch.isalnum() or ch in {"-", "_"})
        if not safe:
            raise ValueError("run_id must contain at least one safe character")
        return self.root / f"{safe}.json"

    def create(self, spec: WorkflowSpec, run_id: str | None = None) -> dict[str, Any]:
        new_run_id = run_id or f"{spec.id}-{uuid.uuid4().hex[:12]}"
        path = self.path_for(new_run_id)
        if path.exists():
            raise FileExistsError(f"run already exists: {new_run_id}")
        now = _now()
        state = {
            "run_id": new_run_id,
            "workflow_id": spec.id,
            "workflow_version": spec.version,
            "title": spec.title,
            "status": "pending",
            "created_at": now,
            "updated_at": now,
            "current_step": None,
            "steps": [
                {
                    "id": step.id,
                    "name": step.name,
                    "kind": step.kind,
                    "status": "pending",
                    "attempts": 0,
                    "idempotency_key": step.idempotency_key or step.id,
                }
                for step in spec.steps
            ],
            "events": [
                {
                    "at": now,
                    "type": "run_created",
                    "workflow_id": spec.id,
                    "workflow_version": spec.version,
                }
            ],
        }
        self.save(state)
        return state

    def load(self, run_id: str) -> dict[str, Any]:
        return json.loads(self.path_for(run_id).read_text(encoding="utf-8"))

    def save(self, state: dict[str, Any]) -> None:
        state["updated_at"] = _now()
        path = self.path_for(str(state["run_id"]))
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)

    def list_runs(self) -> list[dict[str, Any]]:
        runs = []
        for path in sorted(self.root.glob("*.json")):
            try:
                state = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            runs.append(
                {
                    "run_id": state.get("run_id"),
                    "workflow_id": state.get("workflow_id"),
                    "status": state.get("status"),
                    "updated_at": state.get("updated_at"),
                }
            )
        return runs


class DurableWorkflowRunner:
    """Execute a workflow spec while persisting every state transition."""

    def __init__(self, spec: WorkflowSpec, store: DurableRunStore | None = None) -> None:
        self.spec = spec
        self.store = store or DurableRunStore()

    def start(self, run_id: str | None = None) -> dict[str, Any]:
        return self.store.create(self.spec, run_id=run_id)

    def run(self, run_id: str | None = None) -> dict[str, Any]:
        if run_id is None or not self.store.path_for(run_id).exists():
            state = self.start(run_id=run_id)
        else:
            state = self.store.load(run_id)
        return self.resume_state(state)

    def resume(self, run_id: str) -> dict[str, Any]:
        return self.resume_state(self.store.load(run_id))

    def resume_state(self, state: dict[str, Any]) -> dict[str, Any]:
        if state.get("status") in TERMINAL_STATUSES:
            return state

        state["status"] = "running"
        _event(state, "run_resumed")
        self.store.save(state)

        step_specs = {step.id: step for step in self.spec.steps}
        for recorded_step in state["steps"]:
            if recorded_step["status"] == "succeeded":
                continue

            step = step_specs[recorded_step["id"]]
            state["current_step"] = step.id

            if step.requires_approval and recorded_step.get("approval") != "approved":
                recorded_step["status"] = "blocked"
                state["status"] = "blocked"
                _event(state, "step_blocked", step_id=step.id, reason="approval_required")
                self.store.save(state)
                return state

            self._execute_step(state, recorded_step, step)
            if recorded_step["status"] != "succeeded":
                state["status"] = "failed"
                _event(state, "run_failed", step_id=step.id)
                self.store.save(state)
                return state

        state["status"] = "succeeded"
        state["current_step"] = None
        _event(state, "run_succeeded")
        self.store.save(state)
        return state

    def approve(self, run_id: str, step_id: str) -> dict[str, Any]:
        state = self.store.load(run_id)
        for step in state["steps"]:
            if step["id"] == step_id:
                step["approval"] = "approved"
                if step["status"] == "blocked":
                    step["status"] = "pending"
                _event(state, "step_approved", step_id=step_id)
                self.store.save(state)
                return state
        raise KeyError(f"unknown step: {step_id}")

    def _execute_step(
        self,
        state: dict[str, Any],
        recorded_step: dict[str, Any],
        step: WorkflowStep,
    ) -> None:
        max_attempts_this_resume = max(1, step.retries + 1)
        attempt_limit = recorded_step["attempts"] + max_attempts_this_resume
        while recorded_step["attempts"] < attempt_limit:
            recorded_step["attempts"] += 1
            recorded_step["status"] = "running"
            _event(
                state,
                "step_started",
                step_id=step.id,
                attempt=recorded_step["attempts"],
                idempotency_key=recorded_step["idempotency_key"],
            )
            self.store.save(state)

            result = _run_step(step)
            recorded_step["last_result"] = result
            recorded_step["status"] = "succeeded" if result["returncode"] == 0 else "failed"
            _event(
                state,
                "step_finished",
                step_id=step.id,
                attempt=recorded_step["attempts"],
                status=recorded_step["status"],
                returncode=result["returncode"],
            )
            self.store.save(state)

            if recorded_step["status"] == "succeeded":
                return
            if recorded_step["attempts"] < attempt_limit and step.retry_backoff_seconds > 0:
                time.sleep(step.retry_backoff_seconds)


def load_workflow(path: str | Path) -> WorkflowSpec:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    steps = []
    for item in raw.get("steps", []):
        command = tuple(_expand_command(str(part)) for part in item.get("command", []))
        steps.append(
            WorkflowStep(
                id=str(item["id"]),
                name=str(item.get("name", item["id"])),
                kind=str(item.get("kind", "command")),
                command=command,
                timeout_seconds=int(item.get("timeout_seconds", 300)),
                retries=int(item.get("retries", 0)),
                retry_backoff_seconds=float(item.get("retry_backoff_seconds", 0)),
                requires_approval=bool(item.get("requires_approval", False)),
                idempotency_key=str(item.get("idempotency_key") or item["id"]),
            )
        )
    return WorkflowSpec(
        id=str(raw["id"]),
        title=str(raw.get("title", raw["id"])),
        version=str(raw.get("version", "1")),
        steps=tuple(steps),
    )


def _run_step(step: WorkflowStep) -> dict[str, Any]:
    if step.kind != "command":
        return {
            "returncode": 2,
            "stdout": "",
            "stderr": f"unsupported step kind: {step.kind}",
            "duration_ms": 0,
        }
    if not step.command:
        return {"returncode": 2, "stdout": "", "stderr": "missing command", "duration_ms": 0}

    started = time.perf_counter()
    try:
        proc = subprocess.run(
            list(step.command),
            cwd=Path.cwd(),
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=step.timeout_seconds,
        )
        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-4000:],
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": 124,
            "stdout": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
            "stderr": f"timeout after {step.timeout_seconds}s",
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
        }


def _expand_command(value: str) -> str:
    return value.replace("{python}", sys.executable)


def _event(state: dict[str, Any], event_type: str, **extra: Any) -> None:
    event = {"at": _now(), "type": event_type}
    event.update(extra)
    state.setdefault("events", []).append(event)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
