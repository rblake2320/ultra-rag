#!/usr/bin/env python3
"""Run Ultra RAG agent workflows with durable local execution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.durable_agents import DurableRunStore, DurableWorkflowRunner, load_workflow


ROOT = Path(__file__).parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Durable Ultra RAG agent workflow runner")
    parser.add_argument("--workflow", default="workflows/production_green_gate.yaml")
    parser.add_argument("--store", default="runs")
    parser.add_argument("--json", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    start_cmd = sub.add_parser("start", help="Create a run without executing it")
    start_cmd.add_argument("--run-id")
    start_cmd.add_argument("--json", action="store_true")

    run_cmd = sub.add_parser("run", help="Create and execute a run")
    run_cmd.add_argument("--run-id")
    run_cmd.add_argument("--json", action="store_true")

    resume_cmd = sub.add_parser("resume", help="Resume an existing run")
    resume_cmd.add_argument("run_id")
    resume_cmd.add_argument("--json", action="store_true")

    approve_cmd = sub.add_parser("approve", help="Approve a blocked step")
    approve_cmd.add_argument("run_id")
    approve_cmd.add_argument("step_id")
    approve_cmd.add_argument("--json", action="store_true")

    show_cmd = sub.add_parser("show", help="Show a stored run")
    show_cmd.add_argument("run_id")
    show_cmd.add_argument("--json", action="store_true")

    list_cmd = sub.add_parser("list", help="List stored runs")
    list_cmd.add_argument("--json", action="store_true")

    args = parser.parse_args()
    store = DurableRunStore(_resolve(args.store))

    if args.command == "list":
        _print(store.list_runs(), args.json)
        return
    if args.command == "show":
        _print(store.load(args.run_id), args.json)
        return

    spec = load_workflow(_resolve(args.workflow))
    runner = DurableWorkflowRunner(spec, store)

    if args.command == "start":
        payload = runner.start(run_id=args.run_id)
    elif args.command == "run":
        payload = runner.run(run_id=args.run_id)
    elif args.command == "resume":
        payload = runner.resume(args.run_id)
    elif args.command == "approve":
        payload = runner.approve(args.run_id, args.step_id)
    else:
        raise SystemExit(f"unknown command: {args.command}")

    _print(payload, args.json)
    if payload.get("status") not in {"succeeded", "pending"}:
        raise SystemExit(1)


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _print(payload, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2))
        return
    if isinstance(payload, list):
        for item in payload:
            print(f"{item['run_id']} {item['workflow_id']} {item['status']} {item['updated_at']}")
    else:
        print(f"{payload['run_id']}: {payload['workflow_id']} {payload['status']}")


if __name__ == "__main__":
    main()
