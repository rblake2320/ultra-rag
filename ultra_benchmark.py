#!/usr/bin/env python3
"""Run evidence-based Ultra RAG benchmark targets.

This module intentionally does not rank products from subjective feature scores.
It reports only measured command/HTTP results and explicitly marks competitors
as unmeasured until a real endpoint or command is configured.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx


ROOT = Path(__file__).parent


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_benchmarks(config: dict[str, Any], require_all_configured: bool = False) -> dict[str, Any]:
    results = []
    for target in config.get("targets", []):
        if not target.get("enabled", False):
            results.append(_not_configured(target))
            continue
        kind = target.get("kind")
        if kind == "command":
            results.append(_run_command_target(target))
        elif kind == "http_chat":
            results.append(_run_http_chat_target(target, config.get("queries", [])))
        else:
            results.append(_failed(target, f"unknown target kind: {kind}"))

    configured = [r for r in results if r["status"] != "not_configured"]
    passed = all(r["status"] == "passed" for r in configured)
    if require_all_configured and any(r["status"] == "not_configured" for r in results):
        passed = False

    return {
        "passed": passed,
        "measured_count": len(configured),
        "unmeasured_count": sum(1 for r in results if r["status"] == "not_configured"),
        "results": results,
    }


def render_markdown(report: dict[str, Any], config: dict[str, Any]) -> str:
    lines = [
        "# Ultra RAG Evidence Benchmark",
        "",
        "This report contains measured results only. Unconfigured products are not ranked, scored, or treated as beaten.",
        "",
        f"Measured targets: {report['measured_count']}",
        f"Unmeasured targets: {report['unmeasured_count']}",
        f"Overall gate: {'PASS' if report['passed'] else 'FAIL'}",
        "",
        "| Target | Class | Status | Evidence |",
        "|---|---|---|---|",
    ]
    for result in report["results"]:
        evidence = result.get("evidence", "")
        if result.get("metrics"):
            evidence = json.dumps(result["metrics"], sort_keys=True)
        lines.append(
            f"| {result['name']} | {result.get('class', '')} | {result['status']} | {evidence} |"
        )

    lines.extend([
        "",
        "## Required Comparison Areas",
        "",
    ])
    for area in config.get("required_areas", []):
        lines.append(f"- {area}")

    lines.extend([
        "",
        "## How To Add Real Competitor Evidence",
        "",
        "Configure a target in `benchmarks/targets.json` with either a command that returns JSON metrics or an HTTP chat endpoint.",
        "Until that target runs successfully, it remains `not_configured` and no superiority claim is made.",
    ])
    return "\n".join(lines) + "\n"


def _run_command_target(target: dict[str, Any]) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            target["command"],
            cwd=ROOT,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=int(target.get("timeout_seconds", 120)),
        )
        latency_ms = int((time.perf_counter() - start) * 1000)
        metrics = _extract_json(proc.stdout)
        status = "passed" if proc.returncode == 0 and _passes_thresholds(metrics, target) else "failed"
        return {
            "name": target["name"],
            "class": target.get("class", ""),
            "status": status,
            "latency_ms": latency_ms,
            "metrics": metrics,
            "evidence": target.get("evidence", "command executed"),
            "output": proc.stdout.strip()[-4000:],
        }
    except Exception as exc:
        return _failed(target, str(exc))


def _run_http_chat_target(target: dict[str, Any], queries: list[dict[str, Any]]) -> dict[str, Any]:
    endpoint = target.get("endpoint")
    if not endpoint:
        return _not_configured(target)

    rows = []
    start = time.perf_counter()
    try:
        with httpx.Client(timeout=float(target.get("timeout_seconds", 30))) as client:
            for case in queries:
                payload = dict(target.get("payload_template", {}))
                payload[target.get("query_field", "query")] = case["query"]
                response = client.post(endpoint, json=payload)
                response.raise_for_status()
                body = response.json()
                text = json.dumps(body)
                rows.append(
                    {
                        "query": case["query"],
                        "has_required_terms": all(term.lower() in text.lower() for term in case.get("must_contain", [])),
                    }
                )
        passed_cases = sum(1 for row in rows if row["has_required_terms"])
        metrics = {
            "n_cases": len(rows),
            "passed_cases": passed_cases,
            "pass_rate": round(passed_cases / len(rows), 4) if rows else 0,
        }
        return {
            "name": target["name"],
            "class": target.get("class", ""),
            "status": "passed" if _passes_thresholds(metrics, target) else "failed",
            "latency_ms": int((time.perf_counter() - start) * 1000),
            "metrics": metrics,
            "evidence": endpoint,
        }
    except Exception as exc:
        return _failed(target, str(exc))


def _extract_json(output: str) -> dict[str, Any]:
    text = output.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {"raw_output": text[-1000:]}
    return json.loads(text[start : end + 1])


def _passes_thresholds(metrics: dict[str, Any], target: dict[str, Any]) -> bool:
    thresholds = target.get("thresholds", {})
    for key, minimum in thresholds.items():
        value = metrics.get(key)
        if value is None or float(value) < float(minimum):
            return False
    return True


def _not_configured(target: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": target["name"],
        "class": target.get("class", ""),
        "status": "not_configured",
        "evidence": target.get("configure_hint", "no runnable command or endpoint configured"),
    }


def _failed(target: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "name": target.get("name", "unknown"),
        "class": target.get("class", ""),
        "status": "failed",
        "evidence": reason,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evidence-based Ultra RAG benchmarks")
    parser.add_argument("--config", default="benchmarks/targets.json", help="Benchmark targets JSON")
    parser.add_argument("--out", default="benchmarks/BENCHMARK_REPORT.md", help="Markdown report path")
    parser.add_argument("--json", action="store_true", help="Print JSON report")
    parser.add_argument(
        "--require-all-configured",
        action="store_true",
        help="Fail if any competitor target is not configured",
    )
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    report = run_benchmarks(config, require_all_configured=args.require_all_configured)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        out_path = ROOT / args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(render_markdown(report, config), encoding="utf-8")
        print(f"Wrote {out_path}")
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
