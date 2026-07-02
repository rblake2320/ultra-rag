#!/usr/bin/env python3
"""Architecture and operations contract audit for Ultra RAG."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).parent

REQUIRED_FILES = [
    "docs/ARCHITECTURE_ASSESSMENT.md",
    "docs/FAILURE_RUNBOOK.md",
    "docs/GAP_REGISTER.md",
    "docs/SECURITY_CHECKLIST.md",
    "docs/DURABLE_AGENTS.md",
    "docs/PRODUCTION.md",
    "docs/LAUNCH.md",
    "static/app.html",
    "static/dashboard.html",
    "benchmarks/targets.json",
    "ultra_check.py",
    "ultra_product_smoke.py",
    "ultra_benchmark.py",
    "ultra_agent_run.py",
    "workflows/production_green_gate.yaml",
]

REQUIRED_ENDPOINT_STRINGS = [
    "/app",
    "/dashboard",
    "/api/chat",
    "/api/upload",
    "/api/collections",
    "/api/documents",
    "/api/observability/summary",
    "/api/live",
    "/api/ready",
    "/api/health",
]

FORBIDDEN_CLAIMS = [
    "in a class of its own",
    "beats anythingllm",
    "beats dify",
    "beats ragflow",
    "best in class",
]


def run_audit() -> dict:
    checks = [
        _required_files(),
        _endpoint_wiring(),
        _gap_register_rules(),
        _no_forbidden_claims(),
        _benchmark_is_evidence_based(),
    ]
    return {"passed": all(check["passed"] for check in checks), "checks": checks}


def _required_files() -> dict:
    missing = [path for path in REQUIRED_FILES if not (ROOT / path).exists()]
    return {"name": "required_files", "passed": not missing, "missing": missing}


def _endpoint_wiring() -> dict:
    text = (ROOT / "ultra_server.py").read_text(encoding="utf-8")
    missing = [value for value in REQUIRED_ENDPOINT_STRINGS if value not in text]
    return {"name": "endpoint_wiring", "passed": not missing, "missing": missing}


def _gap_register_rules() -> dict:
    text = (ROOT / "docs/GAP_REGISTER.md").read_text(encoding="utf-8")
    required = ["Real corpus quality", "Real competitor benchmarks", "Rules:"]
    missing = [value for value in required if value not in text]
    return {"name": "gap_register_rules", "passed": not missing, "missing": missing}


def _no_forbidden_claims() -> dict:
    scanned = []
    violations = []
    for path in [ROOT / "README.md", ROOT / "docs", ROOT / "benchmarks"]:
        files = path.rglob("*") if path.is_dir() else [path]
        for file in files:
            if file.is_file() and file.suffix.lower() in {".md", ".json", ".txt"}:
                scanned.append(str(file.relative_to(ROOT)))
                text = file.read_text(encoding="utf-8", errors="replace").lower()
                for claim in FORBIDDEN_CLAIMS:
                    if claim in text:
                        violations.append({"file": str(file.relative_to(ROOT)), "claim": claim})
    return {"name": "no_forbidden_claims", "passed": not violations, "violations": violations, "scanned": scanned}


def _benchmark_is_evidence_based() -> dict:
    text = (ROOT / "ultra_benchmark.py").read_text(encoding="utf-8")
    required = ["not_configured", "measured_count", "require_all_configured"]
    forbidden = ["score_product", "Rank | Product | Score"]
    missing = [value for value in required if value not in text]
    present_forbidden = [value for value in forbidden if value in text]
    return {
        "name": "benchmark_is_evidence_based",
        "passed": not missing and not present_forbidden,
        "missing": missing,
        "forbidden": present_forbidden,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run architecture/operations audit")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = run_audit()
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("Ultra RAG architecture audit:", "PASS" if result["passed"] else "FAIL")
        for check in result["checks"]:
            print(f"  {'PASS' if check['passed'] else 'FAIL'} {check['name']}")
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
