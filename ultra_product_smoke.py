#!/usr/bin/env python3
"""Static product-surface smoke checks that do not require DB/model services."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).parent


def run_smoke() -> dict:
    checks = [
        _file_contains("static/app.html", ["Workspace", "Upload + ingest", "/api/chat", "/api/upload"]),
        _file_contains("static/dashboard.html", ["Ultra RAG Dashboard", "/api/observability/summary"]),
        _file_contains("ultra_server.py", ["/api/chat", "/api/upload", "/api/collections", "/app"]),
        _file_contains("src/product.py", ["extractive_answer", "safe_upload_name", "is_supported_upload"]),
    ]
    return {"passed": all(check["passed"] for check in checks), "checks": checks}


def _file_contains(path: str, required: list[str]) -> dict:
    full = ROOT / path
    if not full.exists():
        return {"name": path, "passed": False, "missing": required, "output": "file missing"}
    text = full.read_text(encoding="utf-8")
    missing = [item for item in required if item not in text]
    return {
        "name": path,
        "passed": not missing,
        "missing": missing,
        "output": "ok" if not missing else "missing required strings",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run product-surface smoke checks")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = run_smoke()
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("Ultra RAG product smoke:", "PASS" if result["passed"] else "FAIL")
        for check in result["checks"]:
            print(f"  {'PASS' if check['passed'] else 'FAIL'} {check['name']}")
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
