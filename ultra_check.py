#!/usr/bin/env python3
"""Local readiness checks for Ultra RAG."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).parent


def _run(name: str, cmd: list[str]) -> dict:
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return {
        "name": name,
        "cmd": " ".join(cmd),
        "passed": proc.returncode == 0,
        "returncode": proc.returncode,
        "output": proc.stdout.strip(),
    }


def _check_config() -> dict:
    try:
        from src.config import get_config, reset_config_cache

        reset_config_cache()
        cfg = get_config()
        ok = bool(cfg.get("db")) and bool(cfg.get("search")) and "code_graph" in cfg
        return {
            "name": "config",
            "passed": ok,
            "output": "config loaded" if ok else "config missing required sections",
        }
    except Exception as exc:
        return {"name": "config", "passed": False, "output": str(exc)}


def _check_code_graph_smoke() -> dict:
    fixture_root = ROOT / ".tmp-check-code-graph-src"
    out_root = ROOT / ".tmp-check-graph"
    try:
        if fixture_root.exists():
            shutil.rmtree(fixture_root)
        if out_root.exists():
            shutil.rmtree(out_root)
        fixture_root.mkdir()
        (fixture_root / "sample.py").write_text(
            "def helper(value):\n"
            "    return str(value)\n\n"
            "def main():\n"
            "    return helper(1)\n",
            encoding="utf-8",
        )

        from src.code_graph import build_code_graph, write_artifacts

        graph = build_code_graph([fixture_root])
        stats = write_artifacts(graph, out_root / "default", "default")
        passed = stats["nodes"] >= 3 and stats["edges"] >= 2
        return {
            "name": "code_graph_smoke",
            "passed": passed,
            "output": json.dumps(stats, indent=2),
        }
    except Exception as exc:
        return {"name": "code_graph_smoke", "passed": False, "output": str(exc)}
    finally:
        shutil.rmtree(fixture_root, ignore_errors=True)
        shutil.rmtree(out_root, ignore_errors=True)


def _check_package_metadata() -> dict:
    try:
        pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        project = pyproject.get("project", {})
        scripts = project.get("scripts", {})
        required_scripts = {
            "ultrarag-ingest",
            "ultrarag-query",
            "ultrarag-ultra-ingest",
            "ultrarag-ultra-query",
            "ultrarag-eval",
            "ultrarag-server",
            "ultrarag-code-graph",
            "ultrarag-retrieval-eval",
            "ultrarag-check",
            "ultrarag-benchmark",
            "ultrarag-playbooks",
            "ultrarag-product-smoke",
            "ultrarag-arch-audit",
            "ultrarag-agent-run",
        }
        missing = sorted(required_scripts - set(scripts))
        passed = not missing and bool(project.get("name")) and bool(project.get("version"))
        output = "package metadata and scripts present"
        if missing:
            output = "missing scripts: " + ", ".join(missing)
        return {"name": "package_metadata", "passed": passed, "output": output}
    except Exception as exc:
        return {"name": "package_metadata", "passed": False, "output": str(exc)}


def run_checks(include_docker: bool = True) -> dict:
    py = sys.executable
    checks = [
        _check_config(),
        _check_package_metadata(),
        _run("tests", [py, "-m", "pytest", "-q"]),
        _run(
            "compile",
            [
                py,
                "-m",
                "compileall",
                "src",
                "ingest.py",
                "query.py",
                "ultra_ingest.py",
                "ultra_query.py",
                "ultra_eval.py",
                "ultra_server.py",
                "ultra_code_graph.py",
                "ultra_retrieval_eval.py",
                "ultra_check.py",
                "ultra_benchmark.py",
                "ultra_playbooks.py",
                "ultra_product_smoke.py",
                "ultra_arch_audit.py",
                "ultra_agent_run.py",
            ],
        ),
        _run(
            "retrieval_eval_fixture",
            [
                py,
                "ultra_retrieval_eval.py",
                "--golden",
                "eval/golden_retrieval.jsonl",
                "--fixture-results",
                "eval/fixture_results.json",
                "--json",
            ],
        ),
        _run("product_smoke", [py, "ultra_product_smoke.py", "--json"]),
        _run("architecture_audit", [py, "ultra_arch_audit.py", "--json"]),
        _run(
            "durable_agent_workflow",
            [
                py,
                "ultra_agent_run.py",
                "--workflow",
                "workflows/production_green_gate.yaml",
                "--store",
                ".tmp-check-runs",
                "run",
                "--run-id",
                "readiness",
                "--json",
            ],
        ),
        _run("competitive_benchmark", [py, "ultra_benchmark.py", "--json"]),
        _run("agent_playbooks", [py, "ultra_playbooks.py", "recommend", "launch readiness", "--json"]),
        _check_code_graph_smoke(),
    ]
    if include_docker:
        checks.append(_run("docker_compose_config", ["docker", "compose", "config"]))

    passed = all(check["passed"] for check in checks)
    shutil.rmtree(ROOT / ".tmp-check-runs", ignore_errors=True)
    return {"passed": passed, "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local Ultra RAG readiness checks")
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    parser.add_argument("--no-docker", action="store_true", help="Skip docker compose config")
    args = parser.parse_args()

    result = run_checks(include_docker=not args.no_docker)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("Ultra RAG readiness:", "PASS" if result["passed"] else "FAIL")
        for check in result["checks"]:
            mark = "PASS" if check["passed"] else "FAIL"
            print(f"  {mark:<4} {check['name']}")
            if not check["passed"]:
                print(check["output"])
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
