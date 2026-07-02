from pathlib import Path

import ultra_benchmark


def test_benchmark_runs_only_measured_targets():
    config = ultra_benchmark.load_config(Path("benchmarks/targets.json"))
    report = ultra_benchmark.run_benchmarks(config)

    assert report["measured_count"] == 1
    assert report["unmeasured_count"] >= 1
    assert report["passed"] is True


def test_benchmark_markdown_does_not_rank_products():
    config = ultra_benchmark.load_config(Path("benchmarks/targets.json"))
    result = ultra_benchmark.run_benchmarks(config)
    report = ultra_benchmark.render_markdown(result, config)

    assert "Ultra RAG Evidence Benchmark" in report
    assert "| Target | Class | Status | Evidence |" in report
    assert "| Rank |" not in report


def test_require_all_configured_fails_with_unmeasured_targets():
    config = ultra_benchmark.load_config(Path("benchmarks/targets.json"))
    report = ultra_benchmark.run_benchmarks(config, require_all_configured=True)

    assert report["passed"] is False
