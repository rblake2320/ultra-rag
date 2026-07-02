"""
Deterministic retrieval evaluation utilities.

These metrics are intentionally independent of LLM judges and external
services. They are used for fast regression tests and CI gates before running
heavier RAGAS or human-judged evaluations.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable, Iterable


def _stable_id(row: dict) -> str:
    for key in ("stable_id", "id", "chunk_id", "document_id"):
        if key in row and row[key] is not None:
            return str(row[key])
    return ""


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top = retrieved[:k]
    if not top:
        return 0.0
    return sum(1 for item in top if item in relevant) / len(top)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top = retrieved[:k]
    return sum(1 for item in top if item in relevant) / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for idx, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant or k <= 0:
        return 0.0
    dcg = 0.0
    for idx, item in enumerate(retrieved[:k], start=1):
        if item in relevant:
            dcg += 1.0 / math.log2(idx + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def load_golden_queries(path: str | Path) -> list[dict]:
    """Load JSONL golden retrieval cases.

    Each line must contain:
      - query: str
      - relevant_ids: list[str]
    Optional:
      - tags: list[str]
      - min_recall_at_k / min_precision_at_k / min_mrr / min_ndcg_at_k
    """
    cases: list[dict] = []
    p = Path(path)
    for line_no, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        data = json.loads(line)
        if not isinstance(data.get("query"), str) or not data["query"].strip():
            raise ValueError(f"{p}:{line_no}: missing non-empty query")
        relevant = data.get("relevant_ids")
        if not isinstance(relevant, list) or not relevant:
            raise ValueError(f"{p}:{line_no}: relevant_ids must be a non-empty list")
        data["relevant_ids"] = [str(x) for x in relevant]
        cases.append(data)
    return cases


def evaluate_results(
    cases: Iterable[dict],
    search_fn: Callable[[str, int], list[dict]],
    *,
    k: int = 5,
) -> dict:
    """Evaluate a search function against golden cases."""
    case_results = []
    for case in cases:
        query = case["query"]
        relevant = set(case["relevant_ids"])
        rows = search_fn(query, k)
        retrieved = [_stable_id(row) for row in rows]
        retrieved = [item for item in retrieved if item]
        result = {
            "query": query,
            "retrieved_ids": retrieved,
            "relevant_ids": sorted(relevant),
            "precision_at_k": precision_at_k(retrieved, relevant, k),
            "recall_at_k": recall_at_k(retrieved, relevant, k),
            "mrr": reciprocal_rank(retrieved, relevant),
            "ndcg_at_k": ndcg_at_k(retrieved, relevant, k),
        }
        result["passed"] = (
            result["precision_at_k"] >= float(case.get("min_precision_at_k", 0.0))
            and result["recall_at_k"] >= float(case.get("min_recall_at_k", 1.0))
            and result["mrr"] >= float(case.get("min_mrr", 0.0))
            and result["ndcg_at_k"] >= float(case.get("min_ndcg_at_k", 0.0))
        )
        case_results.append(result)

    n = len(case_results)
    if n == 0:
        return {
            "n_cases": 0,
            "passed": False,
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "mrr": 0.0,
            "ndcg_at_k": 0.0,
            "cases": [],
        }
    summary = {
        "n_cases": n,
        "passed": all(r["passed"] for r in case_results),
        "precision_at_k": sum(r["precision_at_k"] for r in case_results) / n,
        "recall_at_k": sum(r["recall_at_k"] for r in case_results) / n,
        "mrr": sum(r["mrr"] for r in case_results) / n,
        "ndcg_at_k": sum(r["ndcg_at_k"] for r in case_results) / n,
        "cases": case_results,
    }
    return summary


def search_fixture(fixture_path: str | Path) -> Callable[[str, int], list[dict]]:
    """Build a deterministic search function from a JSON fixture.

    The fixture maps query strings to ordered result rows:
      {"query": [{"stable_id": "doc-a"}, ...]}
    """
    data = json.loads(Path(fixture_path).read_text(encoding="utf-8"))

    def _search(query: str, k: int) -> list[dict]:
        return list(data.get(query, []))[:k]

    return _search
