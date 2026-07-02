from pathlib import Path

from src.retrieval_eval import (
    evaluate_results,
    load_golden_queries,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    search_fixture,
)


def test_retrieval_metric_formulas():
    retrieved = ["a", "b", "c"]
    relevant = {"b", "c", "d"}

    assert precision_at_k(retrieved, relevant, 2) == 0.5
    assert recall_at_k(retrieved, relevant, 3) == 2 / 3
    assert reciprocal_rank(retrieved, relevant) == 0.5
    assert 0.0 < ndcg_at_k(retrieved, relevant, 3) <= 1.0


def test_golden_fixture_eval_passes():
    cases = load_golden_queries(Path("eval/golden_retrieval.jsonl"))
    search_fn = search_fixture(Path("eval/fixture_results.json"))

    result = evaluate_results(cases, search_fn, k=5)

    assert result["passed"] is True
    assert result["n_cases"] == 3
    assert result["recall_at_k"] == 1.0
    assert result["mrr"] == 1.0


def test_eval_fails_when_required_relevant_id_missing():
    cases = [
        {
            "query": "missing",
            "relevant_ids": ["wanted"],
            "min_recall_at_k": 1.0,
        }
    ]

    result = evaluate_results(cases, lambda _query, _k: [{"stable_id": "other"}], k=5)

    assert result["passed"] is False
    assert result["cases"][0]["recall_at_k"] == 0.0
