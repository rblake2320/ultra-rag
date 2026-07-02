from pathlib import Path

from src.agent_playbooks import (
    catalog,
    load_playbooks,
    mask_env_values,
    propose_playbook_from_trace,
    recommend_playbooks,
)


def test_playbook_catalog_excludes_step_bodies():
    playbooks = load_playbooks(Path("playbooks"))
    entries = catalog(playbooks)

    assert entries
    assert "steps" not in entries[0]


def test_recommend_playbooks_respects_allowed_tools():
    playbooks = load_playbooks(Path("playbooks"))

    results = recommend_playbooks(
        "launch readiness production green",
        playbooks,
        allowed_tools={"filesystem", "shell", "docker"},
    )

    assert results
    assert results[0]["id"] == "production_green_gate"


def test_recommend_playbooks_filters_disallowed_tools():
    playbooks = load_playbooks(Path("playbooks"))

    results = recommend_playbooks(
        "youtube research",
        playbooks,
        allowed_tools={"filesystem"},
    )

    assert all("web" not in result["allowed_tools"] for result in results)


def test_mask_env_values_replaces_configured_secret(monkeypatch):
    monkeypatch.setenv("ULTRARAG_API_KEY", "secret-value")

    assert mask_env_values("key=secret-value", ["ULTRARAG_API_KEY"]) == "key=${ULTRARAG_API_KEY}"


def test_propose_playbook_from_trace_normalizes_id():
    proposal = propose_playbook_from_trace(
        "Launch Gate!",
        "Launch Gate",
        "Run checks",
        ["python ultra_check.py --json"],
        ["shell"],
    )

    assert proposal["id"] == "launch_gate"
    assert proposal["steps"] == ["python ultra_check.py --json"]
