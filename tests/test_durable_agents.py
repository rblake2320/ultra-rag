from __future__ import annotations

import sys
from pathlib import Path

import yaml

from src.durable_agents import DurableRunStore, DurableWorkflowRunner, load_workflow


def _write_workflow(path: Path, steps: list[dict]) -> None:
    path.write_text(
        yaml.safe_dump({"id": "test_flow", "title": "Test flow", "version": "1", "steps": steps}),
        encoding="utf-8",
    )


def test_durable_run_persists_success_and_events(tmp_path):
    workflow = tmp_path / "workflow.yaml"
    marker = tmp_path / "marker.txt"
    _write_workflow(
        workflow,
        [
            {
                "id": "write_marker",
                "kind": "command",
                "command": [
                    "{python}",
                    "-c",
                    f"from pathlib import Path; Path(r'{marker}').write_text('done')",
                ],
            }
        ],
    )

    runner = DurableWorkflowRunner(load_workflow(workflow), DurableRunStore(tmp_path / "runs"))
    state = runner.run(run_id="run-1")

    assert state["status"] == "succeeded"
    assert marker.read_text(encoding="utf-8") == "done"
    assert (tmp_path / "runs" / "run-1.json").exists()
    assert [event["type"] for event in state["events"]] == [
        "run_created",
        "run_resumed",
        "step_started",
        "step_finished",
        "run_succeeded",
    ]


def test_resume_skips_completed_step_after_failure(tmp_path):
    workflow = tmp_path / "workflow.yaml"
    marker = tmp_path / "count.txt"
    missing = tmp_path / "missing.txt"
    _write_workflow(
        workflow,
        [
            {
                "id": "increment_once",
                "kind": "command",
                "command": [
                    sys.executable,
                    "-c",
                    (
                        "from pathlib import Path; "
                        f"p=Path(r'{marker}'); "
                        "p.write_text(str(int(p.read_text() or '0')+1) if p.exists() else '1')"
                    ),
                ],
            },
            {
                "id": "needs_file",
                "kind": "command",
                "command": [
                    sys.executable,
                    "-c",
                    f"from pathlib import Path; raise SystemExit(0 if Path(r'{missing}').exists() else 3)",
                ],
            },
        ],
    )
    runner = DurableWorkflowRunner(load_workflow(workflow), DurableRunStore(tmp_path / "runs"))

    failed = runner.run(run_id="run-2")
    assert failed["status"] == "failed"
    assert marker.read_text(encoding="utf-8") == "1"

    missing.write_text("ok", encoding="utf-8")
    resumed = runner.resume("run-2")

    assert resumed["status"] == "succeeded"
    assert marker.read_text(encoding="utf-8") == "1"
    assert resumed["steps"][0]["attempts"] == 1
    assert resumed["steps"][1]["attempts"] == 2


def test_approval_blocks_side_effect_step_until_approved(tmp_path):
    workflow = tmp_path / "workflow.yaml"
    _write_workflow(
        workflow,
        [
            {
                "id": "side_effect",
                "kind": "command",
                "requires_approval": True,
                "command": [sys.executable, "-c", "print('approved')"],
            }
        ],
    )
    runner = DurableWorkflowRunner(load_workflow(workflow), DurableRunStore(tmp_path / "runs"))

    blocked = runner.run(run_id="run-3")
    assert blocked["status"] == "blocked"
    assert blocked["steps"][0]["status"] == "blocked"

    runner.approve("run-3", "side_effect")
    resumed = runner.resume("run-3")

    assert resumed["status"] == "succeeded"
    assert resumed["steps"][0]["last_result"]["returncode"] == 0
