import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import httpx
import pytest

spec = importlib.util.spec_from_file_location("maintenance", Path(__file__).parents[1] / "scripts" / "scheduled_maintenance.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@pytest.mark.parametrize("key", ["../escape", "/absolute", "C:/secret", "a/../../escape", "a\\..\\secret", "a:stream"])
def test_remote_path_escape_rejected(tmp_path, key):
    with pytest.raises(ValueError):
        m.safe_destination(tmp_path, key)


def test_nested_download_stays_inside_root(tmp_path):
    assert m.safe_destination(tmp_path, "docs/file.txt") == tmp_path / "docs" / "file.txt"


def test_git_parent_fallback_cannot_stage_owner_home(tmp_path):
    git = shutil.which("git")
    subprocess.run([git, "init", "-q", str(tmp_path)], check=True)
    child = tmp_path / "state"
    child.mkdir()
    (tmp_path / "private.txt").write_text("do not stage")
    with pytest.raises(RuntimeError, match="boundary"):
        m.echo({"root": str(child), "git": git})
    assert subprocess.check_output([git, "-C", str(tmp_path), "diff", "--cached", "--name-only"]) == b""


def test_nonzero_child_is_failure():
    with pytest.raises(RuntimeError, match="exited 7"):
        m.run([sys.executable, "-c", "raise SystemExit(7)"])


def test_missing_source_is_not_a_successful_sync(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "sync_s3", lambda source: (_ for _ in ()).throw(OSError("offline")))
    with pytest.raises(RuntimeError, match="sources failed"):
        m.personal({"sources": [{"type": "s3", "name": "required"}], "progress": str(tmp_path / "progress.json")})
    assert json.loads((tmp_path / "progress.json").read_text())["sources"][0]["status"] == "failed"


def test_watchdog_refuses_other_installation(tmp_path, monkeypatch):
    calls = []
    def command(args):
        calls.append(args)
        return json.dumps({"AppDirectory": str(tmp_path / "wrong"), "Application": sys.executable})
    monkeypatch.setattr(m, "run", command)
    with pytest.raises(RuntimeError, match="identity"):
        m.watchdog({"powershell": "powershell", "project": str(tmp_path / "correct"), "python": sys.executable})
    assert len(calls) == 1


class FakeDB:
    def __init__(self):
        self.updates = []
        self.rows = [{"id": 1, "created_at": __import__("datetime").datetime.now(),
                      "query_text": "unit fixture", "summary": "unit fixture", "query_log_id": 1}]
    def cursor(self, **kwargs):
        return self
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def execute(self, query, args=None):
        if query.startswith("UPDATE"):
            self.updates.append((query, args))
    def fetchone(self):
        return self.rows.pop(0) if self.rows else None
    def fetchall(self):
        return [("done", 1)]
    def commit(self):
        pass
    def close(self):
        pass


@pytest.mark.parametrize("status,result,success", [
    ("FAILURE", {}, False),
    ("SUCCESS", {}, False),
    ("SUCCESS", {"source_id": 1, "messages": 0}, False),
    ("SUCCESS", {"source_id": 1, "messages": 2}, True),
])
def test_bridge_only_marks_verified_ingestion_done(tmp_path, monkeypatch, status, result, success):
    db = FakeDB()
    monkeypatch.setattr(m, "db_connect", lambda root: db)
    monkeypatch.setattr(m, "load_key", lambda path: "unit-only")
    calls = []
    def response(request):
        calls.append(request)
        assert request.headers["X-API-Key"] == "unit-only"
        if request.url.path == "/api/search":
            return httpx.Response(200, json={"results": []})
        if request.url.path == "/api/ingest/session":
            return httpx.Response(200, json={"task_id": "unit-task", "status": "queued"})
        return httpx.Response(200, json={"status": status, "result": result})
    client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: client(transport=httpx.MockTransport(response), **kw))
    cfg = {"spool": str(tmp_path), "url": "http://unit.invalid", "key_file": "unused", "project": "unused"}
    if success:
        assert m.bridge(cfg)["completed"] == 1
        assert len(db.updates) == 2
    else:
        with pytest.raises(RuntimeError):
            m.bridge(cfg)
        assert not db.updates
    assert any(r.url.path == "/api/ingest/status/unit-task" for r in calls)


def test_bridge_ambiguous_send_is_not_replayed(tmp_path, monkeypatch):
    db = FakeDB()
    monkeypatch.setattr(m, "db_connect", lambda root: db)
    monkeypatch.setattr(m, "load_key", lambda path: "unit-only")
    (tmp_path / "bridge_1.receipt.json").write_text('{"status":"sending"}')
    calls = []
    def response(request):
        calls.append(request.url.path)
        return httpx.Response(200, json={"results": []})
    client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: client(transport=httpx.MockTransport(response), **kw))
    with pytest.raises(RuntimeError, match="unknown"):
        m.bridge({"spool": str(tmp_path), "url": "http://unit.invalid", "key_file": "unused", "project": "unused"})
    assert calls == ["/api/search"]
    assert not db.updates
