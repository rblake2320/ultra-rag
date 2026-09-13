"""Windows scheduled maintenance with explicit outcomes and durable receipts.

Machine paths and endpoints live in a private JSON configuration, not this file.
No task reports success when a child process or required source failed.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import time
from datetime import datetime, timezone


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def run(command, *, cwd=None, timeout=120):
    p = subprocess.run(command, cwd=cwd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=timeout)
    if p.returncode:
        # Do not propagate arbitrary credential-bearing child output to receipts.
        raise RuntimeError(f"{Path(command[0]).name} exited {p.returncode}")
    return p.stdout.strip()


@contextlib.contextmanager
def job_lock(path):
    """Kernel lock releases on process death; do not guess stale PID locks."""
    import msvcrt
    with open(path, "a+b") as f:
        f.seek(0, 2)
        if not f.tell():
            f.write(b"0")
            f.flush()
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)


def safe_destination(root, key):
    rel = PurePosixPath(key)
    if rel.is_absolute() or any(p in ("..", "") or ":" in p or "\\" in p for p in rel.parts):
        raise ValueError("Remote source attempted path escape")
    root = Path(root).resolve()
    target = root.joinpath(*rel.parts).resolve()
    if target == root or not target.is_relative_to(root):
        raise ValueError("Remote source attempted path escape")
    return target


def echo(cfg):
    root = Path(cfg["root"]).resolve()
    git = cfg["git"]
    actual = Path(run([git, "-C", str(root), "rev-parse", "--show-toplevel"])).resolve()
    if actual != root or not (root / ".git").is_dir():
        raise RuntimeError("Echo repository boundary mismatch; refusing to stage anything")
    remote = run([git, "-C", str(root), "remote", "get-url", "origin"])
    if remote != cfg["remote"]:
        raise RuntimeError("Echo backup remote differs from approved private repository")
    # Check traversal explicitly: Git can misread inaccessible directories as deletions.
    def denied(err):
        raise err
    for base, dirs, files in os.walk(root, onerror=denied):
        dirs[:] = [d for d in dirs if d not in cfg["exclude_dirs"]]
        for name in files:
            if name not in cfg["exclude_files"]:
                with open(Path(base) / name, "rb") as f:
                    f.read(1)
    run([git, "-C", str(root), "add", "-A", "--", "."])
    changed = bool(run([git, "-C", str(root), "diff", "--cached", "--name-only"]))
    if changed:
        run([git, "-C", str(root), "commit", "-m", "auto: nightly Echo state backup"])
    branch = run([git, "-C", str(root), "symbolic-ref", "--short", "HEAD"])
    run([git, "-C", str(root), "push", "origin", f"HEAD:refs/heads/{branch}"], timeout=180)
    head = run([git, "-C", str(root), "rev-parse", "HEAD"])
    remote_line = run([git, "-C", str(root), "ls-remote", "origin", f"refs/heads/{branch}"])
    if not remote_line or remote_line.split()[0] != head:
        raise RuntimeError("Echo remote commit verification failed")
    return {"commit": head, "remote_verified": True, "changed": changed}


def load_key(path):
    key = os.environ.get("MEMORYWEB_API_KEY", "").strip()
    if key:
        return key
    # Task Scheduler may retain a pre-rotation environment until next logon.
    if os.name == "nt":
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as registry:
            try:
                key = winreg.QueryValueEx(registry, "MEMORYWEB_API_KEY")[0].strip()
                if key:
                    return key
            except FileNotFoundError:
                pass
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        name, sep, value = line.partition("=")
        if sep and name.strip() == "MEMORYWEB_API_KEY":
            key = value.strip().strip("\"'")
            if key:
                return key
    raise RuntimeError("MemoryWeb API key missing from private configuration")


def db_connect(root):
    import psycopg2
    sys.path.insert(0, str(root))
    from src.config import get_config
    d = get_config()["db"]
    return psycopg2.connect(host=d["host"], port=d["port"], dbname=d["database"],
                            user=d["user"], password=d["password"], connect_timeout=5)


def bridge(cfg):
    import httpx
    import psycopg2.extras
    spool = Path(cfg["spool"])
    spool.mkdir(parents=True, exist_ok=True)
    with httpx.Client(base_url=cfg["url"], headers={"X-API-Key": load_key(cfg["key_file"])},
                      timeout=20) as client, contextlib.closing(db_connect(cfg["project"])) as conn:
        # A read-only authenticated request validates credentials even with an empty queue.
        client.post("/api/search", json={"query": "bridge health", "k": 1}).raise_for_status()
        completed = 0
        for _ in range(20):
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT * FROM rag.bridge_queue WHERE status='pending' "
                            "AND attempts < 3 ORDER BY created_at LIMIT 1 FOR UPDATE SKIP LOCKED")
                row = cur.fetchone()
                if row is None:
                    conn.commit()
                    break
                path = spool / f"bridge_{row['id']}.jsonl"
                state_file = spool / f"bridge_{row['id']}.receipt.json"
                state = json.loads(state_file.read_text()) if state_file.exists() else {}
                if not path.exists():
                    records = [{"type": role, "message": {"role": role, "content": content},
                                "timestamp": row["created_at"].isoformat()}
                               for role, content in [("user", row["query_text"]),
                                                     ("assistant", row["summary"])]]
                    temp = path.with_suffix(".tmp")
                    temp.write_text("\n".join(json.dumps(r, default=str) for r in records)+"\n", encoding="utf-8")
                    os.replace(temp, path)
                if state.get("status") == "sending" and not state.get("task_id"):
                    raise RuntimeError("Bridge outcome unknown; reconcile stored session before retry")
                if not state.get("task_id"):
                    atomic_json(state_file, {"status": "sending", "path": str(path)})
                    response = client.post("/api/ingest/session", json={"path": str(path), "force": False})
                    response.raise_for_status()
                    task_id = response.json().get("task_id")
                    if not task_id:
                        raise RuntimeError("MemoryWeb omitted asynchronous task ID")
                    state = {"status": "queued", "task_id": task_id}
                    atomic_json(state_file, state)
                deadline = time.monotonic()+60
                while True:
                    response = client.get(f"/api/ingest/status/{state['task_id']}")
                    response.raise_for_status()
                    status = response.json()
                    if status.get("status") == "SUCCESS":
                        result = status.get("result") or {}
                        if not result.get("source_id") or not (result.get("skipped") or result.get("messages", 0) >= 2):
                            raise RuntimeError("MemoryWeb completion lacks ingested source evidence")
                        break
                    if status.get("status") == "FAILURE":
                        raise RuntimeError("MemoryWeb ingestion failed; retained task ID needs investigation")
                    if time.monotonic() >= deadline:
                        raise RuntimeError("MemoryWeb still pending; next run will poll the same task")
                    time.sleep(2)
                atomic_json(state_file, {**state, "status": "completed", "result": result})
                cur.execute("UPDATE rag.bridge_queue SET status='done', processed_at=now() WHERE id=%s", (row["id"],))
                if row.get("query_log_id"):
                    cur.execute("UPDATE rag.query_log SET memoryweb_ingested=true WHERE id=%s", (row["query_log_id"],))
                conn.commit()
                completed += 1
        with conn.cursor() as cur:
            cur.execute("SELECT status,count(*) FROM rag.bridge_queue GROUP BY status")
            counts = dict(cur.fetchall())
        if counts.get("failed") or counts.get("processing"):
            raise RuntimeError("Bridge queue contains failed or ambiguous processing rows")
        return {"completed": completed, "queue": counts, "authenticated": True}


def sync_s3(source):
    import boto3
    from botocore.config import Config
    client = boto3.client("s3", config=Config(connect_timeout=10, read_timeout=60, retries={"max_attempts": 2}))
    Path(source["destination"]).mkdir(parents=True, exist_ok=True)
    counts = {"examined": 0, "downloaded": 0}
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=source["bucket"]):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            ext = PurePosixPath(key).suffix.lower()
            if key.endswith("/") or ext in source.get("skip", []) or (source.get("include") and ext not in source["include"]):
                continue
            path = safe_destination(source["destination"], key)
            counts["examined"] += 1
            stamp = obj["LastModified"].timestamp()
            if path.exists() and path.stat().st_size == obj["Size"] and path.stat().st_mtime >= stamp:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_name(path.name+".sync.tmp")
            client.download_file(source["bucket"], key, str(temp))
            if temp.stat().st_size != obj["Size"]:
                raise RuntimeError("S3 downloaded size mismatch")
            os.replace(temp, path)
            os.utime(path, (stamp, stamp))
            counts["downloaded"] += 1
    return counts


def sync_ssh(source):
    """Native OpenSSH/SFTP uses the existing login and host-key policy."""
    import shlex
    openssh = Path(os.environ["SystemRoot"]) / "System32" / "OpenSSH"
    script = ("import os,json,sys; p=os.path.abspath(os.path.expanduser(sys.argv[1])); "
              "print(json.dumps({'root':p,'files':[{'name':e.name,'size':e.stat().st_size,"
              "'mtime':e.stat().st_mtime} for e in os.scandir(p) if e.is_file(follow_symlinks=False)]}))")
    command = "python3 -c " + shlex.quote(script) + " " + shlex.quote(source["directory"])
    manifest = json.loads(run([str(openssh / "ssh.exe"), "-o", "BatchMode=yes", "-o",
                               "ConnectTimeout=8", source["host"], command], timeout=30))
    counts = {"examined": 0, "downloaded": 0}
    for item in manifest["files"]:
        name = item["name"]
        if PurePosixPath(name).suffix.lower() not in source["include"]:
            continue
        if any(c in name for c in "\r\n:*?[]") or "/" in name:
            raise ValueError("Unsupported remote filename")
        target = safe_destination(source["destination"], name)
        counts["examined"] += 1
        if target.exists() and target.stat().st_size == item["size"] and target.stat().st_mtime >= item["mtime"]:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name(target.name+".sync.tmp")
        remote = source["host"]+":"+manifest["root"]+"/"+name
        run([str(openssh / "scp.exe"), "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
             remote, str(temp)], timeout=180)
        if temp.stat().st_size != item["size"]:
            raise RuntimeError("SFTP downloaded size mismatch")
        os.replace(temp, target)
        os.utime(target, (item["mtime"], item["mtime"]))
        counts["downloaded"] += 1
    return counts


def personal(cfg):
    outcomes = []
    for source in cfg["sources"]:
        try:
            stats = sync_s3(source) if source["type"] == "s3" else sync_ssh(source)
            outcomes.append({"name": source["name"], "status": "success", **stats})
        except Exception as e:
            outcomes.append({"name": source["name"], "status": "failed", "error_type": type(e).__name__,
                             "error": str(e) if isinstance(e, RuntimeError) else "Source access failed"})
        atomic_json(cfg["progress"], {"sources": outcomes, "stage": "sync", "updated": datetime.now(timezone.utc).isoformat()})
    if any(s["status"] != "success" for s in outcomes):
        raise RuntimeError("Required sync sources failed; see personal source receipts")
    sys.path.insert(0, cfg["project"])
    from rag_ingest import ingest_collection
    from src.embedder import embed_collection
    from src.config import get_config
    # Every required download destination must participate in re-ingestion.
    # The legacy patent bucket destination was absent from the collection paths.
    paths = get_config()["collections"]["personal"]["paths"]
    known = [Path(p["path"] if isinstance(p, dict) else p).resolve() for p in paths]
    for source in cfg["sources"]:
        destination = Path(source["destination"]).resolve()
        if not any(destination == p or destination.is_relative_to(p) for p in known):
            paths.append(str(destination))
            known.append(destination)
    with contextlib.closing(db_connect(cfg["project"])) as conn:
        parsed = ingest_collection("personal", conn)
        if parsed.get("errors") or not parsed.get("files"):
            raise RuntimeError("Personal parse incomplete; see private task log")
        embedded = embed_collection(conn, "personal")
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM rag.chunks WHERE collection='personal' AND embedding IS NULL")
            remaining = cur.fetchone()[0]
        if remaining:
            raise RuntimeError(f"Personal embeddings incomplete: {remaining} pending")
    return {"sources": outcomes, "parse": parsed, "embedded": embedded, "remaining": 0}


def watchdog(cfg):
    import httpx
    def health():
        try:
            r = httpx.get(cfg["health"], timeout=8)
            return r.status_code == 200 and r.json().get("status") == "ok" and r.json().get("db") == "ok"
        except (httpx.HTTPError, ValueError):
            return False
    # Never kill a listener or launch a competing installation.
    ps = cfg["powershell"]
    identity = json.loads(run([ps, "-NoProfile", "-Command",
        "Get-ItemProperty -LiteralPath 'HKLM:\\SYSTEM\\CurrentControlSet\\Services\\UltraRAG-API\\Parameters' | Select-Object Application,AppDirectory,AppParameters | ConvertTo-Json -Compress"]))
    if Path(identity["AppDirectory"]).resolve() != Path(cfg["project"]).resolve() or Path(identity["Application"]).resolve() != Path(cfg["python"]).resolve():
        raise RuntimeError("UltraRAG service identity changed; restart refused")
    if not service_restart_access("UltraRAG-API"):
        raise RuntimeError("Task account lacks UltraRAG-API restart rights; administrator repair required")
    if health():
        return {"healthy": True, "restarted": False, "service": "UltraRAG-API"}
    state_path = Path(cfg["state"])
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    recent = [t for t in state.get("restarts", []) if t > time.time()-900]
    if len(recent) >= 3:
        raise RuntimeError("UltraRAG restart circuit breaker: 3 attempts in 15 minutes")
    atomic_json(state_path, {"restarts": recent+[time.time()]})
    run([ps, "-NoProfile", "-Command", "$ErrorActionPreference='Stop'; Restart-Service -Name 'UltraRAG-API' -ErrorAction Stop"], timeout=90)
    deadline = time.monotonic()+120
    while time.monotonic() < deadline:
        if health():
            return {"healthy": True, "restarted": True, "service": "UltraRAG-API"}
        time.sleep(3)
    raise RuntimeError("UltraRAG service restarted but did not become healthy")


def service_restart_access(name):
    """Check the exact SCM access mask without restarting a healthy service."""
    import ctypes
    from ctypes import wintypes
    adv = ctypes.WinDLL("advapi32", use_last_error=True)
    adv.OpenSCManagerW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD]
    adv.OpenSCManagerW.restype = wintypes.HANDLE
    adv.OpenServiceW.argtypes = [wintypes.HANDLE, wintypes.LPCWSTR, wintypes.DWORD]
    adv.OpenServiceW.restype = wintypes.HANDLE
    adv.CloseServiceHandle.argtypes = [wintypes.HANDLE]
    manager = adv.OpenSCManagerW(None, None, 1)
    if not manager:
        return False
    try:
        service = adv.OpenServiceW(manager, name, 0x34)
        if service:
            adv.CloseServiceHandle(service)
            return True
        return False
    finally:
        adv.CloseServiceHandle(manager)


JOBS = {"echo": echo, "bridge": bridge, "personal": personal, "watchdog": watchdog}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("job", choices=JOBS)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    output = Path(cfg["receipts"])
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(handlers=[RotatingFileHandler(output / f"{args.job}.log", maxBytes=2000000,
                        backupCount=3, encoding="utf-8")], level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    receipt = {"job": args.job, "started": datetime.now(timezone.utc).isoformat(), "status": "running"}
    try:
        with job_lock(output / f"{args.job}.lock"):
            atomic_json(output / f"{args.job}.latest.json", receipt)
            try:
                receipt["result"] = JOBS[args.job](cfg[args.job])
                receipt["status"] = "success"
            except Exception as e:
                receipt["status"] = "failed"
                receipt["error_type"] = type(e).__name__
                receipt["error"] = str(e) if isinstance(e, RuntimeError) else "Task dependency or file access failed"
                if isinstance(e, OSError) and e.filename:
                    receipt["file"] = str(e.filename)
            receipt["finished"] = datetime.now(timezone.utc).isoformat()
            atomic_json(output / f"{args.job}.latest.json", receipt)
            atomic_json(output / f"{args.job}.{time.time_ns()}.json", receipt)
            logging.info("%s", json.dumps(receipt))
    except OSError:
        # Do not overwrite the active run's receipt when another process owns the lock.
        print(json.dumps({"job": args.job, "status": "busy_or_receipt_unwritable"}))
        return 75
    print(json.dumps(receipt))
    return 0 if receipt["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
