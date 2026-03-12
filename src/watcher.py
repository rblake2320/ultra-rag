"""
Polling-based folder watcher. Uses file hashing to detect changes.
Polling (not watchdog events) avoids triple-firing on OneDrive sync.
"""
import hashlib
import logging
import time
from pathlib import Path

log = logging.getLogger(__name__)


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _is_stable(path: Path, delay: int) -> bool:
    """Return True if the file size hasn't changed in `delay` seconds."""
    try:
        size1 = path.stat().st_size
        time.sleep(delay)
        size2 = path.stat().st_size
        return size1 == size2
    except Exception:
        return False


def watch(collection: str, conn, ingest_fn) -> None:
    """
    Poll the configured paths for a collection. Call ingest_fn(path, conn)
    on new/changed files. Runs forever (Ctrl+C to stop).
    """
    from .config import get_config
    from .parsers import PARSERS

    cfg      = get_config()
    coll_cfg = cfg["collections"].get(collection, {})
    paths    = [Path(p) for p in coll_cfg.get("paths", [])]
    excl_dirs = set(coll_cfg.get("exclude_dirs", []))
    skip_files= set(coll_cfg.get("skip_files", []))
    poll_sec  = cfg["watcher"]["poll_interval"]
    stab_sec  = cfg["watcher"]["stability_delay"]

    # Track known file → hash
    known: dict[str, str] = {}

    log.info(f"Watcher started for '{collection}' — polling every {poll_sec}s")

    while True:
        for base_path in paths:
            for fpath in sorted(base_path.rglob("*")):
                if not fpath.is_file():
                    continue
                rel = fpath.relative_to(base_path)
                if any(d in excl_dirs for d in rel.parts):
                    continue
                if fpath.name in skip_files:
                    continue
                if fpath.suffix.lower() not in PARSERS:
                    continue

                key = str(fpath)
                try:
                    current_hash = _hash_file(fpath)
                except Exception as e:
                    log.warning(f"Cannot hash {fpath}: {e}")
                    continue

                if known.get(key) != current_hash:
                    if _is_stable(fpath, stab_sec):
                        log.info(f"Change detected: {fpath.name}")
                        try:
                            ingest_fn(fpath, conn, collection)
                            known[key] = current_hash
                        except Exception as e:
                            log.error(f"Ingest failed for {fpath.name}: {e}")
                    # else: file still changing, check again next poll

        time.sleep(poll_sec)
