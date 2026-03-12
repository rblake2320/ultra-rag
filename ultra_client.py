#!/usr/bin/env python3
"""
Ultra RAG Universal Client
===========================
Works from any system — laptops, Spark nodes, VPS, remote machines.
Auto-detects whether to use:
  1. Local server (localhost:8300) — if running on the hub
  2. LAN server (192.168.12.198:8300) — if on the same network
  3. Remote tunnel URL — if remote (set ULTRA_RAG_URL env var)
  4. Offline SQLite — if no server reachable

Usage:
    from ultra_client import UltraClient
    client = UltraClient()
    results = client.search("What is the Equipment ID validation rule?")

    # Or CLI:
    python ultra_client.py "your question"
    python ultra_client.py "your question" --collection imds --top-k 10
    python ultra_client.py --register --system-id laptop-office --role client
"""
import argparse
import json
import os
import socket
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Dependencies ──────────────────────────────────────────────────────────────
try:
    import httpx
    _HAS_HTTPX = True
except ImportError:
    _HAS_HTTPX = False

# ── Configuration ─────────────────────────────────────────────────────────────
# Priority order for server discovery:
_CANDIDATE_URLS = [
    os.environ.get("ULTRA_RAG_URL", ""),          # 1. Explicit env override
    "http://localhost:8300",                        # 2. Local (running on hub)
    "http://192.168.12.198:8300",                  # 3. LAN (same network as hub)
    "http://192.168.12.132:8300",                  # 4. Spark-1 if it runs one
]
_API_KEY   = os.environ.get("ULTRA_RAG_API_KEY", "")
_SYSTEM_ID = os.environ.get("ULTRA_RAG_SYSTEM_ID", socket.gethostname())

# Offline SQLite paths (checked in order)
_SQLITE_PATHS = [
    Path(__file__).parent / "imds-corpus.db",
    Path("E:/imds-agent/imds-corpus.db"),
    Path("D:/rag-ingest/imds-corpus.db"),
    Path.home() / "imds-corpus.db",
]


# =============================================================================
# UltraClient
# =============================================================================

class UltraClient:
    """
    Universal client for the Ultra RAG hub.
    Falls back gracefully to offline SQLite if hub unreachable.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key:  Optional[str] = None,
        system_id: Optional[str] = None,
        timeout:  float = 30.0,
        auto_register: bool = True,
    ):
        self.api_key   = api_key or _API_KEY
        self.system_id = system_id or _SYSTEM_ID
        self.timeout   = timeout
        self._offline_db: Optional[str] = None
        self._base_url: Optional[str] = None

        # Resolve server URL
        if base_url:
            self._base_url = base_url.rstrip("/")
        else:
            self._base_url = self._discover_server()

        if self._base_url:
            print(f"[UltraClient] Connected to: {self._base_url}")
            if auto_register:
                self._register()
        else:
            self._offline_db = self._find_sqlite()
            if self._offline_db:
                print(f"[UltraClient] Offline mode — SQLite: {self._offline_db}")
            else:
                print("[UltraClient] WARNING: No server or SQLite found — search will return empty")

    # ── Server discovery ──────────────────────────────────────────────────────

    def _discover_server(self) -> Optional[str]:
        """Try each candidate URL, return the first that responds to /api/health."""
        if not _HAS_HTTPX:
            return None
        for url in _CANDIDATE_URLS:
            if not url:
                continue
            try:
                resp = httpx.get(f"{url}/api/health", timeout=2.0,
                                 headers=self._headers())
                if resp.status_code in (200, 401):  # 401 = server is up, just needs key
                    return url.rstrip("/")
            except Exception:
                continue
        return None

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["X-API-Key"] = self.api_key
        return h

    def _find_sqlite(self) -> Optional[str]:
        for p in _SQLITE_PATHS:
            if p.exists():
                return str(p)
        return None

    # ── Registration ──────────────────────────────────────────────────────────

    def _register(self) -> None:
        """Register this system with the hub (fire-and-forget)."""
        try:
            payload = {
                "system_id":    self.system_id,
                "role":         "client",
                "host":         socket.gethostname(),
                "capabilities": ["search"],
            }
            # Add offline SQLite capability if available
            if self._find_sqlite():
                payload["capabilities"].append("offline-sqlite")
            httpx.post(f"{self._base_url}/api/systems/register",
                       json=payload, headers=self._headers(), timeout=3.0)
        except Exception:
            pass  # non-critical

    def heartbeat(self) -> bool:
        """Ping the hub to keep this system's last_seen fresh."""
        if not self._base_url:
            return False
        try:
            resp = httpx.post(f"{self._base_url}/api/systems/heartbeat/{self.system_id}",
                              headers=self._headers(), timeout=3.0)
            return resp.status_code == 200
        except Exception:
            return False

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query:      str,
        collection: str  = "imds",
        top_k:      int  = 5,
        strategy:   Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search the Ultra RAG index.
        Falls back to offline SQLite FTS if hub unreachable.
        """
        # Try hub first
        if self._base_url and _HAS_HTTPX:
            try:
                payload = {"query": query, "collection": collection,
                           "top_k": top_k, "strategy": strategy}
                resp = httpx.post(f"{self._base_url}/api/search",
                                  json=payload, headers=self._headers(),
                                  timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                return data.get("results", [])
            except Exception as exc:
                print(f"[UltraClient] Hub search failed: {exc} — trying offline SQLite")

        # Offline fallback
        if self._offline_db:
            return self._sqlite_search(query, top_k)

        return []

    def _sqlite_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Full-text search against local SQLite corpus."""
        try:
            conn = sqlite3.connect(self._offline_db)
            cur  = conn.cursor()
            # Try FTS5
            try:
                cur.execute("""
                    SELECT chunk_id, content, content_type, screen_tag,
                           bm25(chunks_fts) AS score
                    FROM   chunks_fts
                    WHERE  chunks_fts MATCH ?
                    ORDER  BY score
                    LIMIT  ?
                """, (query, top_k))
            except Exception:
                # Fallback to LIKE
                cur.execute("""
                    SELECT id, content, content_type, screen_tag, 0.5
                    FROM   chunks
                    WHERE  content LIKE ?
                    LIMIT  ?
                """, (f"%{query}%", top_k))
            rows = cur.fetchall()
            conn.close()
            return [
                {"id": r[0], "content": r[1], "content_type": r[2],
                 "chunk_metadata": {"imds_screens": [r[3]]} if r[3] else {},
                 "score": abs(float(r[4])), "source": "offline-sqlite"}
                for r in rows
            ]
        except Exception as exc:
            print(f"[UltraClient] SQLite search error: {exc}")
            return []

    # ── System info ───────────────────────────────────────────────────────────

    def health(self) -> Dict:
        """Get hub health status."""
        if not self._base_url:
            return {"status": "offline", "mode": "sqlite" if self._offline_db else "none"}
        try:
            resp = httpx.get(f"{self._base_url}/api/health",
                             headers=self._headers(), timeout=5.0)
            return resp.json()
        except Exception as exc:
            return {"status": "error", "detail": str(exc)}

    def systems(self) -> Dict:
        """List all registered systems."""
        if not self._base_url:
            return {}
        try:
            resp = httpx.get(f"{self._base_url}/api/systems",
                             headers=self._headers(), timeout=5.0)
            return resp.json()
        except Exception:
            return {}

    def stats(self, collection: str = "imds") -> Dict:
        """Get collection statistics."""
        if not self._base_url:
            return {"mode": "offline"}
        try:
            resp = httpx.get(f"{self._base_url}/api/stats",
                             params={"collection": collection},
                             headers=self._headers(), timeout=10.0)
            return resp.json()
        except Exception as exc:
            return {"error": str(exc)}

    def memory_recall(self, query: str, top_k: int = 3) -> List[Dict]:
        """Pull relevant memories from MemoryWeb via hub."""
        if not self._base_url:
            return []
        try:
            resp = httpx.get(f"{self._base_url}/api/memory/recall",
                             params={"q": query, "top_k": top_k},
                             headers=self._headers(), timeout=5.0)
            if resp.status_code == 200:
                return resp.json().get("memories", [])
        except Exception:
            pass
        return []

    def __repr__(self):
        mode = f"hub={self._base_url}" if self._base_url else f"offline={self._offline_db}"
        return f"UltraClient(system_id={self.system_id!r}, {mode})"


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Ultra RAG Universal Client")
    parser.add_argument("query", nargs="?", help="Search query")
    parser.add_argument("--collection", "-c", default="imds")
    parser.add_argument("--top-k",      "-n", type=int, default=5)
    parser.add_argument("--strategy",   "-s", default=None,
                        choices=["hybrid", "kg_local", "kg_global", "hyde",
                                 "multi_hop", "compound", None])
    parser.add_argument("--url",    default=None, help="Override hub URL")
    parser.add_argument("--system-id", default=None)
    parser.add_argument("--register",  action="store_true",
                        help="Register this system and exit")
    parser.add_argument("--health",    action="store_true")
    parser.add_argument("--systems",   action="store_true")
    parser.add_argument("--stats",     action="store_true")
    args = parser.parse_args()

    client = UltraClient(
        base_url=args.url,
        system_id=args.system_id,
        auto_register=True,
    )

    if args.health or not args.query:
        h = client.health()
        print(json.dumps(h, indent=2))
        if args.systems:
            print("\n--- Systems ---")
            print(json.dumps(client.systems(), indent=2))
        if args.stats:
            print("\n--- Stats ---")
            print(json.dumps(client.stats(args.collection), indent=2))
        if args.register:
            print(f"[OK] Registered as: {client.system_id}")
        if not args.query:
            return

    t0 = time.time()
    results = client.search(args.query, args.collection, args.top_k, args.strategy)
    elapsed = time.time() - t0

    print(f"\nQuery: {args.query!r}")
    print(f"Results: {len(results)}  ({elapsed*1000:.0f}ms)\n")

    for i, r in enumerate(results, 1):
        score   = r.get("score", 0)
        ctype   = r.get("content_type", "")
        screens = r.get("chunk_metadata", {})
        source  = r.get("source", "hub")
        content = r.get("content", "")[:300]
        print(f"[{i}] score={score:.4f}  type={ctype}  source={source}")
        if screens:
            print(f"     screens={screens}")
        print(f"     {content}")
        print()


if __name__ == "__main__":
    main()
