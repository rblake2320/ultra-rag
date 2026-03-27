#!/usr/bin/env python3
"""
Ultra RAG API Server — port 8300.

Provides a REST API over the full Ultra RAG pipeline including intelligent
query routing, knowledge graph browsing, provenance drill-down, and
synthetic evaluation.

Endpoints:
  GET  /                 — Dashboard UI (static/index.html)
  POST /api/search       — Ultra search with all strategies
  POST /api/ingest       — Trigger full 7-stage pipeline
  GET  /api/entities     — Browse knowledge graph entities
  GET  /api/communities  — Browse detected communities
  GET  /api/collections  — List all collection names
  GET  /api/provenance/{id} — Drill-down confidence decomposition
  POST /api/eval         — Run evaluation suite (generate + RAGAS)
  POST /api/upload       — Upload and ingest a document file
  GET  /api/health       — System health check
  GET  /api/stats        — Collection statistics

Run:
  python ultra_server.py
  python ultra_server.py --port 8300 --host 0.0.0.0
"""
import asyncio
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Logging ───────────────────────────────────────────────────────────────────
logs_dir = Path(__file__).parent / "logs"
logs_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(logs_dir / "ultra_server.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── Project root ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

# ── FastAPI imports ───────────────────────────────────────────────────────────
try:
    import uvicorn
    import httpx
    from fastapi import FastAPI, File, Form, HTTPException, Query, Request, Security, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
    from fastapi.security.api_key import APIKeyHeader
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field
except ImportError as exc:
    print(f"ERROR: FastAPI/uvicorn not installed. Run: pip install fastapi uvicorn httpx")
    print(f"  {exc}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# API Key Auth — multi-tenant collection isolation
# ---------------------------------------------------------------------------
# Global admin key (required for /api/systems, /api/ingest, /api/eval):
#   ULTRA_RAG_API_KEY=<key>
#
# Per-collection keys (required to access that collection's data):
#   ULTRA_RAG_KEY_<CORPUS>=<key>   → only allows access to the named collection
#   ULTRA_RAG_KEY_PERSONAL=<key>   → only allows access to 'personal' collection
#   ULTRA_RAG_KEY_<NAME>=<key>     → pattern for future clients
#
# If NO keys are configured: open mode (localhost/LAN only — never expose this to internet)
# If keys ARE configured: all external requests must supply matching X-API-Key header

_API_KEY        = os.environ.get("ULTRA_RAG_API_KEY", "")
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Build collection→key map from env vars matching ULTRA_RAG_KEY_<COLLECTION>
_COLLECTION_KEYS: Dict[str, str] = {}
for _env_k, _env_v in os.environ.items():
    if _env_k.startswith("ULTRA_RAG_KEY_") and _env_v:
        _col = _env_k[len("ULTRA_RAG_KEY_"):].lower()
        _COLLECTION_KEYS[_col] = _env_v


def _is_localhost(request: Request) -> bool:
    """Return True ONLY for direct local connections — no proxy, no tunnel.

    Cloudflare tunnel (cloudflared) connects from 127.0.0.1 but adds
    CF-Connecting-IP / X-Forwarded-For headers with the real remote IP.
    Any request that passed through a proxy/tunnel is treated as external.
    """
    client_ip = request.client.host if request.client else ""
    if client_ip not in ("127.0.0.1", "::1"):
        return False
    # If any forwarding header is present, the request came through a proxy/tunnel
    forwarding_headers = (
        "cf-connecting-ip",      # Cloudflare real IP
        "x-forwarded-for",       # Generic proxy
        "x-real-ip",             # Nginx proxy
        "x-forwarded-host",      # Proxy host header
    )
    headers_lower = {k.lower(): v for k, v in request.headers.items()}
    for h in forwarding_headers:
        if h in headers_lower:
            return False
    return True


def _check_api_key(request: Request, api_key: str = Security(_api_key_header)) -> None:
    """Global auth check — used for admin/system endpoints."""
    if not _API_KEY:
        return   # no global key configured — open mode
    if _is_localhost(request):
        return   # localhost always trusted
    if api_key != _API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def _check_collection_access(collection: str, request: Request,
                              api_key: str) -> None:
    """
    Collection-level auth check.

    Rules:
      1. Localhost always allowed (local development / CLI tools)
      2. Admin key allows access to any collection
      3. Per-collection key allows access only to that collection
      4. If no keys configured at all: open (should only be LAN)
      5. Otherwise: reject
    """
    if _is_localhost(request):
        return

    col = collection.lower()
    col_key = _COLLECTION_KEYS.get(col, "")

    # No auth configured at all → open (warn: should not be internet-exposed)
    if not _API_KEY and not col_key and not _COLLECTION_KEYS:
        log.warning("No API keys configured — server is open. Set ULTRA_RAG_KEY_%s.", col.upper())
        return

    # Admin key → full access
    if _API_KEY and api_key == _API_KEY:
        return

    # Per-collection key → access only to that collection
    if col_key and api_key == col_key:
        return

    # Wrong key or no key
    raise HTTPException(
        status_code=401,
        detail=f"Access to collection '{collection}' requires a valid API key",
    )


# ---------------------------------------------------------------------------
# MemoryWeb Bridge (optional — non-blocking if MemoryWeb is down)
# ---------------------------------------------------------------------------
_MEMORYWEB_URL = os.environ.get("MEMORYWEB_URL", "http://localhost:8100")
_MEMORYWEB_ENABLED = os.environ.get("MEMORYWEB_ENABLED", "true").lower() == "true"


async def _memoryweb_recall(query: str, top_k: int = 3) -> List[Dict]:
    """Pull relevant memories from MemoryWeb to augment RAG context. Silent on failure."""
    if not _MEMORYWEB_ENABLED:
        return []
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.post(
                f"{_MEMORYWEB_URL}/api/search",
                json={"query": query, "top_k": top_k, "force_tier": 3},
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("results", [])
    except Exception:
        pass   # MemoryWeb down or slow — never fail the RAG query
    return []


async def _memoryweb_store(query: str, answer_summary: str, collection: str,
                            query_log_id: int | None = None) -> None:
    """Store query+answer in MemoryWeb via durable bridge_queue. Fire-and-forget."""
    if not _MEMORYWEB_ENABLED:
        return
    try:
        import json as _json
        # Sanitize inputs
        safe_query = query.replace("\x00", "")[:2000]
        safe_answer = answer_summary.replace("\x00", "")[:3000]

        # Synthesize a 2-sentence summary via local LLM (gemma3 is fast, already loaded)
        summary = await asyncio.to_thread(
            _synthesize_bridge_summary, safe_query, safe_answer, collection
        )

        # Write to durable queue (postgres) — survives server restarts
        conn = _get_conn()
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO rag.bridge_queue
                   (query_text, collection, summary, query_log_id, status)
                   VALUES (%s, %s, %s, %s, 'pending')""",
                (safe_query, collection, summary, query_log_id)
            )
        conn.commit()
        log.debug("bridge_queue: queued query for collection=%s", collection)
    except Exception as exc:
        log.warning("_memoryweb_store failed (non-fatal): %s", exc)


def _synthesize_bridge_summary(query: str, answer: str, collection: str) -> str:
    """Call gemma3 via Ollama to produce a 2-sentence memory-quality summary. Sync."""
    try:
        prompt = (
            f"Summarize in exactly 2 sentences what was asked and what was found. "
            f"Be specific and factual. No preamble.\n\n"
            f"Query: {query}\n\n"
            f"Result: {answer[:1500]}"
        )
        resp = httpx.post(
            "http://localhost:11434/api/generate",
            json={"model": "mistral:7b", "prompt": prompt, "stream": False,
                  "options": {"num_predict": 100, "temperature": 0.1}},
            timeout=30.0,
        )
        if resp.status_code == 200:
            text = resp.json().get("response", "").strip()
            if text:
                return text[:500]
    except Exception as exc:
        log.warning("summarizer failed, using fallback: %s", exc)
    # Fallback: truncated answer as summary
    return f"Query about {collection}: {query[:100]}. Result: {answer[:300]}"


# ---------------------------------------------------------------------------
# System Registry — tracks all nodes that connect to this hub
# ---------------------------------------------------------------------------
_SYSTEMS: Dict[str, Dict] = {
    "windows-pc": {
        "role": "hub",
        "host": "localhost",
        "services": ["ultra-rag:8300", "ollama:11434", "postgres:5432", "memoryweb:8100"],
        "gpu": "RTX 5090 32GB",
        "last_seen": datetime.utcnow().isoformat(),
    },
    "spark-1": {
        "role": "gpu-cluster",
        "host": "192.168.12.132",
        "services": ["ollama:11434", "mcp-agents"],
        "gpu": "GB10 119.7GB",
        "last_seen": None,
    },
    "spark-2": {
        "role": "gpu-cluster",
        "host": "10.0.0.2",
        "services": ["ollama:11434"],
        "gpu": "GB10 119.7GB",
        "last_seen": None,
    },
    "vps": {
        "role": "gateway",
        "host": "76.13.118.222",
        "services": ["nginx"],
        "last_seen": None,
    },
}
_LAPTOPS: Dict[str, Dict] = {}  # dynamically registered by laptops

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ResultItem(BaseModel):
    id:                   Optional[int]   = None
    content:              str             = ""
    content_type:         str             = "text"
    score:                float           = 0.0
    context_prefix:       Optional[str]   = None
    chunk_metadata:       Optional[Dict]  = None
    parent_content:       Optional[str]   = None
    rerank_score:         Optional[float] = None
    utility_score:        Optional[float] = None
    provenance_components: Optional[Dict] = None


class SearchRequest(BaseModel):
    query:              str
    collection:         str             = "my-docs"
    top_k:              int             = Field(5, ge=1, le=50)
    strategy:           Optional[str]   = None   # None = auto-route
    include_provenance: bool            = False
    include_parent:     bool            = False
    no_rerank:          bool            = False
    no_crag:            bool            = False


class SearchResponse(BaseModel):
    results:            List[ResultItem]
    strategy_used:      str
    quality_score:      float
    latency_ms:         int
    provenance_chain_id: Optional[int]  = None
    classification:     Optional[Dict]  = None


class IngestRequest(BaseModel):
    collection:  str
    stages:      str  = "all"
    reprocess:   bool = False


class IngestResponse(BaseModel):
    stages_completed:    List[str]
    stats:               Dict[str, Any]
    total_time_seconds:  float


class EntityQuery(BaseModel):
    collection:   str           = "my-docs"
    search_term:  Optional[str] = None
    entity_type:  Optional[str] = None
    limit:        int           = Field(20, ge=1, le=200)


class CommunityQuery(BaseModel):
    collection: str           = "my-docs"
    level:      Optional[int] = None
    limit:      int           = Field(20, ge=1, le=200)


class EvalRequest(BaseModel):
    collection:   str           = "my-docs"
    n_questions:  int           = Field(20, ge=1, le=500)
    run_name:     Optional[str] = None
    run_eval:     bool          = True


class ProvenanceResponse(BaseModel):
    chain_id:           int
    query_log_id:       Optional[int]
    overall_confidence: float
    steps:              List[Dict]


class HealthResponse(BaseModel):
    status:    str
    db:        str
    ollama:    str
    pgvector:  str
    timestamp: str
    memoryweb: str = "unknown"
    systems_online: int = 0


class SystemRegisterRequest(BaseModel):
    system_id:   str                  # e.g. "laptop-1", "spark-1"
    role:        str   = "client"
    host:        Optional[str] = None
    services:    List[str] = []
    gpu:         Optional[str] = None
    capabilities: List[str] = []     # e.g. ["search", "ingest", "offline-sqlite"]


class SystemsResponse(BaseModel):
    hub:     Dict[str, Any]
    systems: Dict[str, Any]
    laptops: Dict[str, Any]
    total:   int


class StatsResponse(BaseModel):
    collection:         str
    chunks:             int
    documents:          int
    embedded_chunks:    int
    entities:           int
    relationships:      int
    communities:        int
    parent_chunks:      int
    eval_questions:     int
    # Pipeline progress fields
    kg_chunks_done:     int = 0   # chunks with KG entity links
    contextual_done:    int = 0   # chunks with contextual context
    summaries:          int = 0   # RAPTOR summary nodes


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Ultra RAG API",
    description="7-stage RAG pipeline with KG, RAPTOR, CRAG, and RAGAS evaluation",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static files ───────────────────────────────────────────────────────────────
_static_dir = Path(__file__).parent / "static"
_static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


# ---------------------------------------------------------------------------
# DB pool (one persistent connection per worker — FastAPI is single-process)
# ---------------------------------------------------------------------------

_conn = None


def _get_conn():
    global _conn
    if _conn is None or _conn.closed:
        from src.db import get_conn  # noqa: PLC0415
        _conn = get_conn()
    try:
        # Quick liveness check
        _conn.cursor().execute("SELECT 1")
    except Exception:
        from src.db import get_conn  # noqa: PLC0415
        _conn = get_conn()
    return _conn


# ---------------------------------------------------------------------------
# RLS-scoped connection for query operations
# ---------------------------------------------------------------------------

_RAG_APP_DSN = (
    "host=localhost port=5432 dbname=postgres "
    "user=rag_app password=ragapp_rls_2026"
)

_scoped_conn = None


def _get_scoped_conn(collection: str):
    """
    Return a psycopg2 connection authenticated as rag_app (no BYPASSRLS),
    with the RLS session variable set so only rows for `collection` are visible.

    When collection is '' (empty string), admin mode: all rows visible.
    This is called for all /api/search and /api/stats query paths.
    Ingest paths use _get_conn() (superuser) so they can write freely.
    """
    global _scoped_conn
    import psycopg2  # noqa: PLC0415
    if _scoped_conn is None or _scoped_conn.closed:
        _scoped_conn = psycopg2.connect(_RAG_APP_DSN)
    try:
        _scoped_conn.cursor().execute("SELECT 1")
    except Exception:
        _scoped_conn = psycopg2.connect(_RAG_APP_DSN)

    # Set the RLS session variable — safe_col prevents SQL injection
    safe_col = collection.replace("'", "''") if collection else ""
    with _scoped_conn.cursor() as cur:
        cur.execute(f"SET app.rag_collection = '{safe_col}'")
    _scoped_conn.commit()
    return _scoped_conn


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

def _cleanup_mw_sessions() -> None:
    """Remove stale mw-session JSONL files older than 24 h (called at startup)."""
    import time
    session_dir = str(Path(__file__).resolve().parent / "data" / "mw-sessions")
    if not os.path.isdir(session_dir):
        return
    cutoff = time.time() - 86400
    removed = 0
    for fname in os.listdir(session_dir):
        fp = os.path.join(session_dir, fname)
        try:
            if os.path.getmtime(fp) < cutoff:
                os.unlink(fp)
                removed += 1
        except OSError:
            pass
    if removed:
        log.info("Cleaned up %d stale mw-session file(s) older than 24h.", removed)


@app.on_event("startup")
async def startup_event():
    log.info("Ultra RAG Server starting up…")
    conn = _get_conn()
    await asyncio.to_thread(_ensure_schemas, conn)
    log.info("Schemas ready.")
    # Clean up stale MemoryWeb session files from previous runs
    await asyncio.to_thread(_cleanup_mw_sessions)
    # Warm up the reranker in background (avoids 5s first-query delay)
    async def _warm_reranker():
        try:
            from src.reranker import Reranker  # noqa: PLC0415
            rr = Reranker()
            rr.rerank("warmup", [{"id": 0, "content": "warmup passage"}], top_k=1)
            log.info("Reranker warmed up.")
        except Exception as exc:
            log.debug("Reranker warmup skipped: %s", exc)
    asyncio.create_task(_warm_reranker())


def _ensure_schemas(conn) -> None:
    from src.db       import create_schema        # noqa: PLC0415
    from src.db_ultra import create_ultra_schema  # noqa: PLC0415
    create_schema(conn)
    create_ultra_schema(conn)


# ---------------------------------------------------------------------------
# GET / — serve dashboard
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def root(request: Request):
    """Admin dashboard — localhost only. External requests get 403."""
    if not _is_localhost(request):
        raise HTTPException(status_code=403, detail="This endpoint is only accessible from localhost.")
    admin_path = _static_dir / "admin.html"
    if not admin_path.exists():
        return HTMLResponse(content="<h1>Admin dashboard not found</h1>", status_code=404)
    return FileResponse(str(admin_path), media_type="text/html")


@app.get("/admin", response_class=HTMLResponse, tags=["UI"])
async def admin_dashboard(request: Request):
    """Admin dashboard alias — localhost only."""
    if not _is_localhost(request):
        raise HTTPException(status_code=403, detail="Admin dashboard is only accessible from localhost.")
    admin_path = _static_dir / "admin.html"
    if not admin_path.exists():
        return HTMLResponse(content="<h1>Admin dashboard not found</h1>", status_code=404)
    return FileResponse(str(admin_path), media_type="text/html")


# ---------------------------------------------------------------------------
# /api/systems — System Registry
# ---------------------------------------------------------------------------

@app.get("/api/systems", tags=["System"])
async def get_systems(request: Request, api_key: str = Security(_api_key_header)):
    """Return all known systems — hub, fixed nodes, and dynamically registered laptops."""
    _check_api_key(request, api_key)
    _SYSTEMS["windows-pc"]["last_seen"] = datetime.utcnow().isoformat()
    return SystemsResponse(
        hub=_SYSTEMS["windows-pc"],
        systems={k: v for k, v in _SYSTEMS.items() if k != "windows-pc"},
        laptops=_LAPTOPS,
        total=len(_SYSTEMS) + len(_LAPTOPS),
    )


@app.post("/api/systems/register", tags=["System"])
async def register_system(req: SystemRegisterRequest, request: Request,
                           api_key: str = Security(_api_key_header)):
    """Laptops and remote systems call this on startup to register themselves."""
    _check_api_key(request, api_key)
    client_ip = request.client.host if request.client else "unknown"
    _LAPTOPS[req.system_id] = {
        "role":         req.role,
        "host":         req.host or client_ip,
        "services":     req.services,
        "gpu":          req.gpu,
        "capabilities": req.capabilities,
        "last_seen":    datetime.utcnow().isoformat(),
        "ip":           client_ip,
    }
    log.info("System registered: %s from %s", req.system_id, client_ip)
    return {"registered": req.system_id, "hub_url": "http://localhost:8300"}


@app.post("/api/systems/heartbeat/{system_id}", tags=["System"])
async def system_heartbeat(system_id: str, request: Request,
                            api_key: str = Security(_api_key_header)):
    """Lightweight ping — update last_seen for a registered system."""
    _check_api_key(request, api_key)
    target = _SYSTEMS.get(system_id) or _LAPTOPS.get(system_id)
    if target:
        target["last_seen"] = datetime.utcnow().isoformat()
    return {"ok": True, "system_id": system_id}


# ---------------------------------------------------------------------------
# /api/memory — MemoryWeb bridge
# ---------------------------------------------------------------------------

@app.get("/api/memory/status", tags=["Memory"])
async def memory_status(request: Request, api_key: str = Security(_api_key_header)):
    """Check if MemoryWeb is reachable."""
    _check_api_key(request, api_key)
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{_MEMORYWEB_URL}/api/status")
            if resp.status_code == 200:
                data = resp.json()
                return {"status": "online", "memories": data.get("memories", "?"),
                        "url": _MEMORYWEB_URL}
    except Exception as exc:
        pass
    return {"status": "offline", "url": _MEMORYWEB_URL}


@app.get("/api/memory/recall", tags=["Memory"])
async def memory_recall(q: str, top_k: int = 5, request: Request = None,
                         api_key: str = Security(_api_key_header)):
    """Query MemoryWeb for memories relevant to a search term."""
    _check_api_key(request, api_key)
    memories = await _memoryweb_recall(q, top_k)
    return {"query": q, "memories": memories, "count": len(memories)}


# ---------------------------------------------------------------------------
# /api/collections
# ---------------------------------------------------------------------------

@app.get("/api/collections", tags=["System"])
async def collections():
    """Return list of distinct collection names from rag.documents."""
    def _fetch() -> list:
        conn = _get_conn()
        with conn.cursor() as cur:
            try:
                cur.execute("SELECT DISTINCT collection FROM rag.documents ORDER BY collection")
                return [row[0] for row in cur.fetchall()]
            except Exception:
                return ["my-docs"]

    rows = await asyncio.to_thread(_fetch)
    return {"collections": rows}


# ---------------------------------------------------------------------------
# GET /api/entities  (also supports GET with query params for the dashboard)
# ---------------------------------------------------------------------------

@app.get("/api/entities", tags=["Knowledge Graph"])
async def entities_get(
    collection: str = Query("my-docs"),
    search_term: Optional[str] = Query(None),
    entity_type: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=200),
):
    """Browse knowledge graph entities via GET (dashboard-friendly)."""
    req = EntityQuery(collection=collection, search_term=search_term,
                      entity_type=entity_type, limit=limit)
    return await entities(req)


# ---------------------------------------------------------------------------
# GET /api/communities  (GET variant for dashboard)
# ---------------------------------------------------------------------------

@app.get("/api/communities", tags=["Knowledge Graph"])
async def communities_get(
    collection: str = Query("my-docs"),
    level: Optional[int] = Query(None),
    limit: int = Query(20, ge=1, le=200),
):
    """Browse detected communities via GET (dashboard-friendly)."""
    req = CommunityQuery(collection=collection, level=level, limit=limit)
    return await communities(req)


# ---------------------------------------------------------------------------
# /api/upload
# ---------------------------------------------------------------------------

@app.post("/api/upload", tags=["Ingestion"])
async def upload_document(
    file: UploadFile = File(...),
    collection: str = Form("my-docs"),
):
    """
    Accept a single document upload, save it to a temp directory, run the
    parse + embed stages of the ingest pipeline, and return chunk counts.
    """
    allowed_suffixes = {".pdf", ".docx", ".txt", ".md", ".csv", ".json", ".html"}
    # Sanitize filename — reject path traversal and null bytes
    safe_filename = Path(file.filename).name  # strip any directory components
    if "\x00" in safe_filename or ".." in safe_filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    suffix = Path(safe_filename).suffix.lower()
    if suffix not in allowed_suffixes:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}. Allowed: {', '.join(allowed_suffixes)}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="ultra_upload_"))
    dest    = tmp_dir / file.filename
    try:
        # Save uploaded bytes
        content = await file.read()
        dest.write_bytes(content)
        log.info("Uploaded %s (%d bytes) to %s for collection '%s'", file.filename, len(content), dest, collection)

        def _ingest(path: Path, coll: str) -> dict:
            from ultra_ingest import _run_parse, _run_embed  # noqa: PLC0415
            conn = _get_conn()

            # Temporarily register the file in the collection directory expected by the ingest pipeline.
            # We copy the file into the collection's raw data directory (data/<collection>/) if it exists,
            # otherwise we use a temp collection path.
            data_dir = Path(__file__).parent / "data" / coll
            data_dir.mkdir(parents=True, exist_ok=True)
            target = data_dir / path.name
            shutil.copy2(str(path), str(target))

            try:
                parse_stats = _run_parse(coll, conn)
                embed_stats = _run_embed(coll, conn)
                chunks_added = parse_stats.get("new_chunks", 0) if isinstance(parse_stats, dict) else 0
                embedded     = embed_stats.get("embedded", 0)   if isinstance(embed_stats, dict) else 0
                return {
                    "chunks_added": chunks_added,
                    "embedded":     embedded,
                    "message":      f"Ingested '{path.name}' into '{coll}': {chunks_added} chunks, {embedded} embedded.",
                }
            except Exception as exc:
                log.error("Upload ingest failed: %s", exc, exc_info=True)
                return {
                    "chunks_added": 0,
                    "embedded":     0,
                    "message":      f"Ingest error: {exc}",
                }

        result = await asyncio.to_thread(_ingest, dest, collection)
        return result

    finally:
        shutil.rmtree(str(tmp_dir), ignore_errors=True)


# ---------------------------------------------------------------------------
# /api/waitlist
# ---------------------------------------------------------------------------

class WaitlistEntry(BaseModel):
    name:     str
    email:    str
    org:      str  = ""
    use_case: str  = ""

_NOTIFY_EMAIL = "Rblake2320@aol.com"


def _send_waitlist_notification(entry: WaitlistEntry, ip: str) -> None:
    """Fire-and-forget SES notification — runs in a thread so it never blocks the response."""
    try:
        import boto3  # noqa: PLC0415
        ses = boto3.client("ses", region_name="us-east-1")
        use_label = entry.use_case.replace("_", " ").title() if entry.use_case else "Not specified"
        body = (
            f"New UltraRAG waitlist signup:\n\n"
            f"  Name:      {entry.name}\n"
            f"  Email:     {entry.email}\n"
            f"  Org:       {entry.org or 'Not provided'}\n"
            f"  Use case:  {use_label}\n"
            f"  IP:        {ip}\n"
            f"  Time:      {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n"
            f"Reply directly to their email to follow up."
        )
        ses.send_email(
            Source=_NOTIFY_EMAIL,
            Destination={"ToAddresses": [_NOTIFY_EMAIL]},
            Message={
                "Subject": {"Data": f"[UltraRAG] New signup: {entry.name} — {entry.org or entry.email}"},
                "Body":    {"Text": {"Data": body}},
            },
            ReplyToAddresses=[entry.email] if entry.email else [],
        )
        log.info("Waitlist notification sent for %s", entry.email)
    except Exception as exc:
        log.warning("Waitlist SES notification failed (non-fatal): %s", exc)


@app.post("/api/waitlist", tags=["Public"])
async def waitlist(entry: WaitlistEntry, request: Request):
    """
    Public endpoint — no auth required.
    Saves waitlist sign-ups to a local CSV file and sends an email notification via SES.
    """
    import csv  # noqa: PLC0415
    import threading  # noqa: PLC0415
    wl_file = Path(__file__).parent / "waitlist.csv"
    write_header = not wl_file.exists()
    ip = request.client.host if request.client else "unknown"
    with open(wl_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "name", "email", "org", "use_case", "ip"])
        if write_header:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.utcnow().isoformat(),
            "name":      entry.name[:120],
            "email":     entry.email[:120],
            "org":       entry.org[:120],
            "use_case":  entry.use_case[:80],
            "ip":        ip,
        })
    log.info("Waitlist signup: %s <%s> — %s (%s)", entry.name, entry.email, entry.org, entry.use_case)
    threading.Thread(target=_send_waitlist_notification, args=(entry, ip), daemon=True).start()
    return {"ok": True, "message": "You're on the waitlist. We'll try to get back to you within 24 hours."}


# /api/health
# ---------------------------------------------------------------------------

@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Check DB connection, Ollama reachability, and pgvector availability."""
    from datetime import datetime  # noqa: PLC0415

    db_status = "ok"
    pgvector_status = "ok"
    ollama_status = "ok"

    # DB check
    try:
        conn = _get_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        conn.commit()
    except Exception as exc:
        db_status = f"error: {exc}"

    # pgvector check
    try:
        conn = _get_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM rag.chunks WHERE embedding IS NOT NULL LIMIT 1")
        conn.commit()
    except Exception as exc:
        pgvector_status = f"error: {exc}"

    # Ollama check
    try:
        from src.config import get_config  # noqa: PLC0415
        cfg = get_config()
        ollama_url = cfg.get("llm", {}).get("ollama_url", "http://localhost:11434")
        resp = httpx.get(f"{ollama_url}/api/tags", timeout=5.0)
        resp.raise_for_status()
    except Exception as exc:
        ollama_status = f"error: {exc}"

    # MemoryWeb check — use /api/memories?limit=1 (fast, no Celery/Redis probe)
    memoryweb_status = "disabled"
    if _MEMORYWEB_ENABLED:
        try:
            resp = httpx.get(f"{_MEMORYWEB_URL}/api/memories?limit=1", timeout=5.0)
            memoryweb_status = "ok" if resp.status_code == 200 else f"error: {resp.status_code}"
        except Exception:
            memoryweb_status = "offline"

    overall = "ok" if all(s == "ok" for s in [db_status, pgvector_status, ollama_status]) else "degraded"
    systems_seen = sum(1 for s in list(_SYSTEMS.values()) + list(_LAPTOPS.values())
                       if s.get("last_seen"))

    return HealthResponse(
        status=overall,
        db=db_status,
        ollama=ollama_status,
        pgvector=pgvector_status,
        memoryweb=memoryweb_status,
        systems_online=systems_seen,
        timestamp=datetime.utcnow().isoformat(),
    )


# ---------------------------------------------------------------------------
# /api/stats
# ---------------------------------------------------------------------------

@app.get("/api/stats", response_model=StatsResponse, tags=["System"])
async def stats(request: Request, collection: str = Query("my-docs", description="Collection name"),
                api_key: str = Security(_api_key_header)):
    """Return chunk, entity, community, and parent-chunk counts."""
    _check_collection_access(collection, request, api_key)

    def _fetch(coll: str) -> StatsResponse:
        # Use RLS-scoped connection so stats cannot leak cross-collection data
        conn = _get_scoped_conn(coll)
        counts: dict[str, int] = {}

        def _q(sql, params=()):
            with conn.cursor() as cur:
                try:
                    cur.execute(sql, params)
                    row = cur.fetchone()
                    return row[0] if row else 0
                except Exception:
                    return -1

        counts["chunks"]          = _q("SELECT COUNT(*) FROM rag.chunks WHERE collection=%s", (coll,))
        counts["documents"]       = _q("SELECT COUNT(*) FROM rag.documents WHERE collection=%s", (coll,))
        counts["embedded_chunks"] = _q("SELECT COUNT(*) FROM rag.chunks WHERE collection=%s AND embedding IS NOT NULL", (coll,))
        counts["entities"]        = _q("SELECT COUNT(*) FROM rag.entities WHERE collection=%s", (coll,))
        counts["relationships"]   = _q("SELECT COUNT(*) FROM rag.relationships WHERE collection=%s", (coll,))
        counts["communities"]     = _q("SELECT COUNT(*) FROM rag.communities WHERE collection=%s", (coll,))
        counts["parent_chunks"]   = _q("SELECT COUNT(*) FROM rag.parent_chunks WHERE collection=%s", (coll,))
        counts["eval_questions"]  = _q("SELECT COUNT(*) FROM rag.eval_questions WHERE collection=%s", (coll,))
        counts["kg_chunks_done"]  = _q("SELECT COUNT(DISTINCT ce.chunk_id) FROM rag.chunk_entities ce JOIN rag.chunks c ON c.id=ce.chunk_id WHERE c.collection=%s", (coll,))
        counts["contextual_done"] = _q("SELECT COUNT(*) FROM rag.contextual_contexts cc JOIN rag.chunks c ON c.id=cc.chunk_id WHERE c.collection=%s", (coll,))
        counts["summaries"]       = _q("SELECT COUNT(*) FROM rag.summaries WHERE collection=%s", (coll,))
        conn.commit()  # close transaction so connection stays idle (not idle-in-transaction)
        return StatsResponse(collection=coll, **counts)

    return await asyncio.to_thread(_fetch, collection)


# ---------------------------------------------------------------------------
# /api/search
# ---------------------------------------------------------------------------

@app.post("/api/search", response_model=SearchResponse, tags=["Retrieval"])
async def search(req: SearchRequest, request: Request,
                 api_key: str = Security(_api_key_header)):
    """
    Ultra search — runs full pipeline: routing → retrieval → rerank →
    CRAG → Self-RAG → utility boost → parent expansion → provenance.
    Optionally augments context with MemoryWeb memories.
    """
    _check_collection_access(req.collection, request, api_key)

    # ── Prompt injection guard ────────────────────────────────────────────────
    from src.prompt_guard import check_query  # noqa: PLC0415
    query_ok, reason = check_query(req.query)
    if not query_ok:
        log.warning("/api/search rejected query — %s", reason)
        raise HTTPException(status_code=400, detail=f"Query rejected: {reason}")

    # Pull MemoryWeb context in parallel (non-blocking)
    memory_task = asyncio.create_task(_memoryweb_recall(req.query, top_k=3))

    def _run():
        from ultra_query import run_ultra_query  # noqa: PLC0415
        # Use RLS-scoped connection (rag_app role, no BYPASSRLS) so
        # PostgreSQL row-level security enforces collection isolation at
        # the DB layer — not just at the HTTP/application layer.
        conn = _get_scoped_conn(req.collection)
        result = run_ultra_query(
            conn,
            query=req.query,
            collection=req.collection,
            top_k=req.top_k,
            strategy_override=req.strategy,
            no_rerank=req.no_rerank,
            no_crag=req.no_crag,
        )
        return result

    try:
        result = await asyncio.to_thread(_run)
    except Exception as exc:
        log.error("/api/search failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

    # Attach MemoryWeb memories to first result's provenance (if available)
    memories = await memory_task
    if memories:
        result.setdefault("memory_context", memories)

    # Fire-and-forget: store this query in MemoryWeb
    top_content = result.get("results", [{}])[0].get("content", "")[:200] if result.get("results") else ""
    if top_content:
        _qlog_id: int | None = result.get("query_log_id") or None
        asyncio.create_task(_memoryweb_store(req.query, top_content, req.collection, query_log_id=_qlog_id))

    # Convert results to ResultItem list
    items = []
    for r in result.get("results", []):
        items.append(ResultItem(
            id=r.get("id"),
            content=r.get("content", ""),
            content_type=r.get("content_type", "text"),
            score=float(r.get("score", 0.0)),
            context_prefix=r.get("context_prefix"),
            chunk_metadata=r.get("chunk_metadata"),
            parent_content=r.get("parent_content") if req.include_parent else None,
            rerank_score=r.get("rerank_score"),
            utility_score=r.get("utility_score"),
            provenance_components=r.get("provenance_components") if req.include_provenance else None,
        ))

    return SearchResponse(
        results=items,
        strategy_used=result.get("strategy_used", "hybrid"),
        quality_score=float(result.get("quality_score", 0.0)),
        latency_ms=result.get("latency_ms", 0),
        provenance_chain_id=result.get("provenance_chain_id") if req.include_provenance else None,
        classification=result.get("classification") if req.include_provenance else None,
    )


# ---------------------------------------------------------------------------
# /api/ingest
# ---------------------------------------------------------------------------

@app.post("/api/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest(req: IngestRequest):
    """
    Trigger the full Ultra RAG ingestion pipeline for a collection.
    Long-running — consider running via CLI for large corpora.
    """
    def _run_ingest() -> IngestResponse:
        from ultra_ingest import (  # noqa: PLC0415
            ALL_STAGES, LLM_STAGES,
            _parse_stages, _run_parse, _run_embed,
            _run_contextual, _run_parents, _run_kg,
            _run_communities, _run_raptor,
        )
        conn = _get_conn()
        stages = _parse_stages(req.stages)
        all_stats: dict = {}
        completed: list = []
        t_start = time.time()

        dispatch = {
            "parse":       lambda: _run_parse(req.collection, conn),
            "embed":       lambda: _run_embed(req.collection, conn),
            "contextual":  lambda: _run_contextual(req.collection, conn, req.reprocess),
            "parents":     lambda: _run_parents(req.collection, conn),
            "kg":          lambda: _run_kg(req.collection, conn),
            "communities": lambda: _run_communities(req.collection, conn),
            "raptor":      lambda: _run_raptor(req.collection, conn),
        }

        for stage in stages:
            try:
                stats_out = dispatch[stage]()
                all_stats[stage] = stats_out
                completed.append(stage)
            except Exception as exc:
                log.error("Ingest stage '%s' failed: %s", stage, exc)
                all_stats[stage] = {"error": str(exc)}
                try:
                    conn.rollback()
                except Exception:
                    pass

        return IngestResponse(
            stages_completed=completed,
            stats=all_stats,
            total_time_seconds=round(time.time() - t_start, 2),
        )

    try:
        response = await asyncio.to_thread(_run_ingest)
    except Exception as exc:
        log.error("/api/ingest failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))
    return response


# ---------------------------------------------------------------------------
# /api/entities
# ---------------------------------------------------------------------------

@app.post("/api/entities", tags=["Knowledge Graph"])
async def entities(req: EntityQuery, request: Request,
                   api_key: str = Security(_api_key_header)):
    """
    Browse knowledge graph entities.  Filter by collection, search term,
    or entity type.  Returns list of entity dicts.
    """
    _check_collection_access(req.collection, request, api_key)
    def _fetch() -> list:
        import psycopg2.extras  # noqa: PLC0415
        conn = _get_scoped_conn(req.collection)
        conditions = ["collection = %s"]
        params: list = [req.collection]

        if req.entity_type:
            conditions.append("entity_type = %s")
            params.append(req.entity_type)
        if req.search_term:
            conditions.append("(name ILIKE %s OR description ILIKE %s)")
            like = f"%{req.search_term}%"
            params += [like, like]

        where = " AND ".join(conditions)
        params.append(req.limit)

        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            try:
                cur.execute(f"""
                    SELECT id, name, entity_type, description, specificity, aliases,
                           created_at
                    FROM   rag.entities
                    WHERE  {where}
                    ORDER  BY specificity DESC, name
                    LIMIT  %s
                """, params)
                return [dict(r) for r in cur.fetchall()]
            except Exception as exc:
                log.error("entities query failed: %s", exc)
                raise

    try:
        rows = await asyncio.to_thread(_fetch)
        return {"entities": rows, "count": len(rows)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# /api/communities
# ---------------------------------------------------------------------------

@app.post("/api/communities", tags=["Knowledge Graph"])
async def communities(req: CommunityQuery, request: Request,
                      api_key: str = Security(_api_key_header)):
    """
    Browse detected communities.  Optionally filter by resolution level.
    """
    _check_collection_access(req.collection, request, api_key)
    def _fetch() -> list:
        import psycopg2.extras  # noqa: PLC0415
        conn = _get_scoped_conn(req.collection)
        conditions = ["collection = %s"]
        params: list = [req.collection]

        if req.level is not None:
            conditions.append("level = %s")
            params.append(req.level)

        where = " AND ".join(conditions)
        params.append(req.limit)

        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            try:
                cur.execute(f"""
                    SELECT id, level, title, summary, entity_count, created_at
                    FROM   rag.communities
                    WHERE  {where}
                    ORDER  BY level, entity_count DESC
                    LIMIT  %s
                """, params)
                return [dict(r) for r in cur.fetchall()]
            except Exception as exc:
                log.error("communities query failed: %s", exc)
                raise

    try:
        rows = await asyncio.to_thread(_fetch)
        return {"communities": rows, "count": len(rows)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# /api/provenance/{id}
# ---------------------------------------------------------------------------

@app.get("/api/provenance/{chain_id}", response_model=ProvenanceResponse, tags=["Provenance"])
async def provenance(chain_id: int):
    """
    Return the full provenance chain with all score components for a given
    chain id.  Use the chain_id returned by /api/search (include_provenance=true).
    """
    def _fetch(cid: int) -> ProvenanceResponse:
        import psycopg2.extras  # noqa: PLC0415
        conn = _get_conn()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT id, query_log_id, overall_confidence
                FROM   rag.provenance_chains
                WHERE  id = %s
            """, (cid,))
            chain_row = cur.fetchone()
            if not chain_row:
                raise HTTPException(status_code=404, detail=f"Provenance chain {cid} not found")

            cur.execute("""
                SELECT ps.id, ps.chunk_id, ps.entity_id,
                       ps.score_components, ps.rank_position,
                       c.content, c.content_type, c.context_prefix
                FROM   rag.provenance_steps ps
                LEFT JOIN rag.chunks c ON c.id = ps.chunk_id
                WHERE  ps.chain_id = %s
                ORDER  BY ps.rank_position
            """, (cid,))
            steps = [dict(r) for r in cur.fetchall()]

        return ProvenanceResponse(
            chain_id=cid,
            query_log_id=chain_row["query_log_id"],
            overall_confidence=float(chain_row["overall_confidence"] or 0.0),
            steps=steps,
        )

    try:
        return await asyncio.to_thread(_fetch, chain_id)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# /api/eval
# ---------------------------------------------------------------------------

@app.post("/api/eval", tags=["Evaluation"])
async def eval_endpoint(req: EvalRequest):
    """
    Generate a synthetic evaluation dataset and/or run RAGAS evaluation.
    Returns metric scores and the list of generated questions.
    """
    def _run_eval() -> dict:
        from src.db_ultra        import create_ultra_schema   # noqa: PLC0415
        from src.eval_generator  import EvalDatasetGenerator  # noqa: PLC0415
        from src.eval_runner     import RAGASEvalRunner        # noqa: PLC0415
        from src.search          import search as base_search  # noqa: PLC0415

        conn = _get_conn()
        create_ultra_schema(conn)

        # Generate questions
        generator = EvalDatasetGenerator(conn, req.collection)
        questions = generator.generate_dataset(n_questions=req.n_questions)
        log.info("Generated %d eval questions", len(questions))

        eval_scores: dict = {}
        if req.run_eval and questions:
            def search_fn(q: str, top_k: int = 5):
                return base_search(conn, q, req.collection, top_k=top_k)

            runner = RAGASEvalRunner(conn, req.collection)
            eval_scores = runner.evaluate_dataset(
                questions, search_fn, run_name=req.run_name
            )

        return {
            "questions_generated": len(questions),
            "questions": [
                {k: v for k, v in q.items() if k not in ("source_chunk_ids",)}
                for q in questions[:50]   # cap response size
            ],
            "eval_scores": eval_scores,
        }

    try:
        result = await asyncio.to_thread(_run_eval)
    except Exception as exc:
        log.error("/api/eval failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import argparse  # noqa: PLC0415
    parser = argparse.ArgumentParser(description="Ultra RAG API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8300, help="Port (default: 8300)")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    args = parser.parse_args()

    log.info("Starting Ultra RAG Server on %s:%d", args.host, args.port)
    uvicorn.run(
        "ultra_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        log_level="info",
    )



# Import statements needed (only new ones not in existing code)
from fastapi import FastAPI, File, UploadFile, HTTPException

# New endpoint for document upload
@FastAPI.post("/api/upload")
async def upload_document(file: UploadFile = File(...), collection: str):
    if not _is_localhost(request):
        raise HTTPException(status_code=403, detail="Uploads from external sources are not allowed.")
    
    if not _API_KEY:
        raise HTTPException(status_code=403, detail="API key is required for uploads.")
    
    # Ensure the collection is valid
    if collection.lower() not in _COLLECTION_KEYS:
        raise HTTPException(status_code=400, detail="Invalid collection name.")
    
    # Create a temporary file path
    temp_file_path = tempfile.mktemp(dir="/tmp/uploads/", suffix=".txt")
    
    # Write the file to the temporary path
    with open(temp_file_path, "wb") as buffer:
        content = await file.file.read()
        buffer.write(content)
    
    # Get file size in bytes
    size_bytes = len(content)
    
    # Return the upload response
    return {"ok": True, "filename": file.filename, "collection": collection, "size_bytes": size_bytes}

if __name__ == "__main__":
    main()
