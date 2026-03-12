#!/usr/bin/env python3
"""
Ultra RAG API Server — port 8300.

Provides a REST API over the full Ultra RAG pipeline including intelligent
query routing, knowledge graph browsing, provenance drill-down, and
synthetic evaluation.

Endpoints:
  POST /api/search       — Ultra search with all strategies
  POST /api/ingest       — Trigger full 7-stage pipeline
  GET  /api/entities     — Browse knowledge graph entities
  GET  /api/communities  — Browse detected communities
  GET  /api/provenance/{id} — Drill-down confidence decomposition
  POST /api/eval         — Run evaluation suite (generate + RAGAS)
  GET  /api/health       — System health check
  GET  /api/stats        — Collection statistics

Run:
  python ultra_server.py
  python ultra_server.py --port 8300 --host 0.0.0.0
"""
import asyncio
import json
import logging
import sys
import time
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
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
except ImportError as exc:
    print(f"ERROR: FastAPI/uvicorn not installed. Run: pip install fastapi uvicorn")
    print(f"  {exc}")
    sys.exit(1)

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
    collection:         str             = "default"
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
    collection:   str           = "imds"
    search_term:  Optional[str] = None
    entity_type:  Optional[str] = None
    limit:        int           = Field(20, ge=1, le=200)


class CommunityQuery(BaseModel):
    collection: str           = "imds"
    level:      Optional[int] = None
    limit:      int           = Field(20, ge=1, le=200)


class EvalRequest(BaseModel):
    collection:   str           = "imds"
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


class StatsResponse(BaseModel):
    collection:       str
    chunks:           int
    documents:        int
    embedded_chunks:  int
    entities:         int
    relationships:    int
    communities:      int
    parent_chunks:    int
    eval_questions:   int


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
# Startup / shutdown
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    log.info("Ultra RAG Server starting up…")
    conn = _get_conn()
    await asyncio.to_thread(_ensure_schemas, conn)
    log.info("Schemas ready.")


def _ensure_schemas(conn) -> None:
    from src.db       import create_schema        # noqa: PLC0415
    from src.db_ultra import create_ultra_schema  # noqa: PLC0415
    create_schema(conn)
    create_ultra_schema(conn)


# ---------------------------------------------------------------------------
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
    except Exception as exc:
        db_status = f"error: {exc}"

    # pgvector check
    try:
        conn = _get_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM rag.chunks WHERE embedding IS NOT NULL LIMIT 1")
    except Exception as exc:
        pgvector_status = f"error: {exc}"

    # Ollama check
    try:
        import httpx  # noqa: PLC0415
        from src.config import get_config  # noqa: PLC0415
        cfg = get_config()
        ollama_url = cfg.get("llm", {}).get("ollama_url", "http://localhost:11434")
        resp = httpx.get(f"{ollama_url}/api/tags", timeout=5.0)
        resp.raise_for_status()
    except Exception as exc:
        ollama_status = f"error: {exc}"

    overall = "ok" if all(s == "ok" for s in [db_status, pgvector_status, ollama_status]) else "degraded"

    return HealthResponse(
        status=overall,
        db=db_status,
        ollama=ollama_status,
        pgvector=pgvector_status,
        timestamp=datetime.utcnow().isoformat(),
    )


# ---------------------------------------------------------------------------
# /api/stats
# ---------------------------------------------------------------------------

@app.get("/api/stats", response_model=StatsResponse, tags=["System"])
async def stats(collection: str = Query("default", description="Collection name")):
    """Return chunk, entity, community, and parent-chunk counts."""
    def _fetch(coll: str) -> StatsResponse:
        conn = _get_conn()
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
        return StatsResponse(collection=coll, **counts)

    return await asyncio.to_thread(_fetch, collection)


# ---------------------------------------------------------------------------
# /api/search
# ---------------------------------------------------------------------------

@app.post("/api/search", response_model=SearchResponse, tags=["Retrieval"])
async def search(req: SearchRequest):
    """
    Ultra search — runs full pipeline: routing → retrieval → rerank →
    CRAG → Self-RAG → utility boost → parent expansion → provenance.
    """
    def _run():
        from ultra_query import run_ultra_query  # noqa: PLC0415
        conn = _get_conn()
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
async def entities(req: EntityQuery):
    """
    Browse knowledge graph entities.  Filter by collection, search term,
    or entity type.  Returns list of entity dicts.
    """
    def _fetch() -> list:
        import psycopg2.extras  # noqa: PLC0415
        conn = _get_conn()
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
async def communities(req: CommunityQuery):
    """
    Browse detected communities.  Optionally filter by resolution level.
    """
    def _fetch() -> list:
        import psycopg2.extras  # noqa: PLC0415
        conn = _get_conn()
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


if __name__ == "__main__":
    main()
