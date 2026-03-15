```python
#!/usr/bin/env python3
""" Ultra RAG API Server — port 8300. Provides a REST API over the full Ultra RAG pipeline including intelligent query routing, knowledge graph browsing, provenance drill-down, and synthetic evaluation.
Endpoints:
GET / — Dashboard UI (static/index.html)
POST /api/search — Ultra search with all strategies
POST /api/ingest — Trigger full 7-stage pipeline
GET /api/entities — Browse knowledge graph entities
GET /api/communities — Browse detected communities
GET /api/collections — List all collection names
GET /api/provenance/{id} — Drill-down confidence decomposition
POST /api/eval — Run evaluation suite (generate + RAGAS)
POST /api/upload — Upload and ingest a document file
GET /api/health — System health check
GET /api/stats — Collection statistics
GET /api/collections/{name}/stats — Collection statistics for a specific collection
Run: python ultra_server.py
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
    print(f" {exc}")
    sys.exit(1)

# --------------------------------------------------------------------------- 
# API Key Auth — multi-tenant collection isolation
# --------------------------------------------------------------------------- 
# Global admin key (required for /api/systems, /api/ingest, /api/eval):
# ULTRA_RAG_API_KEY=<key>
# 
# Per-collection keys (required to access that collection's data):
# ULTRA_RAG_KEY_<CORPUS>=<key> → only allows access to the named collection
# ULTRA_RAG_KEY_PERSONAL=<key> → only allows access to 'personal' collection
# ULTRA_RAG_KEY_<NAME>=<key> → pattern for f

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the collection stats endpoint
@app.get("/api/collections/{name}/stats")
async def get_collection_stats(name: str):
    # Connect to the database
    from src.db import get_conn
    conn = get_conn()

    # Get the collection stats
    with conn.cursor() as cur:
        cur.execute("""
            SELECT 
                COUNT(*) as chunk_count,
                COUNT(DISTINCT entity_id) as entity_count,
                AVG(embedding_coverage) as embedding_coverage,
                MAX(last_ingested_timestamp) as last_ingested_timestamp,
                COUNT(DISTINCT document_id) as document_count
            FROM 
                rag.collection_stats
            WHERE 
                collection_name = %s
        """, (name,))
        stats = cur.fetchone()

    # Return the collection stats as JSON
    return JSONResponse(content=json.dumps({
        "chunk_count": stats[0],
        "entity_count": stats[1],
        "embedding_coverage": stats[2],
        "last_ingested_timestamp": stats[3],
        "document_count": stats[4]
    }), media_type="application/json")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8300)
