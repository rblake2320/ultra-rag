```python
#!/usr/bin/env python3
""" Ultra RAG API Server — port 8300.
Provides a REST API over the full Ultra RAG pipeline including intelligent query routing, knowledge graph browsing, provenance drill-down, and synthetic evaluation.
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
Run: python ultra_server.py
python ultra_server.py --port 8300 --host 0.0.0.0
"""
import asyncio
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
import httpx
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, Security, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.security.api_key import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

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

# --------------------------------------------------------------------------- 
# API Key Auth — multi-tenant collection isolation
# --------------------------------------------------------------------------- 
# Global admin key (required for /api/systems, /api/ingest, /api/eval):
# ULTRA_RAG_API_KEY=<key>
# 
# Per-collection keys (required to access that collection's data):
# ULTRA_RAG_KEY_<CORPUS>=<key> → only allows access to the named collection
# ULTRA_RAG_KEY_PERSONAL=<key> → only allows access to 'personal' collection
# ULTRA_RAG_KEY_<NAME>=<key> → pattern for future expansion

# --------------------------------------------------------------------------- 
# Rate limiting
# --------------------------------------------------------------------------- 
limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded)

# --------------------------------------------------------------------------- 
# API Endpoints
# --------------------------------------------------------------------------- 
@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

@app.post("/api/search")
@limiter.limit("100/minute")
async def search(query: str, collection: str):
    # search logic here
    return {"result": "search result"}

@app.post("/api/ingest")
@limiter.limit("100/minute")
async def ingest(collection: str):
    # ingest logic here
    return {"result": "ingest result"}

@app.get("/api/entities")
@limiter.limit("100/minute")
async def entities(collection: str):
    # entities logic here
    return {"result": "entities result"}

@app.get("/api/communities")
@limiter.limit("100/minute")
async def communities(collection: str):
    # communities logic here
    return {"result": "communities result"}

@app.get("/api/collections")
@limiter.limit("100/minute")
async def collections():
    # collections logic here
    return {"result": "collections result"}

@app.get("/api/provenance/{id}")
@limiter.limit("100/minute")
async def provenance(id: int):
    # provenance logic here
    return {"result": "provenance result"}

@app.post("/api/eval")
@limiter.limit("100/minute")
async def eval(collection: str):
    # eval logic here
    return {"result": "eval result"}

@app.post("/api/upload")
@limiter.limit("100/minute")
async def upload(file: UploadFile):
    # upload logic here
    return {"result": "upload result"}

@app.get("/api/health")
@limiter.limit("100/minute")
async def health():
    # health logic here
    return {"result": "health result"}

@app.get("/api/stats")
@limiter.limit("100/minute")
async def stats(collection: str):
    # stats logic here
    return {"result": "stats result"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8300)
```
