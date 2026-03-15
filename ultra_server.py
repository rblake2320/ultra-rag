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
    GET /api/status — Pipeline stage completion percentages per collection
Run: python ultra_server.py
       python ultra_server.py --port 8300 --host 0.0.0.0
"""
import asyncio
import json
import logging
import os
import psycopg2
import sys
import time
from datetime import datetime
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, Security, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.security.api_key import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

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
# ULTRA_RAG_KEY_<NAME>=<key> → pattern for future collections

app = FastAPI()

# Define the status endpoint
@app.get("/api/status")
async def get_status():
    # Connect to the database
    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password="%3FBooker78%21",
        host="localhost",
        port="5432"
    )

    # Create a cursor object
    cur = conn.cursor()

    # Initialize the response dictionary
    response = {"collections": {}}

    # Query the database for collection names
    cur.execute("SELECT DISTINCT collection FROM rag.chunks")
    collections = [row[0] for row in cur.fetchall()]

    # Iterate over each collection
    for collection in collections:
        # Initialize the collection dictionary
        response["collections"][collection] = {
            "chunks": 0,
            "embedded_pct": 0,
            "contextual_pct": 0,
            "raptor_pct": 0,
            "kg_pct": 0
        }

        # Query the database for chunk counts
        cur.execute("SELECT COUNT(*) FROM rag.chunks WHERE collection = %s", (collection,))
        chunk_count = cur.fetchone()[0]
        response["collections"][collection]["chunks"] = chunk_count

        # Query the database for embedded chunk counts
        cur.execute("SELECT COUNT(*) FROM rag.chunks WHERE collection = %s AND embedding IS NOT NULL", (collection,))
        embedded_count = cur.fetchone()[0]
        response["collections"][collection]["embedded_pct"] = (embedded_count / chunk_count) * 100 if chunk_count > 0 else 0

        # Query the database for contextual chunk counts
        cur.execute("SELECT COUNT(*) FROM rag.chunks WHERE collection = %s AND context IS NOT NULL", (collection,))
        contextual_count = cur.fetchone()[0]
        response["collections"][collection]["contextual_pct"] = (contextual_count / chunk_count) * 100 if chunk_count > 0 else 0

        # Query the database for raptor chunk counts
        cur.execute("SELECT COUNT(*) FROM rag.chunks WHERE collection = %s AND raptor IS NOT NULL", (collection,))
        raptor_count = cur.fetchone()[0]
        response["collections"][collection]["raptor_pct"] = (raptor_count / chunk_count) * 100 if chunk_count > 0 else 0

        # Query the database for kg chunk counts
        cur.execute("SELECT COUNT(*) FROM rag.chunks WHERE collection = %s AND kg IS NOT NULL", (collection,))
        kg_count = cur.fetchone()[0]
        response["collections"][collection]["kg_pct"] = (kg_count / chunk_count) * 100 if chunk_count > 0 else 0

    # Close the cursor and connection
    cur.close()
    conn.close()

    # Return the response
    return JSONResponse(content=response, media_type="application/json")

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8300)
