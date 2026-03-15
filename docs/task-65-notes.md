# Task #65: [GH#2] Add bulk delete endpoint: DELETE /api/documents with list of doc_ids

---FILE: ultra_server.py---
```python
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, Security, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

# ... existing code ...

app = FastAPI()

# ... existing code ...

class DeleteDocumentsRequest(BaseModel):
    collection: str
    doc_ids: List[str]

@app.delete("/api/documents")
async def delete_documents(request: DeleteDocumentsRequest):
    """
    Bulk delete endpoint: DELETE /api/documents with list of doc_ids
    """
    collection = request.collection
    doc_ids = request.doc_ids

    # Validate input
    if not collection or not doc_ids:
        raise HTTPException(status_code=400, detail="Invalid request")

    # Connect to database
    from src.db import get_conn
    conn = get_conn()

    try:
        # Delete chunks
        with conn.cursor() as cur:
            cur.execute("""
                DELETE FROM rag.chunks
                WHERE collection = %s AND doc_id IN %s
            """, (collection, tuple(doc_ids)))
            conn.commit()

        # Delete entities
        with conn.cursor() as cur:
            cur.execute("""
                DELETE FROM rag.entities
                WHERE collection = %s AND doc_id IN %s
            """, (collection, tuple(doc_ids)))
            conn.commit()

        # Delete embeddings
        with conn.cursor() as cur:
            cur.execute("""
                DELETE FROM rag.embeddings
                WHERE collection = %s AND doc_id IN %s
            """, (collection, tuple(doc_ids)))
            conn.commit()

        return JSONResponse(content={"message": "Documents deleted successfully"}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

# ... existing code ...
```

---FILE: src/db.py---
```python
import psycopg2
from psycopg2 import extras

def get_conn():
    # Connect to database
    conn = psycopg2.connect(
        host="localhost",
        database="ultra_rag",
        user="ultra_rag",
        password="ultra_rag"
    )
    return conn
```

---FILE: ultra_query.py---
```python
# No changes needed
```

---FILE: ultra_ingest.py---
```python
# No changes needed
```

---FILE: ultra_eval.py---
```python
# No changes needed
```

---FILE: ultra_client.py---
```python
# No changes needed
```

---FILE: tests/test_ultra_rag.py---
```python
# Add test case for bulk delete endpoint
def test_bulk_delete():
    # Create test data
    collection = "test_collection"
    doc_ids = ["doc1", "doc2", "doc3"]

    # Call bulk delete endpoint
    response = client.delete(f"/api/documents", json={"collection": collection, "doc_ids": doc_ids})

    # Check response
    assert response.status_code == 200
    assert response.json()["message"] == "Documents deleted successfully"

    # Check database
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM rag.chunks WHERE collection = %s AND doc_id IN %s", (collection, tuple(doc_ids)))
        assert cur.fetchall() == []
    conn.close()
```
Note: This implementation assumes a PostgreSQL database with the `rag` schema. You may need to modify the database connection and queries to match your specific database setup. Additionally, this implementation does not include any error handling or logging, which you should add in a production environment.