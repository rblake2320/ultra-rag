# Task #120: Add DELETE /api/collections/{name} endpoint to ultra-rag

# Import statements needed (only new ones not in existing code)
from fastapi.responses import JSONResponse

# New function/endpoint
@app.delete('/api/collections/{name}')
async def delete_collection(name: str):
    """
    DELETE /api/collections/{name} endpoint to delete all documents in the named collection from the vector store.
    Returns {"ok": true, "deleted_count": int, "collection": name}.
    """
    collection_key = _COLLECTION_KEYS.get(name.lower())
    if not collection_key:
        raise HTTPException(status_code=404, detail=f"Collection {name} not found.")
    
    # Assuming 'vector_store' is a function or method that interacts with the actual vector store
    deleted_count = vector_store.delete_documents(collection_key)
    
    return JSONResponse({
        "ok": True,
        "deleted_count": deleted_count,
        "collection": name
    })

# Ensure existing code structure and imports are maintained
# ... (rest of the existing app setup code)

# Remember to add necessary imports and error handling as per your actual vector store implementation
