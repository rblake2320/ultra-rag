"""
Export a collection from pgvector to ChromaDB for portable use.
"""
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def export_to_chroma(conn, collection: str, output_dir: Path,
                     chroma_collection_name: str | None = None) -> int:
    """
    Export all chunks (with embeddings) from rag.chunks → ChromaDB on disk.
    Returns number of chunks exported.
    """
    import chromadb

    chroma_name = chroma_collection_name or collection
    output_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(output_dir))
    coll   = client.get_or_create_collection(
        name=chroma_name,
        metadata={"hnsw:space": "cosine"},
    )

    with conn.cursor() as cur:
        cur.execute("""
            SELECT c.stable_id, c.content, c.content_type, c.context_prefix,
                   c.chunk_metadata, c.token_count, c.embedding,
                   d.file_name, d.file_path
            FROM rag.chunks c
            JOIN rag.documents d ON d.id = c.document_id
            WHERE c.collection = %s AND c.embedding IS NOT NULL
            ORDER BY c.id
        """, (collection,))
        rows = cur.fetchall()

    if not rows:
        log.warning(f"No embedded chunks found for collection '{collection}'")
        return 0

    batch_size = 500
    total = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i: i + batch_size]

        ids        = [r[0] or f"{collection}_{i+j}" for j, r in enumerate(batch)]
        documents  = [r[1] for r in batch]
        embeddings = [
            [float(x) for x in r[6].strip("[]").split(",")]
            if isinstance(r[6], str) else list(r[6])
            for r in batch
        ]
        metadatas  = [{
            "content_type":   r[2],
            "context_prefix": r[3] or "",
            "token_count":    r[5] or 0,
            "file_name":      r[7] or "",
            "file_path":      r[8] or "",
            "chunk_meta":     json.dumps(r[4] or {}),
        } for r in batch]

        coll.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        total += len(batch)
        log.info(f"  Exported {total}/{len(rows)} chunks…")

    log.info(f"ChromaDB export complete: {total} chunks → {output_dir}")
    return total
