from psycopg2 import connect
from psycopg2.extras import DictCursor

def create_ultra_schema(conn):
    # Create the Ultra RAG schema
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS rag.collection_stats (
                collection_name VARCHAR(255) PRIMARY KEY,
                chunk_count INTEGER,
                entity_count INTEGER,
                embedding_coverage FLOAT,
                last_ingested_timestamp TIMESTAMP,
                document_count INTEGER
            );
        """)
        conn.commit()
