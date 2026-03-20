#!/usr/bin/env python3
"""
Apply Row Level Security (RLS) to all rag schema tables that have a
collection column. Forces RLS even for the postgres superuser via
FORCE ROW LEVEL SECURITY. Isolation is controlled by the session variable
app.rag_collection:
  - ''  or not set → all rows visible (admin/ingest mode)
  - 'imds'          → only imds rows visible
  - 'personal'      → only personal rows visible
"""
import psycopg2

conn = psycopg2.connect('postgresql://postgres:%3FBooker78%21@localhost:5432/postgres')
conn.autocommit = True
cur = conn.cursor()

# Tables with collection column that need direct RLS
# (derived from: SELECT table_name FROM information_schema.columns
#  WHERE table_schema='rag' AND column_name='collection')
COLLECTION_TABLES = [
    'chunks',
    'chunks_backup_v1',
    'chunks_backup_v2',
    'communities',
    'documents',
    'documents_backup_v1',
    'entities',
    'eval_questions',
    'eval_runs',
    'parent_chunks',
    'query_log',
    'relationships',
    'summaries',
]

for table in COLLECTION_TABLES:
    print(f"Enabling RLS on rag.{table}...")
    # Enable RLS
    cur.execute(f'ALTER TABLE rag.{table} ENABLE ROW LEVEL SECURITY')
    # Force RLS even for superuser (critical — postgres role bypasses RLS otherwise)
    cur.execute(f'ALTER TABLE rag.{table} FORCE ROW LEVEL SECURITY')
    # Drop existing policies if any
    cur.execute(f"DROP POLICY IF EXISTS collection_isolation ON rag.{table}")
    # Create policy: allow row if collection matches session var,
    # OR session var is empty/unset (admin mode)
    cur.execute(f"""
        CREATE POLICY collection_isolation ON rag.{table}
        USING (
            current_setting('app.rag_collection', true) IS NULL
            OR current_setting('app.rag_collection', true) = ''
            OR collection = current_setting('app.rag_collection', true)
        )
    """)
    print(f"  Done: rag.{table}")

conn.close()
print("\nRLS applied to all collection tables.")
