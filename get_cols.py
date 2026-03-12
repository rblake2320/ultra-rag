import psycopg2, os
from dotenv import load_dotenv
load_dotenv()
conn = psycopg2.connect(host="localhost", port=5432, dbname="postgres",
    user="postgres", password=os.getenv("RAG_DB_PASSWORD","?Booker78!"))
cur = conn.cursor()
cur.execute("SELECT column_name FROM information_schema.columns WHERE table_schema='rag' AND table_name='chunks' ORDER BY ordinal_position")
print([r[0] for r in cur.fetchall()])
cur.execute("SELECT column_name FROM information_schema.columns WHERE table_schema='rag' AND table_name='documents' ORDER BY ordinal_position")
print([r[0] for r in cur.fetchall()])
conn.close()
