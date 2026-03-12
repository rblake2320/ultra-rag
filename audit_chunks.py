"""Chunk quality audit for RAG corpus baseline."""
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(
    host="localhost", port=5432, dbname="postgres",
    user="postgres", password=os.getenv("RAG_DB_PASSWORD", "?Booker78!")
)
cur = conn.cursor()

# --- 1. Basic stats ---
cur.execute("SELECT count(*) FROM rag.chunks")
total = cur.fetchone()[0]

cur.execute("""
SELECT min(token_count), max(token_count), 
       avg(token_count)::int, 
       percentile_cont(0.5) WITHIN GROUP (ORDER BY token_count)::int
FROM rag.chunks
""")
mn, mx, avg_t, med = cur.fetchone()

# --- 2. Size buckets ---
cur.execute("""
SELECT 
  CASE 
    WHEN token_count < 50 THEN 'A: < 50 (noise)'
    WHEN token_count < 100 THEN 'B: 50-99 (small)'
    WHEN token_count < 200 THEN 'C: 100-199 (med-small)'
    WHEN token_count < 512 THEN 'D: 200-511 (medium)'
    WHEN token_count < 800 THEN 'E: 512-799 (target)'
    ELSE 'F: 800+ (large)'
  END as bucket,
  count(*) as cnt,
  round(100.0 * count(*) / %s, 1) as pct
FROM rag.chunks
GROUP BY 1
ORDER BY 1
""", (total,))
buckets = cur.fetchall()

# --- 3. content_type distribution ---
cur.execute("""
SELECT content_type, count(*), round(100.0 * count(*) / %s, 1)
FROM rag.chunks 
GROUP BY 1 ORDER BY 2 DESC
""", (total,))
type_dist = cur.fetchall()

# --- 4. Sample tiny chunks (noise candidates) ---
cur.execute("""
SELECT id, token_count, left(content, 120) 
FROM rag.chunks 
WHERE token_count < 50 
ORDER BY token_count 
LIMIT 15
""")
tiny = cur.fetchall()

# --- 5. Sample huge chunks ---
cur.execute("""
SELECT id, token_count, left(content, 120) 
FROM rag.chunks 
WHERE token_count > 1000 
ORDER BY token_count DESC 
LIMIT 10
""")
huge = cur.fetchall()

# --- 6. Broken boundary check ---
cur.execute("""
SELECT count(*) FROM rag.chunks 
WHERE right(trim(content), 1) NOT IN ('.','!','?',':',';',')',']','}','"')
  AND token_count > 50
""")
broken_end = cur.fetchone()[0]

# --- 7. Duplicate content check ---
cur.execute("""
SELECT content_hash, count(*) as dupes 
FROM rag.chunks 
GROUP BY content_hash 
HAVING count(*) > 1
ORDER BY dupes DESC
LIMIT 10
""")
dupes = cur.fetchall()

cur.execute("""
SELECT count(*) FROM (
    SELECT content_hash FROM rag.chunks 
    GROUP BY content_hash HAVING count(*) > 1
) d
""")
dupe_groups = cur.fetchone()[0]

# --- 8. chunk_metadata type distribution ---
meta_types = []
cur.execute("""
SELECT chunk_metadata->>'chunk_type' as ctype, count(*), 
       round(100.0 * count(*) / %s, 1)
FROM rag.chunks 
WHERE chunk_metadata->>'chunk_type' IS NOT NULL
GROUP BY 1 ORDER BY 2 DESC
""", (total,))
meta_types = cur.fetchall()

# --- 9. Per-document stats ---
cur.execute("""
SELECT d.file_name, count(c.id) as chunks,
       min(c.token_count), max(c.token_count), avg(c.token_count)::int
FROM rag.chunks c
JOIN rag.documents d ON c.document_id = d.id
GROUP BY d.file_name
ORDER BY chunks DESC
LIMIT 10
""")
per_doc = cur.fetchall()

# --- 10. heading_path usage ---
cur.execute("SELECT count(*) FROM rag.chunks WHERE heading_path IS NOT NULL AND array_length(heading_path, 1) > 0")
has_heading = cur.fetchone()[0]

# --- 11. stable_id coverage ---
cur.execute("SELECT count(*) FROM rag.chunks WHERE stable_id IS NOT NULL AND stable_id != ''")
has_stable = cur.fetchone()[0]

# --- 12. embedding coverage ---
cur.execute("SELECT count(*) FROM rag.chunks WHERE embedding IS NOT NULL")
has_embed = cur.fetchone()[0]

conn.close()

# === REPORT ===
print("=" * 65)
print("  CHUNK CORPUS QUALITY AUDIT — BASELINE")
print("=" * 65)
print(f"\n  Total chunks:    {total:,}")
print(f"  Token range:     {mn} - {mx}")
print(f"  Mean tokens:     {avg_t}")
print(f"  Median tokens:   {med}")
print(f"  Embeddings:      {has_embed:,} / {total:,} ({round(100*has_embed/total,1)}%)")
print(f"  Stable IDs:      {has_stable:,} / {total:,} ({round(100*has_stable/total,1)}%)")
print(f"  Heading paths:   {has_heading:,} / {total:,} ({round(100*has_heading/total,1)}%)")

print(f"\n{'— SIZE DISTRIBUTION —':^65}")
for b, c, p in buckets:
    bar = '#' * int(float(p) / 2)
    print(f"  {b:28s} {c:>6,}  ({float(p):>5.1f}%) {bar}")

noise_count = sum(c for b, c, p in buckets if 'noise' in b)
noise_pct = round(100.0 * noise_count / total, 1) if total else 0

print(f"\n{'— CONTENT TYPE DISTRIBUTION —':^65}")
for t, c, p in type_dist:
    print(f"  {str(t):25s} {c:>6,}  ({float(p):>5.1f}%)")

if meta_types:
    print(f"\n{'— METADATA chunk_type DISTRIBUTION —':^65}")
    for t, c, p in meta_types:
        print(f"  {str(t):25s} {c:>6,}  ({float(p):>5.1f}%)")

print(f"\n{'— BROKEN BOUNDARIES —':^65}")
broken_pct = round(100.0 * broken_end / total, 1)
print(f"  Chunks >50tok not ending with sentence punct: {broken_end:,} ({broken_pct}%)")

print(f"\n{'— DUPLICATES —':^65}")
print(f"  Duplicate content_hash groups: {dupe_groups}")
if dupes:
    for h, c in dupes[:5]:
        print(f"    hash ...{str(h)[-8:]}  appears {c}x")

print(f"\n{'— TOP 10 DOCS BY CHUNK COUNT —':^65}")
for fn, cnt, lo, hi, av in per_doc:
    print(f"  {fn[:42]:42s} {cnt:>5} chunks  (tok: {lo}-{hi}, avg {av})")

print(f"\n{'— SAMPLE NOISE CHUNKS (<50 tokens) —':^65}")
for cid, tc, txt in tiny:
    clean = txt.replace('\n', ' ').strip()
    print(f"  [{tc:>3} tok] {clean[:85]}")

print(f"\n{'— SAMPLE OVERSIZED CHUNKS (>1000 tokens) —':^65}")
for cid, tc, txt in huge:
    clean = txt.replace('\n', ' ').strip()
    print(f"  [{tc:>4} tok] {clean[:85]}")

print(f"\n{'=' * 65}")
print(f"  QUALITY SCORECARD")
print(f"{'=' * 65}")
target_count = sum(c for b, c, p in buckets if 'target' in b or 'medium' in b.lower())
target_pct = round(100.0 * target_count / total, 1)
oversized = sum(c for b, c, p in buckets if '800' in b)
print(f"  Right-sized (200-799 tok):  {target_count:,} ({target_pct}%)")
print(f"  Noise (<50 tok):            {noise_count:,} ({noise_pct}%)")
print(f"  Oversized (800+ tok):       {oversized:,} ({round(100*oversized/total,1)}%)")
print(f"  Broken boundaries:          {broken_end:,} ({broken_pct}%)")
print(f"  Duplicate groups:           {dupe_groups}")
print(f"  Embed coverage:             {round(100*has_embed/total,1)}%")
print(f"  Stable ID coverage:         {round(100*has_stable/total,1)}%")
