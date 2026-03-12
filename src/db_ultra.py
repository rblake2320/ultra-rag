"""
Ultra RAG schema extensions: KG, hierarchy, quality, provenance tables.

Creates all additional tables needed for Phase 1 Ultra RAG on top of the
existing rag.documents + rag.chunks base schema (see db.py).
"""
import logging

from .config import get_config
from .db import get_conn

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema creation
# ---------------------------------------------------------------------------

def create_ultra_schema(conn) -> None:
    """
    Create all Ultra RAG extension tables in the rag schema (IF NOT EXISTS).
    Safe to call on every startup — idempotent.
    Requires the base schema (rag.documents, rag.chunks) to already exist.
    """
    import logging
    log = logging.getLogger(__name__)

    # Clear any zombie idle-in-transaction sessions that would block index creation.
    # This happens when background pipeline processes are killed mid-transaction.
    try:
        old_autocommit = conn.autocommit
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(pg_terminate_backend(pid))
                FROM pg_stat_activity
                WHERE state = 'idle in transaction'
                  AND pid != pg_backend_pid()
                  AND query_start < NOW() - INTERVAL '30 minutes'
            """)
            n = cur.fetchone()[0]
            if n:
                log.info("Cleared %d idle-in-transaction zombie session(s).", n)
        conn.autocommit = old_autocommit
    except Exception:
        pass  # non-critical

    with conn.cursor() as cur:

        # ── Make sure pgvector is loaded ────────────────────────────────
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # ════════════════════════════════════════════════════════════════
        # Knowledge Graph
        # ════════════════════════════════════════════════════════════════

        cur.execute("""
            CREATE TABLE IF NOT EXISTS rag.entities (
                id              BIGSERIAL PRIMARY KEY,
                collection      VARCHAR(100) NOT NULL,
                name            TEXT NOT NULL,
                entity_type     VARCHAR(50),
                description     TEXT,
                embedding       vector(768),
                specificity     FLOAT DEFAULT 0.0,
                aliases         TEXT[] DEFAULT '{}',
                created_at      TIMESTAMPTZ DEFAULT now(),
                UNIQUE(collection, name)
            );
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS rag.relationships (
                id                  BIGSERIAL PRIMARY KEY,
                collection          VARCHAR(100) NOT NULL,
                source_id           BIGINT REFERENCES rag.entities(id) ON DELETE CASCADE,
                target_id           BIGINT REFERENCES rag.entities(id) ON DELETE CASCADE,
                rel_type            VARCHAR(100),
                weight              FLOAT DEFAULT 1.0,
                context             TEXT,
                context_embedding   vector(768),
                created_at          TIMESTAMPTZ DEFAULT now()
            );
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS rag.synonymy_edges (
                id          BIGSERIAL PRIMARY KEY,
                entity_a_id BIGINT REFERENCES rag.entities(id) ON DELETE CASCADE,
                entity_b_id BIGINT REFERENCES rag.entities(id) ON DELETE CASCADE,
                cosine_sim  FLOAT NOT NULL,
                created_at  TIMESTAMPTZ DEFAULT now(),
                UNIQUE(entity_a_id, entity_b_id)
            );
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS rag.chunk_entities (
                chunk_id        BIGINT REFERENCES rag.chunks(id) ON DELETE CASCADE,
                entity_id       BIGINT REFERENCES rag.entities(id) ON DELETE CASCADE,
                mention_text    TEXT,
                PRIMARY KEY(chunk_id, entity_id)
            );
        """)

        # ════════════════════════════════════════════════════════════════
        # Hierarchy: communities + hierarchical summaries
        # ════════════════════════════════════════════════════════════════

        cur.execute("""
            CREATE TABLE IF NOT EXISTS rag.communities (
                id                  BIGSERIAL PRIMARY KEY,
                collection          VARCHAR(100) NOT NULL,
                level               INTEGER NOT NULL,
                parent_id           BIGINT REFERENCES rag.communities(id),
                title               TEXT,
                summary             TEXT,
                summary_embedding   vector(768),
                entity_count        INTEGER DEFAULT 0,
                created_at          TIMESTAMPTZ DEFAULT now()
            );
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS rag.community_members (
                community_id    BIGINT REFERENCES rag.communities(id) ON DELETE CASCADE,
                entity_id       BIGINT REFERENCES rag.entities(id) ON DELETE CASCADE,
                PRIMARY KEY(community_id, entity_id)
            );
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS rag.summaries (
                id                  BIGSERIAL PRIMARY KEY,
                collection          VARCHAR(100) NOT NULL,
                level               INTEGER NOT NULL,
                parent_id           BIGINT REFERENCES rag.summaries(id),
                text                TEXT NOT NULL,
                embedding           vector(768),
                source_chunk_ids    BIGINT[] DEFAULT '{}',
                token_count         INTEGER,
                created_at          TIMESTAMPTZ DEFAULT now()
            );
        """)

        # ════════════════════════════════════════════════════════════════
        # Contextual retrieval & parent-child chunk hierarchy
        # ════════════════════════════════════════════════════════════════

        cur.execute("""
            CREATE TABLE IF NOT EXISTS rag.contextual_contexts (
                chunk_id            BIGINT PRIMARY KEY
                                        REFERENCES rag.chunks(id) ON DELETE CASCADE,
                context_text        TEXT NOT NULL,
                context_embedding   vector(768),
                model_used          VARCHAR(100),
                created_at          TIMESTAMPTZ DEFAULT now()
            );
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS rag.parent_chunks (
                id              BIGSERIAL PRIMARY KEY,
                document_id     BIGINT REFERENCES rag.documents(id) ON DELETE CASCADE,
                collection      VARCHAR(100) NOT NULL,
                content         TEXT NOT NULL,
                heading_path    TEXT[] DEFAULT '{}',
                token_count     INTEGER,
                embedding       vector(768),
                created_at      TIMESTAMPTZ DEFAULT now()
            );
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS rag.chunk_parents (
                chunk_id    BIGINT PRIMARY KEY
                                REFERENCES rag.chunks(id) ON DELETE CASCADE,
                parent_id   BIGINT REFERENCES rag.parent_chunks(id) ON DELETE SET NULL
            );
        """)

        # ════════════════════════════════════════════════════════════════
        # Quality, analytics & provenance
        # ════════════════════════════════════════════════════════════════

        cur.execute("""
            CREATE TABLE IF NOT EXISTS rag.query_log (
                id                  BIGSERIAL PRIMARY KEY,
                query_text          TEXT NOT NULL,
                collection          VARCHAR(100),
                strategy            VARCHAR(50),
                quality_score       FLOAT,
                latency_ms          INTEGER,
                corrective_action   VARCHAR(50),
                result_count        INTEGER,
                created_at          TIMESTAMPTZ DEFAULT now()
            );
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS rag.chunk_utility (
                chunk_id        BIGINT PRIMARY KEY
                                    REFERENCES rag.chunks(id) ON DELETE CASCADE,
                utility_ema     FLOAT DEFAULT 0.5,
                retrieve_count  INTEGER DEFAULT 0,
                use_count       INTEGER DEFAULT 0,
                last_used       TIMESTAMPTZ,
                updated_at      TIMESTAMPTZ DEFAULT now()
            );
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS rag.eval_questions (
                id                  BIGSERIAL PRIMARY KEY,
                collection          VARCHAR(100) NOT NULL,
                question            TEXT NOT NULL,
                ground_truth        TEXT,
                difficulty          VARCHAR(20),
                source_chunk_ids    BIGINT[] DEFAULT '{}',
                created_at          TIMESTAMPTZ DEFAULT now()
            );
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS rag.eval_runs (
                id                  BIGSERIAL PRIMARY KEY,
                collection          VARCHAR(100),
                run_name            VARCHAR(200),
                faithfulness        FLOAT,
                answer_relevancy    FLOAT,
                context_precision   FLOAT,
                context_recall      FLOAT,
                n_questions         INTEGER,
                created_at          TIMESTAMPTZ DEFAULT now()
            );
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS rag.provenance_chains (
                id                  BIGSERIAL PRIMARY KEY,
                query_log_id        BIGINT REFERENCES rag.query_log(id) ON DELETE CASCADE,
                answer_text         TEXT,
                overall_confidence  FLOAT,
                created_at          TIMESTAMPTZ DEFAULT now()
            );
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS rag.provenance_steps (
                id                  BIGSERIAL PRIMARY KEY,
                chain_id            BIGINT REFERENCES rag.provenance_chains(id)
                                        ON DELETE CASCADE,
                chunk_id            BIGINT REFERENCES rag.chunks(id) ON DELETE SET NULL,
                entity_id           BIGINT REFERENCES rag.entities(id) ON DELETE SET NULL,
                score_components    JSONB DEFAULT '{}',
                rank_position       INTEGER
            );
        """)

        conn.commit()

        # ════════════════════════════════════════════════════════════════
        # Plain (non-vector) indexes
        # ════════════════════════════════════════════════════════════════

        _plain_indexes = [
            ("rag_entities_collection",     "rag.entities(collection)"),
            ("rag_entities_type",           "rag.entities(entity_type)"),
            ("rag_relationships_source",    "rag.relationships(source_id)"),
            ("rag_relationships_target",    "rag.relationships(target_id)"),
            ("rag_relationships_coll",      "rag.relationships(collection)"),
            ("rag_communities_coll_level",  "rag.communities(collection, level)"),
            ("rag_summaries_coll_level",    "rag.summaries(collection, level)"),
            ("rag_parent_chunks_coll",      "rag.parent_chunks(collection)"),
            ("rag_parent_chunks_doc",       "rag.parent_chunks(document_id)"),
            ("rag_chunk_entities_entity",   "rag.chunk_entities(entity_id)"),
            ("rag_query_log_coll",          "rag.query_log(collection)"),
            ("rag_query_log_created",       "rag.query_log(created_at DESC)"),
            ("rag_chunk_utility_ema",       "rag.chunk_utility(utility_ema DESC)"),
        ]

        for idx_name, idx_target in _plain_indexes:
            try:
                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS {idx_name} ON {idx_target};"
                )
                conn.commit()
            except Exception as exc:
                conn.rollback()
                log.warning("Could not create plain index %s: %s", idx_name, exc)

        # ════════════════════════════════════════════════════════════════
        # HNSW vector indexes
        # ════════════════════════════════════════════════════════════════
        #
        # Each HNSW CREATE INDEX is committed individually so a failure on
        # one column (e.g. NULL-only column with no data yet) doesn't roll
        # back the whole schema migration.
        # ════════════════════════════════════════════════════════════════

        _hnsw_indexes = [
            ("rag_entities_emb",
             "rag.entities USING hnsw(embedding vector_cosine_ops)"),
            ("rag_relationships_ctx_emb",
             "rag.relationships USING hnsw(context_embedding vector_cosine_ops)"),
            ("rag_communities_summary_emb",
             "rag.communities USING hnsw(summary_embedding vector_cosine_ops)"),
            ("rag_summaries_emb",
             "rag.summaries USING hnsw(embedding vector_cosine_ops)"),
            ("rag_parent_chunks_emb",
             "rag.parent_chunks USING hnsw(embedding vector_cosine_ops)"),
            ("rag_contextual_ctx_emb",
             "rag.contextual_contexts USING hnsw(context_embedding vector_cosine_ops)"),
        ]

        for idx_name, idx_target in _hnsw_indexes:
            try:
                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS {idx_name} "
                    f"ON {idx_target} WITH (m=16, ef_construction=200);"
                )
                conn.commit()
                log.debug("HNSW index ready: %s", idx_name)
            except Exception as exc:
                conn.rollback()
                log.warning("Could not create HNSW index %s: %s", idx_name, exc)

    log.info("Ultra RAG schema ready.")


# ---------------------------------------------------------------------------
# Convenience accessor
# ---------------------------------------------------------------------------

def get_ultra_conn():
    """
    Return a psycopg2 connection with the Ultra RAG schema guaranteed to exist.

    Equivalent to: conn = get_conn(); create_ultra_schema(conn); return conn
    This is the primary entry point for all Ultra RAG modules.
    """
    conn = get_conn()
    try:
        create_ultra_schema(conn)
    except Exception as exc:
        log.error("Failed to ensure Ultra RAG schema: %s", exc)
        raise
    return conn
