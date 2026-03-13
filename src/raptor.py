"""
RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval.
GMM soft-clustering + recursive summarization → multi-scale summary tree.

Algorithm
---------
Level 0: leaf chunks (existing rag.chunks rows).
Level k: GMM-cluster the embeddings of level-(k-1) items → merge each cluster
         into one LLM summary → embed → store in rag.summaries.
Stop when fewer than 4 items remain at the current level or max_levels reached.

Retrieval uses the "collapsed" strategy: all summary levels are searched
simultaneously against the query embedding using pgvector cosine similarity.

References
----------
Sarthi et al. (2024) "RAPTOR: Recursive Abstractive Processing for
Tree-Organized Retrieval."  https://arxiv.org/abs/2401.18059
"""
import logging
import time
from typing import Optional

import numpy as np
import psycopg2.extras
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from .config import get_config
from .embedder import _embed_batch

log = logging.getLogger(__name__)

_MIN_ITEMS_TO_CONTINUE = 4   # stop recursion below this
_MAX_SUMMARY_TOKENS    = 500 # soft cap for LLM output
_PCA_TARGET_DIMS       = 2   # dimensionality for GMM input


def _load_llm_client():
    from .llm import LLMClient   # noqa: PLC0415
    from .config import get_config  # noqa: PLC0415
    _fast = get_config().get("llm", {}).get("fast_model", "qwen2.5:7b")
    return LLMClient(model=_fast)


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class RAPTOR:
    """
    Build a RAPTOR summary tree for a collection and provide tree search.

    Usage::

        conn  = get_conn()
        rap   = RAPTOR(conn, "my-docs")
        stats = rap.build_tree()
        print(stats)  # {"levels": 3, "summaries": 142}

        results = rap.search_tree(query_embedding, top_k=5)
    """

    def __init__(
        self,
        conn,
        collection:   str,
        llm_client=None,
        max_levels:   int = 3,
        cluster_size: int = 10,
    ):
        self.conn         = conn
        self.collection   = collection
        self._llm         = llm_client
        self.max_levels   = max_levels
        self.cluster_size = cluster_size
        cfg               = get_config()["embedding"]
        self._emb_model   = cfg["model"]
        self._emb_url     = cfg["ollama_url"]
        self._emb_batch   = cfg.get("batch_size", 32)
        self._emb_dims    = cfg.get("dimensions", 768)

    # ------------------------------------------------------------------
    # LLM accessor
    # ------------------------------------------------------------------

    @property
    def llm(self):
        if self._llm is None:
            self._llm = _load_llm_client()
        return self._llm

    # ------------------------------------------------------------------
    # Leaf embeddings
    # ------------------------------------------------------------------

    def _get_leaf_embeddings(self) -> tuple[list, np.ndarray]:
        """
        Fetch all embedded chunks for the collection.

        Returns
        -------
        tuple[list[int], np.ndarray]
            (chunk_ids, embeddings) where embeddings has shape
            (n_chunks, dimensions).
        """
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT id, embedding
                FROM   rag.chunks
                WHERE  collection = %s
                  AND  embedding IS NOT NULL
                ORDER  BY id
            """, (self.collection,))
            rows = cur.fetchall()

        if not rows:
            return [], np.empty((0, self._emb_dims))

        ids = [r[0] for r in rows]
        # psycopg2 returns pgvector as a string like "[0.1, 0.2, ...]"
        # or as a list depending on adapter.  Handle both.
        emb_list = []
        for _, raw in rows:
            if isinstance(raw, str):
                emb_list.append(
                    [float(x) for x in raw.strip("[]").split(",")]
                )
            else:
                emb_list.append(list(raw))

        return ids, np.array(emb_list, dtype=np.float32)

    # ------------------------------------------------------------------
    # GMM clustering
    # ------------------------------------------------------------------

    def _gmm_cluster(
        self, embeddings: np.ndarray, n_clusters: int
    ) -> np.ndarray:
        """
        Reduce *embeddings* to ``_PCA_TARGET_DIMS`` dimensions, then fit a
        Gaussian Mixture Model with *n_clusters* components.

        Parameters
        ----------
        embeddings:
            Shape (n, d).
        n_clusters:
            Desired number of Gaussian components.

        Returns
        -------
        np.ndarray
            Soft-assignment probabilities, shape (n, n_clusters).
        """
        n, d = embeddings.shape

        # Clamp n_clusters to avoid degenerate GMs
        n_clusters = max(2, min(n_clusters, n - 1))

        # ── Try UMAP first, fall back to PCA ────────────────────────────
        target_dims = min(_PCA_TARGET_DIMS, d, n - 1)
        try:
            import umap  # noqa: PLC0415
            reducer = umap.UMAP(
                n_components=target_dims,
                n_neighbors=min(15, n - 1),
                min_dist=0.0,
                random_state=42,
            )
            reduced = reducer.fit_transform(embeddings)
        except Exception:
            # UMAP unavailable or failed — use PCA
            pca     = PCA(n_components=target_dims)
            reduced = pca.fit_transform(embeddings)

        # ── Fit GMM ──────────────────────────────────────────────────────
        try:
            gm = GaussianMixture(
                n_components=n_clusters,
                covariance_type="full",
                random_state=42,
                max_iter=200,
                reg_covar=1e-5,   # regularise to avoid singular covariance
            )
            gm.fit(reduced)
            proba = gm.predict_proba(reduced)
        except Exception as exc:
            log.warning("GMM fitting failed (n=%d, k=%d): %s", n, n_clusters, exc)
            # Fallback: hard k-means-style uniform assignment
            proba = np.zeros((n, n_clusters), dtype=np.float32)
            for i in range(n):
                proba[i, i % n_clusters] = 1.0

        return proba

    # ------------------------------------------------------------------
    # Soft cluster assignment
    # ------------------------------------------------------------------

    @staticmethod
    def _assign_soft_clusters(
        proba: np.ndarray, threshold: float = 0.1
    ) -> dict[int, list[int]]:
        """
        Assign each item to every cluster where its probability exceeds
        *threshold*.  Items may belong to multiple clusters (soft assignment).

        Returns
        -------
        dict[int, list[int]]
            ``{cluster_id: [item_indices…]}``
        """
        assignments: dict[int, list[int]] = {
            c: [] for c in range(proba.shape[1])
        }
        for item_idx, row in enumerate(proba):
            for cluster_id, p in enumerate(row):
                if p >= threshold:
                    assignments[cluster_id].append(item_idx)

        # Remove empty clusters
        return {k: v for k, v in assignments.items() if v}

    # ------------------------------------------------------------------
    # Cluster summarisation
    # ------------------------------------------------------------------

    def summarize_cluster(self, texts: list[str], level: int) -> str:
        """
        Ask the LLM to produce a hierarchical summary of *texts*.

        Parameters
        ----------
        texts:
            List of text passages to synthesise.
        level:
            Current tree level (used in the prompt for framing).

        Returns
        -------
        str
            Summary text (max ~500 words).
        """
        level_label = "detailed" if level == 1 else "high-level" if level >= 2 else ""
        joined = "\n\n---\n\n".join(t[:800] for t in texts)  # truncate each

        prompt = (
            f"Synthesize these {len(texts)} text passages into a comprehensive "
            f"{level_label} summary that captures the key concepts:\n\n"
            f"{joined}"
        )
        system = (
            "You are a technical summariser. Output a single cohesive summary "
            f"of at most {_MAX_SUMMARY_TOKENS} words. No headers, no bullets."
        )

        try:
            return self.llm.complete(
                prompt, system=system, max_tokens=_MAX_SUMMARY_TOKENS * 2
            ).strip()
        except Exception as exc:
            log.warning("Cluster summarisation failed at level %d: %s", level, exc)
            # Fallback: truncated concatenation
            fallback = " ".join(t[:200] for t in texts[:5])
            return fallback[:1000]

    # ------------------------------------------------------------------
    # Embed and persist a batch of summaries
    # ------------------------------------------------------------------

    def _embed_and_store_summaries(
        self,
        summaries: list[tuple[str, list[int], int]],  # (text, source_ids, level)
    ) -> list[int]:
        """
        Embed summary texts and INSERT into rag.summaries.

        Returns
        -------
        list[int]
            New summary DB ids (same order as input).
        """
        texts = [s[0] for s in summaries]
        try:
            embeddings = _embed_batch(texts, self._emb_url, self._emb_model)
        except Exception as exc:
            log.error("Embedding summaries failed: %s", exc)
            embeddings = [None] * len(texts)

        summary_ids = []
        with self.conn.cursor() as cur:
            for (text, source_ids, level), emb in zip(summaries, embeddings):
                from .chunker import _count_tokens as _tk  # lightweight import
                try:
                    token_count = _tk(text)
                except Exception:
                    token_count = int(len(text.split()) * 1.33)

                cur.execute("""
                    INSERT INTO rag.summaries
                        (collection, level, text, source_chunk_ids,
                         token_count, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s::vector)
                    RETURNING id
                """, (
                    self.collection,
                    level,
                    text,
                    source_ids,
                    token_count,
                    emb if emb else None,
                ))
                summary_ids.append(cur.fetchone()[0])

        self.conn.commit()
        return summary_ids

    # ------------------------------------------------------------------
    # Tree builder
    # ------------------------------------------------------------------

    def build_tree(self) -> dict:
        """
        Recursively build the RAPTOR summary tree for the collection.

        Returns
        -------
        dict
            ``{"levels": int, "summaries": int}``
        """
        t0 = time.time()

        # ── Level 0: leaf chunks ─────────────────────────────────────────
        leaf_ids, leaf_embs = self._get_leaf_embeddings()
        if len(leaf_ids) < _MIN_ITEMS_TO_CONTINUE:
            log.info(
                "Only %d embedded chunks — RAPTOR tree requires at least %d.",
                len(leaf_ids), _MIN_ITEMS_TO_CONTINUE,
            )
            return {"levels": 0, "summaries": 0}

        log.info(
            "Building RAPTOR tree for '%s': %d leaf chunks, max %d levels.",
            self.collection, len(leaf_ids), self.max_levels,
        )

        # Fetch leaf texts for summarisation
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT id,
                       COALESCE(context_prefix || E'\n\n', '') || content
                FROM   rag.chunks
                WHERE  collection = %s AND embedding IS NOT NULL
                ORDER  BY id
            """, (self.collection,))
            leaf_rows = cur.fetchall()

        leaf_text_map: dict[int, str] = {r[0]: r[1] for r in leaf_rows}

        # ── Recursive tree construction ──────────────────────────────────
        # current_ids:  list of DB ids at the current level
        # current_embs: corresponding embeddings (np.ndarray, n×d)
        # current_texts: text for each item
        # is_summary:   True if the item is a rag.summaries row

        current_ids   = leaf_ids
        current_embs  = leaf_embs
        current_texts = [leaf_text_map.get(i, "") for i in current_ids]
        is_summary    = [False] * len(current_ids)

        total_summaries = 0

        for level in range(1, self.max_levels + 1):
            n = len(current_ids)
            if n < _MIN_ITEMS_TO_CONTINUE:
                log.info("Level %d: only %d items — stopping.", level, n)
                break

            n_clusters = max(2, n // self.cluster_size)
            log.info("Level %d: %d items → %d clusters.", level, n, n_clusters)

            proba       = self._gmm_cluster(current_embs, n_clusters)
            assignments = self._assign_soft_clusters(proba, threshold=0.1)

            # Build summaries for each cluster
            new_summaries: list[tuple[str, list[int], int]] = []
            cluster_source_ids: list[list[int]] = []

            for cluster_id, item_indices in assignments.items():
                if not item_indices:
                    continue

                # Collect source chunk IDs (for provenance)
                source_chunk_ids: list[int] = []
                cluster_texts: list[str]    = []

                for idx in item_indices:
                    cid  = current_ids[idx]
                    text = current_texts[idx]
                    cluster_texts.append(text)
                    if is_summary[idx]:
                        # Explode: get original chunk ids from the summary row
                        with self.conn.cursor() as cur:
                            cur.execute(
                                "SELECT source_chunk_ids FROM rag.summaries WHERE id=%s",
                                (cid,),
                            )
                            row = cur.fetchone()
                        if row and row[0]:
                            source_chunk_ids.extend(row[0])
                    else:
                        source_chunk_ids.append(cid)

                summary_text = self.summarize_cluster(cluster_texts, level)
                new_summaries.append((summary_text, source_chunk_ids, level))
                cluster_source_ids.append(source_chunk_ids)

            if not new_summaries:
                log.info("Level %d: no summaries produced — stopping.", level)
                break

            new_ids = self._embed_and_store_summaries(new_summaries)
            total_summaries += len(new_ids)
            log.info("  Level %d: %d summaries created.", level, len(new_ids))

            # Retrieve embeddings for newly created summaries to feed next level
            if len(new_ids) < _MIN_ITEMS_TO_CONTINUE:
                log.info(
                    "Level %d: only %d summaries — tree complete.", level, len(new_ids)
                )
                break

            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT id, embedding, text
                    FROM   rag.summaries
                    WHERE  id = ANY(%s)
                    ORDER  BY id
                """, (new_ids,))
                summary_rows = cur.fetchall()

            current_ids   = []
            current_embs_list: list[list[float]] = []
            current_texts = []

            for sid, raw_emb, text in summary_rows:
                current_ids.append(sid)
                current_texts.append(text)
                if raw_emb is None:
                    current_embs_list.append([0.0] * self._emb_dims)
                elif isinstance(raw_emb, str):
                    current_embs_list.append(
                        [float(x) for x in raw_emb.strip("[]").split(",")]
                    )
                else:
                    current_embs_list.append(list(raw_emb))

            current_embs = np.array(current_embs_list, dtype=np.float32)
            is_summary   = [True] * len(current_ids)

        elapsed = time.time() - t0
        stats = {
            "levels":    self.max_levels,
            "summaries": total_summaries,
            "elapsed_s": round(elapsed, 2),
        }
        log.info("RAPTOR tree built: %s", stats)
        return stats

    # ------------------------------------------------------------------
    # Tree search
    # ------------------------------------------------------------------

    def search_tree(
        self,
        query_embedding: list,
        top_k:    int = 5,
        strategy: str = "collapsed",
    ) -> list[dict]:
        """
        Search the RAPTOR summary tree.

        Parameters
        ----------
        query_embedding:
            Dense vector for the query (same dimensionality as chunk embeddings).
        top_k:
            Number of results to return.
        strategy:
            ``"collapsed"`` (the only strategy currently implemented):
            search all summary levels simultaneously and return the top-k by
            cosine similarity.

        Returns
        -------
        list[dict]
            Each dict has keys: ``id``, ``level``, ``text``, ``score``,
            ``source_chunk_ids``.
        """
        if strategy != "collapsed":
            log.warning(
                "Unknown RAPTOR strategy '%s' — defaulting to 'collapsed'.",
                strategy,
            )

        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT id,
                       level,
                       text,
                       source_chunk_ids,
                       1 - (embedding <=> %s::vector) AS score
                FROM   rag.summaries
                WHERE  collection = %s
                  AND  embedding  IS NOT NULL
                ORDER  BY embedding <=> %s::vector
                LIMIT  %s
            """, (
                query_embedding,
                self.collection,
                query_embedding,
                top_k,
            ))
            rows = cur.fetchall()

        return [
            {
                "id":               r[0],
                "level":            r[1],
                "text":             r[2],
                "source_chunk_ids": r[3] or [],
                "score":            float(r[4]) if r[4] is not None else 0.0,
            }
            for r in rows
        ]
