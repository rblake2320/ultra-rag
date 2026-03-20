"""
ColBERT retrieval via RAGatouille — late interaction token-level matching.

ColBERT outperforms dense bi-encoders on out-of-domain queries because it
scores every token pair rather than compressing documents to a single vector.
Results are fed as a third list into the existing RRF fusion in search.py.

Index storage: D:\\rag-ingest\\colbert_indexes\\{collection}\\

Usage:
    from src.colbert_retriever import ColBERTRetriever

    # Build index (one-time, ~10 min for 4K chunks on RTX 5090)
    retriever = ColBERTRetriever()
    retriever.build_index(conn, "imds")

    # Search
    results = retriever.search("scheduled maintenance procedures", top_k=10)

Alternatively via ultra_ingest.py:
    python ultra_ingest.py imds --stages colbert
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import psycopg2.extras

from .config import get_config

log = logging.getLogger(__name__)

# Default RAGatouille ColBERT model
_DEFAULT_COLBERT_MODEL = "colbert-ir/colbertv2.0"

# Where indexes are stored
_DEFAULT_INDEX_BASE = Path(__file__).resolve().parent.parent / "colbert_indexes"


class ColBERTRetriever:
    """
    Thin wrapper around RAGatouille for ColBERT dense-token retrieval.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID for ColBERT.  Defaults to colbertv2.0.
    index_base : Path or str
        Root directory for PLAID indexes (one subdirectory per collection).
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_COLBERT_MODEL,
        index_base: Optional[Path | str] = None,
    ) -> None:
        cfg = get_config()
        search_cfg = cfg.get("search", {})

        self._model_name = model_name

        # Allow config.yaml override for index path
        cfg_index_path = search_cfg.get("colbert_index_path")
        if index_base is not None:
            self._index_base = Path(index_base)
        elif cfg_index_path:
            self._index_base = Path(cfg_index_path)
        else:
            self._index_base = _DEFAULT_INDEX_BASE

        self._index_base.mkdir(parents=True, exist_ok=True)

        # Lazy-loaded RAGatouille instance and loaded collection name
        self._ragatouille: object = None
        self._loaded_collection: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_index(self, conn, collection: str) -> dict:
        """
        Load all chunks for *collection* from DB and build a PLAID index.

        Returns dict with stats: {collection, n_documents, index_path}.
        """
        log.info("ColBERT: loading chunks for collection '%s'", collection)
        docs, doc_ids = self._load_chunks(conn, collection)

        if not docs:
            log.warning("ColBERT: no chunks found for collection '%s'", collection)
            return {"collection": collection, "n_documents": 0, "index_path": None}

        log.info("ColBERT: building index for %d documents...", len(docs))

        ragatouille = self._get_ragatouille(new_index=True)

        index_path = str(self._index_base / collection)
        ragatouille.index(
            collection=docs,
            document_ids=doc_ids,
            index_name=collection,
            split_documents=False,  # we pre-chunked already
        )

        self._loaded_collection = collection
        log.info("ColBERT: index built at %s", index_path)
        return {
            "collection":  collection,
            "n_documents": len(docs),
            "index_path":  index_path,
        }

    def search(
        self,
        query: str,
        collection: str,
        top_k: int = 10,
    ) -> list:
        """
        Search the ColBERT index for *query*.

        Returns a list of dicts matching the format used by search.py:
            {id, content, score, source}

        Returns [] if the index does not exist or RAGatouille is not installed.
        """
        index_path = self._index_base / collection
        if not index_path.exists():
            log.debug(
                "ColBERT: index not found at %s — run build_index() first",
                index_path,
            )
            return []

        try:
            ragatouille = self._get_ragatouille(collection=collection)
            raw = ragatouille.search(query=query, k=top_k)
        except Exception as exc:
            log.warning("ColBERT: search failed: %s", exc)
            return []

        results = []
        for hit in raw:
            # RAGatouille returns: {content, score, document_id, ...}
            results.append({
                "id":      self._parse_doc_id(hit.get("document_id", "")),
                "content": hit.get("content", ""),
                "score":   float(hit.get("score", 0.0)),
                "source":  "colbert",
                # content_type, context_prefix, chunk_metadata, token_count will be
                # filled in by search.py's _enrich_colbert_results() for chunks that
                # do not already appear in keyword or vector results.
            })
        return results

    def index_exists(self, collection: str) -> bool:
        return (self._index_base / collection).exists()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_ragatouille(
        self,
        collection: Optional[str] = None,
        new_index: bool = False,
    ):
        """Return a RAGPretrainedModel instance, loading lazily."""
        try:
            from ragatouille import RAGPretrainedModel  # noqa: PLC0415
        except ImportError:
            raise ImportError(
                "RAGatouille is not installed. "
                "Run: pip install ragatouille>=0.0.8"
            )

        if new_index or self._ragatouille is None:
            self._ragatouille = RAGPretrainedModel.from_pretrained(
                self._model_name,
                index_root=str(self._index_base),
            )
            return self._ragatouille

        # Load existing index if collection changed
        if collection and collection != self._loaded_collection:
            index_path = self._index_base / collection
            if index_path.exists():
                self._ragatouille = RAGPretrainedModel.from_index(str(index_path))
                self._loaded_collection = collection
            else:
                self._ragatouille = RAGPretrainedModel.from_pretrained(
                    self._model_name,
                    index_root=str(self._index_base),
                )

        return self._ragatouille

    @staticmethod
    def _load_chunks(conn, collection: str) -> tuple[list, list]:
        """Return (texts, doc_ids) for all chunks in *collection*."""
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT id, content, context_prefix
                FROM rag.chunks
                WHERE collection = %s
                ORDER BY id
            """, (collection,))
            rows = cur.fetchall()

        docs    = []
        doc_ids = []
        for row in rows:
            prefix  = row.get("context_prefix") or ""
            content = row.get("content") or ""
            text    = f"{prefix}\n\n{content}".strip() if prefix else content
            docs.append(text)
            doc_ids.append(str(row["id"]))

        return docs, doc_ids

    @staticmethod
    def _parse_doc_id(doc_id_str: str) -> int:
        """Convert doc_id string back to int chunk id."""
        try:
            return int(doc_id_str)
        except (ValueError, TypeError):
            return -1


# ---------------------------------------------------------------------------
# Module-level singleton (for use inside search.py)
# ---------------------------------------------------------------------------

_retriever_cache: dict[str, ColBERTRetriever] = {}


def get_colbert_retriever(
    model_name: str = _DEFAULT_COLBERT_MODEL,
    index_base: Optional[Path] = None,
) -> ColBERTRetriever:
    """Return a cached ColBERTRetriever instance (one per model_name)."""
    key = model_name
    if key not in _retriever_cache:
        _retriever_cache[key] = ColBERTRetriever(
            model_name=model_name,
            index_base=index_base,
        )
    return _retriever_cache[key]
