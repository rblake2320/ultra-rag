"""
Ultra RAG system — src package.

All modules available via ``from src import X`` or ``from src.module import X``.

This init attempts to import every public symbol from each module.  Optional
dependencies (igraph, leidenalg, sentence-transformers, etc.) may not be
installed in all environments; those imports are wrapped in try/except blocks
so a missing optional package does not prevent the rest of the system from
loading.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Core (always required)
# ---------------------------------------------------------------------------

from .config import get_config
from .db import get_conn, create_schema

# ---------------------------------------------------------------------------
# Ultra RAG schema
# ---------------------------------------------------------------------------

try:
    from .db_ultra import create_ultra_schema, get_ultra_conn
except ImportError:
    pass

# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

try:
    from .llm import LLMClient
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

try:
    from .embedder import embed_chunks, _embed_batch
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Document processing
# ---------------------------------------------------------------------------

try:
    from .chunker import chunk_blocks
except ImportError:
    pass

try:
    from .parsers import parse_file
except ImportError:
    pass

try:
    from .parent_chunker import ParentChunker
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

try:
    from .search import search, hybrid_search
except ImportError:
    pass

try:
    from .hyde import HyDERetriever, rrf_merge
except ImportError:
    pass

try:
    from .query_router import QueryRouter, QUERY_TYPES, STRATEGY_MAP
except ImportError:
    pass

try:
    from .query_decomposer import QueryDecomposer
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Knowledge Graph
# ---------------------------------------------------------------------------

try:
    from .kg_extractor import KGExtractor
except ImportError:
    pass

try:
    from .kg_graph import KGGraph
except ImportError:
    pass

try:
    from .kg_communities import CommunityDetector
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Retrieval augmentation
# ---------------------------------------------------------------------------

try:
    from .contextual import ContextualRetriever
except ImportError:
    pass

try:
    from .corrective import CRAGEvaluator, QUALITY_LEVELS
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Multimodal (requires httpx; vision features need ollama llava)
# ---------------------------------------------------------------------------

try:
    from .multimodal import MultimodalProcessor, encode_image_base64
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

try:
    from .provenance import ProvenanceBuilder, build_score_components
except ImportError:
    pass

# ---------------------------------------------------------------------------
# File watcher
# ---------------------------------------------------------------------------

try:
    from .watcher import start_watcher
except ImportError:
    pass

# ---------------------------------------------------------------------------
# __all__: collect all successfully imported public names
# ---------------------------------------------------------------------------

__all__ = [name for name in dir() if not name.startswith("_")]
