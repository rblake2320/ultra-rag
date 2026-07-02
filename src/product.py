"""Consumer product helpers for Ultra RAG."""
from __future__ import annotations

import re
from pathlib import Path

from .parsers import PARSERS


ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_ROOT = ROOT / "documents"


def safe_collection_name(value: str) -> str:
    """Normalize a user-facing workspace/collection name."""
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip()).strip("-_").lower()
    return cleaned or "default"


def collection_upload_dir(collection: str) -> Path:
    """Return the upload directory for a collection."""
    safe = safe_collection_name(collection)
    path = DOCUMENTS_ROOT if safe == "default" else DOCUMENTS_ROOT / safe
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_supported_upload(filename: str) -> bool:
    return Path(filename).suffix.lower() in PARSERS


def safe_upload_name(filename: str) -> str:
    """Strip path components and unsafe characters from upload filename."""
    name = Path(filename).name
    stem = re.sub(r"[^a-zA-Z0-9_.-]+", "-", Path(name).stem).strip("-_.") or "upload"
    suffix = Path(name).suffix.lower()
    return f"{stem}{suffix}"


def extractive_answer(query: str, results: list[dict], max_sources: int = 4) -> str:
    """Build a grounded extractive answer from retrieved passages."""
    if not results:
        return (
            "I did not find matching indexed content for that question. "
            "Upload or ingest more documents, then run the query again."
        )

    lines = [f"Based on the top retrieved sources for: {query}"]
    for idx, result in enumerate(results[:max_sources], start=1):
        content = " ".join((result.get("content") or "").split())
        snippet = content[:360] + ("..." if len(content) > 360 else "")
        context = result.get("context_prefix") or result.get("content_type") or "source"
        lines.append(f"{idx}. {context}: {snippet}")
    return "\n".join(lines)
