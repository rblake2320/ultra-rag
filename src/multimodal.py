"""
Multimodal RAG: process images, tables, and complex layouts via VLM.
RAG-Anything approach: context-aware descriptions → KG entity insertion.

Supports:
  - Image description via Ollama llava (with fallback to filename stub)
  - Table description enhancement via LLM
  - DOCX/PDF image extraction → new image_description chunks
  - Pattern-based entity extraction from descriptions
"""
from __future__ import annotations

import base64
import logging
import re
from pathlib import Path
from typing import Optional

import httpx
import psycopg2.extras

from .config import get_config

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

_DEFAULT_OLLAMA_URL  = "http://localhost:11434"
_DEFAULT_VISION_MODEL = "llava"

# Patterns for lightweight entity extraction from descriptions
_SCREEN_CODE_RE    = re.compile(r"\b(?:screen|form|window)\s*(\d{3}[A-Z]?)\b", re.IGNORECASE)
_EQUIP_CODE_RE     = re.compile(r"\b[A-Z]{2,5}-\d{3,}\b")
_PROC_HEADING_RE   = re.compile(r"\b(?:step|procedure|process|task)\s+\d+", re.IGNORECASE)
_FIELD_LABEL_RE    = re.compile(r"(?:field|column|label)[:\s]+[\"']?([A-Z][A-Za-z /]{2,30})[\"']?")


# ---------------------------------------------------------------------------
# Standalone helper
# ---------------------------------------------------------------------------

def encode_image_base64(path: str) -> str:
    """
    Read an image file and return its content as a base64-encoded string.

    Parameters
    ----------
    path : str
        Absolute or relative path to the image file.

    Returns
    -------
    Base64-encoded string, or "" if the file cannot be read.
    """
    try:
        data = Path(path).read_bytes()
        return base64.b64encode(data).decode("utf-8")
    except Exception as exc:
        log.debug("encode_image_base64: could not read %r — %s", path, exc)
        return ""


# ---------------------------------------------------------------------------
# MultimodalProcessor
# ---------------------------------------------------------------------------

class MultimodalProcessor:
    """
    RAG-Anything-style multimodal processor for images and tables.

    Parameters
    ----------
    conn : psycopg2 connection
        Active database connection used for INSERT/UPDATE operations.
    collection : str
        Collection name used when inserting new image-description chunks.
    llm_client : optional
        A src.llm.LLMClient instance for table descriptions.  When None
        a new client is created lazily using config defaults.
    """

    def __init__(
        self,
        conn,
        collection: str,
        llm_client=None,
    ) -> None:
        self._conn       = conn
        self._collection = collection
        self._llm        = llm_client

        cfg              = get_config()
        llm_cfg          = cfg.get("llm", {})
        self._ollama_url = llm_cfg.get("ollama_url", _DEFAULT_OLLAMA_URL).rstrip("/")

    # ------------------------------------------------------------------
    # Lazy LLM accessor
    # ------------------------------------------------------------------

    @property
    def llm(self):
        if self._llm is None:
            from .llm import LLMClient  # noqa: PLC0415
            self._llm = LLMClient()
        return self._llm

    # ------------------------------------------------------------------
    # Vision model availability
    # ------------------------------------------------------------------

    def _vision_available(self) -> bool:
        """
        Check whether the Ollama llava (or any vision-capable) model is
        available by querying /api/tags.  Returns False on any error.
        """
        try:
            resp = httpx.get(
                f"{self._ollama_url}/api/tags",
                timeout=5.0,
            )
            resp.raise_for_status()
            models = [m.get("name", "") for m in resp.json().get("models", [])]
            # Accept any model that starts with "llava" (llava, llava:13b, llava:34b, etc.)
            return any(m.startswith("llava") for m in models)
        except Exception as exc:
            log.debug("_vision_available check failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Image description
    # ------------------------------------------------------------------

    def describe_image(
        self,
        image_path: str,
        surrounding_text: Optional[str] = None,
    ) -> str:
        """
        Generate a textual description of an image using Ollama llava.

        If llava is not available, falls back to a filename-based stub.

        Parameters
        ----------
        image_path : str
            Path to the image file.
        surrounding_text : str, optional
            Nearby document text used as context for the VLM.

        Returns
        -------
        Description string.
        """
        filename = Path(image_path).name

        if not self._vision_available():
            log.debug("describe_image: no vision model available, using stub for %r", filename)
            return f"Image: {filename} - visual content requires VLM processing"

        b64 = encode_image_base64(image_path)
        if not b64:
            log.warning("describe_image: could not encode %r", image_path)
            return f"Image: {filename} - could not read image data"

        # Build context-aware prompt
        context_note = ""
        if surrounding_text and surrounding_text.strip():
            context_note = (
                f"This image appears in documentation about: "
                f"{surrounding_text.strip()[:200]}\n\n"
            )

        prompt = (
            f"{context_note}"
            "Describe this technical image in detail. "
            "What does it show? "
            "What are the key elements, labels, values, or procedures depicted?"
        )

        payload = {
            "model":  _DEFAULT_VISION_MODEL,
            "prompt": prompt,
            "images": [b64],
            "stream": False,
            "options": {"num_predict": 600},
        }

        try:
            resp = httpx.post(
                f"{self._ollama_url}/api/generate",
                json=payload,
                timeout=120.0,
            )
            resp.raise_for_status()
            description = resp.json().get("response", "").strip()
            if description:
                return description
        except Exception as exc:
            log.warning("describe_image: Ollama call failed for %r: %s", filename, exc)

        return f"Image: {filename} - visual content requires VLM processing"

    # ------------------------------------------------------------------
    # Table description
    # ------------------------------------------------------------------

    def describe_table(
        self,
        table_content: str,
        heading_context: Optional[str] = None,
    ) -> str:
        """
        Generate an enhanced natural-language description of a data table.

        Parameters
        ----------
        table_content : str
            Raw text representation of the table (e.g. pipe-delimited or
            space-aligned columns as extracted from a DOCX/PDF).
        heading_context : str, optional
            The section heading under which the table appears.

        Returns
        -------
        Enhanced description string; falls back to the original content on error.
        """
        heading = heading_context or "technical documentation"

        prompt = (
            f"This is a data table from technical documentation. "
            f"Context: {heading}.\n\n"
            f"TABLE CONTENT:\n{table_content[:2000]}\n\n"
            "Summarize what data this table contains, what each column means, "
            "and any notable values or patterns."
        )

        try:
            description = self.llm.complete(prompt, max_tokens=400)
            if description.strip():
                return description.strip()
        except Exception as exc:
            log.warning("describe_table: LLM call failed: %s", exc)

        # Fallback: return original content unchanged
        return table_content

    # ------------------------------------------------------------------
    # Document image extraction
    # ------------------------------------------------------------------

    def process_document_images(self, doc_id: int, file_path: str) -> int:
        """
        Extract embedded images from a document and insert description chunks.

        Supports DOCX (via python-docx) and PDF (via pypdf / pdfplumber).
        Each extracted image is described via :meth:`describe_image` and
        stored as a new chunk with content_type='image_description'.

        Parameters
        ----------
        doc_id : int
            rag.documents.id for the source document.
        file_path : str
            Absolute path to the document file.

        Returns
        -------
        Number of image chunks created.
        """
        path = Path(file_path)
        suffix = path.suffix.lower()
        images_processed = 0

        if suffix in (".docx", ".doc"):
            images_processed = self._process_docx_images(doc_id, path)
        elif suffix == ".pdf":
            images_processed = self._process_pdf_images(doc_id, path)
        else:
            log.debug("process_document_images: unsupported format %r", suffix)

        return images_processed

    def _process_docx_images(self, doc_id: int, path: Path) -> int:
        """Extract and describe images from a DOCX file."""
        try:
            from docx import Document  # noqa: PLC0415
        except ImportError:
            log.warning(
                "_process_docx_images: python-docx not installed; skipping %r", str(path)
            )
            return 0

        try:
            doc = Document(str(path))
        except Exception as exc:
            log.warning("_process_docx_images: could not open %r: %s", str(path), exc)
            return 0

        count = 0
        img_index = 0

        for rel in doc.part.rels.values():
            if "image" not in rel.target_ref.lower():
                continue
            try:
                img_data  = rel.target_part.blob
                img_ext   = Path(rel.target_ref).suffix.lower() or ".png"
                img_bytes = img_data
            except Exception as exc:
                log.debug("DOCX image extraction failed for rel %r: %s", rel.target_ref, exc)
                continue

            # Write temp file, describe it, delete it
            import tempfile  # noqa: PLC0415
            with tempfile.NamedTemporaryFile(suffix=img_ext, delete=False) as tmp:
                tmp_path = tmp.name
                tmp.write(img_bytes)

            try:
                description = self.describe_image(tmp_path)
                self._insert_image_chunk(
                    doc_id=doc_id,
                    index=img_index,
                    description=description,
                    source_ref=rel.target_ref,
                )
                count += 1
                img_index += 1
            except Exception as exc:
                log.warning("DOCX image chunk insert failed: %s", exc)
            finally:
                try:
                    Path(tmp_path).unlink(missing_ok=True)
                except Exception:
                    pass

        return count

    def _process_pdf_images(self, doc_id: int, path: Path) -> int:
        """Extract and describe images from a PDF file."""
        # Try pypdf first, then pdfplumber
        try:
            from pypdf import PdfReader  # noqa: PLC0415
            reader = PdfReader(str(path))
            count  = 0
            img_index = 0

            for page_num, page in enumerate(reader.pages):
                for img_obj in page.images:
                    import tempfile  # noqa: PLC0415
                    ext = Path(img_obj.name).suffix.lower() or ".png"
                    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                        tmp.write(img_obj.data)
                        tmp_path = tmp.name

                    try:
                        description = self.describe_image(
                            tmp_path,
                            surrounding_text=f"Page {page_num + 1}",
                        )
                        self._insert_image_chunk(
                            doc_id=doc_id,
                            index=img_index,
                            description=description,
                            source_ref=f"page_{page_num + 1}_{img_obj.name}",
                        )
                        count += 1
                        img_index += 1
                    except Exception as exc:
                        log.warning("PDF image chunk insert failed (pypdf): %s", exc)
                    finally:
                        try:
                            Path(tmp_path).unlink(missing_ok=True)
                        except Exception:
                            pass

            return count

        except ImportError:
            pass  # fall through to pdfplumber
        except Exception as exc:
            log.warning("_process_pdf_images (pypdf) failed: %s", exc)
            return 0

        # pdfplumber fallback
        try:
            import pdfplumber  # noqa: PLC0415
            count = 0
            img_index = 0

            with pdfplumber.open(str(path)) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    for img in page.images:
                        # pdfplumber gives us bbox but not always raw bytes
                        # Skip if no data available
                        img_bytes = img.get("stream")
                        if not img_bytes:
                            continue

                        import tempfile  # noqa: PLC0415
                        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                            tmp.write(img_bytes)
                            tmp_path = tmp.name

                        try:
                            description = self.describe_image(
                                tmp_path,
                                surrounding_text=f"Page {page_num + 1}",
                            )
                            self._insert_image_chunk(
                                doc_id=doc_id,
                                index=img_index,
                                description=description,
                                source_ref=f"page_{page_num + 1}_img_{img_index}",
                            )
                            count += 1
                            img_index += 1
                        except Exception as exc:
                            log.warning("PDF image chunk insert (pdfplumber) failed: %s", exc)
                        finally:
                            try:
                                Path(tmp_path).unlink(missing_ok=True)
                            except Exception:
                                pass

            return count

        except ImportError:
            log.warning(
                "_process_pdf_images: neither pypdf nor pdfplumber installed; "
                "install one to enable PDF image extraction."
            )
            return 0
        except Exception as exc:
            log.warning("_process_pdf_images (pdfplumber) failed: %s", exc)
            return 0

    def _insert_image_chunk(
        self,
        doc_id: int,
        index: int,
        description: str,
        source_ref: str,
    ) -> int:
        """
        Insert a single image-description chunk into rag.chunks.

        Returns
        -------
        New chunk id.
        """
        import hashlib, json  # noqa: PLC0415

        content_hash = hashlib.sha256(description.encode()).hexdigest()[:64]
        stable_id    = hashlib.sha256(
            f"{self._collection}:img:{doc_id}:{index}:{content_hash}".encode()
        ).hexdigest()[:64]

        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO rag.chunks
                    (document_id, collection, chunk_index, content, content_hash,
                     context_prefix, content_type, token_count, chunk_metadata, stable_id)
                VALUES (%s, %s, %s, %s, %s, %s, 'image_description', %s, %s, %s)
                ON CONFLICT (document_id, content_hash) DO NOTHING
                RETURNING id
                """,
                (
                    doc_id,
                    self._collection,
                    -1000 - index,     # negative index to avoid collision with text chunks
                    description,
                    content_hash,
                    f"Image from document",
                    len(description.split()),
                    json.dumps({"image_source": source_ref}),
                    stable_id,
                ),
            )
            row = cur.fetchone()
        self._conn.commit()
        return row[0] if row else -1

    # ------------------------------------------------------------------
    # Table enhancement
    # ------------------------------------------------------------------

    def process_document_tables(self, doc_id: int, collection: str) -> int:
        """
        Find table chunks in the database for a document and enhance their
        context_prefix with an LLM-generated description.

        Parameters
        ----------
        doc_id : int
            rag.documents.id for the source document.
        collection : str
            Collection name for the chunk lookup.

        Returns
        -------
        Number of table chunks enhanced.
        """
        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, content, context_prefix, heading_path
                FROM rag.chunks
                WHERE document_id = %s
                  AND collection  = %s
                  AND content_type = 'table'
                ORDER BY chunk_index
                """,
                (doc_id, collection),
            )
            table_chunks = cur.fetchall()

        if not table_chunks:
            return 0

        count = 0
        for chunk in table_chunks:
            heading = None
            hp = chunk.get("heading_path")
            if hp and isinstance(hp, (list, tuple)) and len(hp) > 0:
                # heading_path is stored as text[] — last element is most specific
                last = hp[-1]
                heading = last if isinstance(last, str) else str(last)

            try:
                enhanced = self.describe_table(
                    table_content=chunk["content"],
                    heading_context=heading,
                )
            except Exception as exc:
                log.warning(
                    "process_document_tables: describe_table failed for chunk %d: %s",
                    chunk["id"], exc,
                )
                continue

            if not enhanced or enhanced == chunk["content"]:
                continue

            try:
                with self._conn.cursor() as cur:
                    cur.execute(
                        "UPDATE rag.chunks SET context_prefix = %s WHERE id = %s",
                        (enhanced[:2000], chunk["id"]),
                    )
                self._conn.commit()
                count += 1
            except Exception as exc:
                log.warning(
                    "process_document_tables: UPDATE failed for chunk %d: %s",
                    chunk["id"], exc,
                )
                self._conn.rollback()

        return count

    # ------------------------------------------------------------------
    # Entity extraction from descriptions
    # ------------------------------------------------------------------

    def extract_entities_from_description(
        self,
        description: str,
        chunk_id: int,
    ) -> list:
        """
        Lightweight entity extraction from image or table descriptions.

        Combines fast regex patterns (screen codes, equipment codes, procedure
        names, field labels) with an optional LLM pass for richer extraction.

        Parameters
        ----------
        description : str
            The image or table description text.
        chunk_id : int
            Associated chunk id (used for logging context only).

        Returns
        -------
        List of entity name strings (deduplicated, non-empty).
        """
        found: set[str] = set()

        # Screen/form codes
        for m in _SCREEN_CODE_RE.finditer(description):
            found.add(f"Screen {m.group(1)}")

        # Equipment / error codes
        for m in _EQUIP_CODE_RE.finditer(description):
            found.add(m.group(0))

        # Procedure references
        for m in _PROC_HEADING_RE.finditer(description):
            found.add(m.group(0).title())

        # Field labels
        for m in _FIELD_LABEL_RE.finditer(description):
            label = m.group(1).strip()
            if len(label) >= 3:
                found.add(label)

        # LLM-assisted extraction for any mentions not caught by patterns
        if description.strip():
            prompt = (
                f"List the key technical entities mentioned in this description. "
                f"Include equipment codes, screen names, procedure names, field labels, "
                f"and system identifiers. Return one entity per line, no bullets.\n\n"
                f"DESCRIPTION:\n{description[:800]}"
            )
            try:
                raw = self.llm.complete(prompt, max_tokens=200)
                for line in raw.splitlines():
                    entity = line.strip().lstrip("- •*").strip()
                    if entity and 3 <= len(entity) <= 80:
                        found.add(entity)
            except Exception as exc:
                log.debug(
                    "extract_entities_from_description LLM call failed (chunk %d): %s",
                    chunk_id, exc,
                )

        entities = sorted(e for e in found if e)
        log.debug(
            "extract_entities_from_description: %d entities for chunk %d",
            len(entities), chunk_id,
        )
        return entities
