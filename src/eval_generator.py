"""
Evaluation dataset generator: creates synthetic Q&A pairs at 3 difficulty levels.
Inspired by NeMo Data Designer approach for RAG evaluation.

Difficulty levels:
  extractive  — single-chunk, answer directly stated in the passage
  abstractive — multi-chunk, requires paraphrasing / synthesis
  multi_hop   — multi-chunk, requires connecting 2+ facts across passages
"""
import json
import logging
import random
import time
from typing import Optional

import psycopg2.extras

log = logging.getLogger(__name__)

# Default question distribution
_DEFAULT_DISTRIBUTION: dict[str, float] = {
    "extractive":  0.4,
    "abstractive": 0.4,
    "multi_hop":   0.2,
}

# How many chunks to sample for multi-chunk questions
_MULTI_CHUNK_SAMPLE = 3


# ---------------------------------------------------------------------------
# EvalDatasetGenerator
# ---------------------------------------------------------------------------

class EvalDatasetGenerator:
    """
    Generate synthetic question-answer pairs from a RAG collection.

    The generator samples chunks from ``rag.chunks`` and uses an LLM to create
    questions at three difficulty levels, inspired by NeMo Data Designer's
    approach of diversifying question types for robust evaluation.

    Usage::

        conn = get_conn()
        gen  = EvalDatasetGenerator(conn, "imds")
        questions = gen.generate_dataset(n_questions=50)
        gen.export_jsonl(questions, "/tmp/eval_imds.jsonl")
    """

    def __init__(self, conn, collection: str, llm_client=None):
        self.conn       = conn
        self.collection = collection
        self._llm       = llm_client

    # ------------------------------------------------------------------
    # LLM accessor (lazy)
    # ------------------------------------------------------------------

    @property
    def llm(self):
        if self._llm is None:
            from .llm import LLMClient  # noqa: PLC0415
            self._llm = LLMClient()
        return self._llm

    # ------------------------------------------------------------------
    # Question generators
    # ------------------------------------------------------------------

    def generate_extractive_question(self, chunk_content: str, chunk_id: int) -> dict:
        """
        Create a factoid question whose answer is directly stated in *chunk_content*.

        Parameters
        ----------
        chunk_content:
            Raw text of a single chunk.
        chunk_id:
            Database id of that chunk (stored in result for traceability).

        Returns
        -------
        dict with keys: question, answer, answer_span, difficulty, source_chunk_ids
        """
        prompt = (
            "Create a specific factoid question whose answer is directly stated in this passage.\n"
            "The question must be answerable from this passage alone.\n"
            "The answer_span should be the exact substring from the passage that answers it.\n\n"
            f"PASSAGE:\n{chunk_content[:3000]}\n\n"
            "Return JSON with exactly these keys:\n"
            '  {"question": str, "answer": str, "answer_span": str}'
        )
        system = (
            "You are a precise technical QA generator. "
            "Create clear, unambiguous questions that test exact factual recall. "
            "Return only valid JSON — no prose, no markdown fences."
        )
        result = self.llm.complete_json(prompt, system=system)
        if not result or "question" not in result:
            log.warning("Extractive question generation failed for chunk %d", chunk_id)
            return {}
        return {
            "question":        result.get("question", "").strip(),
            "answer":          result.get("answer", "").strip(),
            "answer_span":     result.get("answer_span", "").strip(),
            "difficulty":      "extractive",
            "source_chunk_ids": [chunk_id],
        }

    def generate_abstractive_question(self, chunks: list) -> dict:
        """
        Create a question requiring synthesis of multiple passages to answer.

        Parameters
        ----------
        chunks:
            List of dicts each having keys ``id`` and ``content``.

        Returns
        -------
        dict with keys: question, answer, difficulty, source_chunk_ids
        """
        if not chunks:
            return {}
        passages_text = "\n\n---\n\n".join(
            f"PASSAGE {i+1}:\n{ch['content'][:1500]}"
            for i, ch in enumerate(chunks)
        )
        prompt = (
            "Create a question that requires synthesising information from multiple passages "
            "to answer. The answer should paraphrase or integrate facts from 2+ passages "
            "rather than quoting any single one.\n\n"
            f"{passages_text}\n\n"
            "Return JSON with exactly these keys:\n"
            '  {"question": str, "answer": str}'
        )
        system = (
            "You are a technical QA generator creating synthesis-level questions. "
            "The question must require reading all provided passages. "
            "Return only valid JSON — no prose, no markdown fences."
        )
        result = self.llm.complete_json(prompt, system=system)
        if not result or "question" not in result:
            log.warning("Abstractive question generation failed")
            return {}
        return {
            "question":        result.get("question", "").strip(),
            "answer":          result.get("answer", "").strip(),
            "difficulty":      "abstractive",
            "source_chunk_ids": [ch["id"] for ch in chunks],
        }

    def generate_multihop_question(
        self,
        chunks: list,
        entity_names: Optional[list] = None,
    ) -> dict:
        """
        Create a multi-hop question requiring connecting facts from 2+ passages.

        Parameters
        ----------
        chunks:
            List of dicts with keys ``id`` and ``content``.
        entity_names:
            Optional list of known entity names to guide the question toward
            explicit bridging entities.

        Returns
        -------
        dict with keys: question, answer, reasoning_chain, difficulty, source_chunk_ids
        """
        if not chunks:
            return {}
        passages_text = "\n\n---\n\n".join(
            f"PASSAGE {i+1}:\n{ch['content'][:1200]}"
            for i, ch in enumerate(chunks)
        )
        entity_hint = ""
        if entity_names:
            entity_hint = f"\nKey entities to bridge: {', '.join(entity_names[:8])}"

        prompt = (
            "Create a multi-hop question that requires connecting facts from at least "
            "2 of the provided passages in sequence. Each hop in reasoning_chain should "
            "describe one inference step that links passages.\n\n"
            f"{passages_text}"
            f"{entity_hint}\n\n"
            "Return JSON with exactly these keys:\n"
            '  {"question": str, "answer": str, "reasoning_chain": [str, ...]}'
        )
        system = (
            "You are a technical QA generator creating multi-hop reasoning questions. "
            "The question should be impossible to answer from any single passage alone. "
            "Return only valid JSON — no prose, no markdown fences."
        )
        result = self.llm.complete_json(prompt, system=system)
        if not result or "question" not in result:
            log.warning("Multi-hop question generation failed")
            return {}
        return {
            "question":        result.get("question", "").strip(),
            "answer":          result.get("answer", "").strip(),
            "reasoning_chain": result.get("reasoning_chain", []),
            "difficulty":      "multi_hop",
            "source_chunk_ids": [ch["id"] for ch in chunks],
        }

    # ------------------------------------------------------------------
    # Batch dataset generation
    # ------------------------------------------------------------------

    def generate_dataset(
        self,
        n_questions: int = 50,
        distribution: Optional[dict] = None,
    ) -> list:
        """
        Generate a synthetic evaluation dataset with ``n_questions`` entries.

        Parameters
        ----------
        n_questions:
            Total number of questions to generate.
        distribution:
            Dict mapping difficulty → fraction.  Must sum to ~1.0.
            Defaults to {extractive: 0.4, abstractive: 0.4, multi_hop: 0.2}.

        Returns
        -------
        List of question dicts with db ids attached (from rag.eval_questions).
        """
        dist = distribution or _DEFAULT_DISTRIBUTION

        # Normalise distribution
        total = sum(dist.values())
        counts = {
            k: max(1, round(n_questions * v / total))
            for k, v in dist.items()
        }
        # Adjust rounding errors so total == n_questions
        diff = n_questions - sum(counts.values())
        if diff != 0:
            counts["abstractive"] = counts.get("abstractive", 0) + diff

        log.info(
            "Generating eval dataset: %d questions (%s)",
            n_questions,
            ", ".join(f"{k}={v}" for k, v in counts.items()),
        )

        # Sample all chunks up-front
        all_chunks = self._sample_chunks(n_questions * _MULTI_CHUNK_SAMPLE)
        if not all_chunks:
            log.error("No chunks available in collection '%s'", self.collection)
            return []

        # Optionally load entity names for multi-hop
        entity_names = self._load_entity_names(limit=50)

        questions: list[dict] = []
        t0 = time.time()

        # ── Extractive ────────────────────────────────────────────────
        for i in range(counts.get("extractive", 0)):
            if i >= len(all_chunks):
                break
            ch = all_chunks[i]
            q = self.generate_extractive_question(ch["content"], ch["id"])
            if q:
                q["ground_truth"] = q.get("answer", "")
                questions.append(q)

        # ── Abstractive ───────────────────────────────────────────────
        offset = counts.get("extractive", 0)
        for i in range(counts.get("abstractive", 0)):
            start = offset + i * _MULTI_CHUNK_SAMPLE
            batch = all_chunks[start: start + _MULTI_CHUNK_SAMPLE]
            if not batch:
                break
            q = self.generate_abstractive_question(batch)
            if q:
                q["ground_truth"] = q.get("answer", "")
                questions.append(q)

        # ── Multi-hop ─────────────────────────────────────────────────
        offset = counts.get("extractive", 0) + counts.get("abstractive", 0) * _MULTI_CHUNK_SAMPLE
        for i in range(counts.get("multi_hop", 0)):
            start = offset + i * _MULTI_CHUNK_SAMPLE
            batch = all_chunks[start: start + _MULTI_CHUNK_SAMPLE]
            if not batch:
                batch = random.sample(all_chunks, min(_MULTI_CHUNK_SAMPLE, len(all_chunks)))
            q = self.generate_multihop_question(batch, entity_names)
            if q:
                q["ground_truth"] = q.get("answer", "")
                questions.append(q)

        elapsed = time.time() - t0
        log.info("Generated %d questions in %.1fs", len(questions), elapsed)

        # ── Persist to DB ─────────────────────────────────────────────
        persisted = self._persist_questions(questions)

        return persisted

    def export_jsonl(self, questions: list, output_path: str) -> None:
        """
        Write *questions* to a newline-delimited JSON file at *output_path*.

        Each line is one question dict.  Existing file is overwritten.
        """
        from pathlib import Path  # noqa: PLC0415
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            for q in questions:
                f.write(json.dumps(q, ensure_ascii=False, default=str) + "\n")
        log.info("Exported %d questions to %s", len(questions), output_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_chunks(self, n: int) -> list:
        """Fetch up to *n* chunks from the collection (random order)."""
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT id, content, content_type, chunk_metadata
                FROM   rag.chunks
                WHERE  collection = %s
                  AND  token_count > 30
                ORDER  BY random()
                LIMIT  %s
            """, (self.collection, n))
            return [dict(r) for r in cur.fetchall()]

    def _load_entity_names(self, limit: int = 50) -> list:
        """Return a list of entity names from the collection (best-effort)."""
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT name FROM rag.entities
                    WHERE  collection = %s
                    ORDER  BY specificity DESC
                    LIMIT  %s
                """, (self.collection, limit))
                return [r[0] for r in cur.fetchall()]
        except Exception as exc:
            log.debug("Could not load entity names: %s", exc)
            return []

    def _persist_questions(self, questions: list) -> list:
        """
        INSERT each question into rag.eval_questions, attach the assigned
        db id to the dict, and return the augmented list.
        """
        persisted = []
        with self.conn.cursor() as cur:
            for q in questions:
                try:
                    cur.execute("""
                        INSERT INTO rag.eval_questions
                            (collection, question, ground_truth, difficulty, source_chunk_ids)
                        VALUES (%s, %s, %s, %s, %s)
                        RETURNING id
                    """, (
                        self.collection,
                        q.get("question", ""),
                        q.get("ground_truth", ""),
                        q.get("difficulty", "extractive"),
                        q.get("source_chunk_ids", []),
                    ))
                    row = cur.fetchone()
                    if row:
                        q["eval_question_id"] = row[0]
                    persisted.append(q)
                except Exception as exc:
                    self.conn.rollback()
                    log.warning("Failed to persist question: %s", exc)
                    persisted.append(q)
        try:
            self.conn.commit()
        except Exception as exc:
            log.error("Commit failed after persisting questions: %s", exc)
            self.conn.rollback()
        return persisted
