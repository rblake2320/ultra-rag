"""
Evaluation runner: computes RAGAS metrics for RAG system quality assessment.
Metrics: faithfulness, answer_relevancy, context_precision, context_recall.

Faithfulness:       Does the generated answer stay grounded in the retrieved context?
Answer relevancy:   Does the answer address the question asked?
Context precision:  Are the retrieved chunks relevant to the ground-truth answer?
Context recall:     Does the context cover all statements in the ground-truth answer?
"""
import json
import logging
import math
import time
from typing import Callable, Optional

import psycopg2.extras

log = logging.getLogger(__name__)

# How many hypothetical questions to generate for answer-relevancy scoring
_N_REVERSE_QUESTIONS = 3

# Character budget for context fed to faithfulness / recall LLM prompts
_CONTEXT_CHAR_BUDGET = 3000


# ---------------------------------------------------------------------------
# Cosine similarity (numpy-free fallback)
# ---------------------------------------------------------------------------

def _cosine(a: list, b: list) -> float:
    """Compute cosine similarity between two float lists."""
    if not a or not b or len(a) != len(b):
        return 0.0
    try:
        import numpy as np  # noqa: PLC0415
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    except ImportError:
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(y * y for y in b))
        return dot / (mag_a * mag_b + 1e-9)


# ---------------------------------------------------------------------------
# RAGASEvalRunner
# ---------------------------------------------------------------------------

class RAGASEvalRunner:
    """
    Compute RAGAS-style evaluation metrics for a RAG collection.

    Metrics implemented:
      - faithfulness        : LLM-graded answer grounding in context
      - answer_relevancy    : embedding-based reverse-question similarity
      - context_precision   : per-context LLM relevance → Mean Precision@k
      - context_recall      : ground-truth statement coverage by context

    Usage::

        conn = get_conn()
        runner = RAGASEvalRunner(conn, "my-docs")
        from src.search import search
        search_fn = lambda q, top_k: search(conn, q, "my-docs", top_k)
        results = runner.evaluate_dataset(questions, search_fn, run_name="baseline")
    """

    def __init__(self, conn, collection: str, llm_client=None):
        self.conn       = conn
        self.collection = collection
        self._llm       = llm_client

    # ------------------------------------------------------------------
    # LLM / embedder accessors (lazy)
    # ------------------------------------------------------------------

    @property
    def llm(self):
        if self._llm is None:
            from .llm import LLMClient  # noqa: PLC0415
            self._llm = LLMClient()
        return self._llm

    def _embed(self, texts: list) -> list:
        """Embed a list of texts, returning list of float-list embeddings."""
        from .config import get_config      # noqa: PLC0415
        from .embedder import _embed_batch  # noqa: PLC0415
        cfg = get_config()["embedding"]
        return _embed_batch(texts, cfg["ollama_url"], cfg["model"])

    # ------------------------------------------------------------------
    # Query runner
    # ------------------------------------------------------------------

    def run_query_for_eval(
        self,
        question: str,
        search_fn: Callable,
        ground_truth: str = "",
    ) -> dict:
        """
        Execute *search_fn(question, top_k=5)*, build an LLM answer from the
        retrieved context, and return a full eval bundle.

        Parameters
        ----------
        question:
            The question to run.
        search_fn:
            Callable accepting (query: str, top_k: int) → list of result dicts.
        ground_truth:
            Expected correct answer (from eval dataset), carried through for
            later metric computation.

        Returns
        -------
        dict:
            {question, answer, contexts: [str...], ground_truth,
             chunk_ids: [int...]}
        """
        try:
            raw_results = search_fn(question, top_k=5)
        except Exception as exc:
            log.warning("search_fn failed for question '%s': %s", question[:80], exc)
            raw_results = []

        contexts: list[str] = []
        chunk_ids: list[int] = []
        for r in raw_results:
            content = r.get("content", "")
            ctx     = r.get("context_prefix", "") or ""
            text    = f"{ctx}\n\n{content}".strip() if ctx else content
            if text:
                contexts.append(text)
            if r.get("id"):
                chunk_ids.append(r["id"])

        combined_context = "\n\n---\n\n".join(contexts[:5])

        # Generate answer from context
        if combined_context:
            answer_prompt = (
                f"Answer the following question using ONLY the provided context.\n"
                f"If the context does not contain the answer, say 'Not found in context'.\n\n"
                f"QUESTION: {question}\n\n"
                f"CONTEXT:\n{combined_context[:_CONTEXT_CHAR_BUDGET]}"
            )
            answer_system = (
                "You are a precise technical assistant. Answer concisely and accurately "
                "using only information from the provided context."
            )
            try:
                answer = self.llm.complete(answer_prompt, system=answer_system, max_tokens=400)
            except Exception as exc:
                log.warning("Answer generation failed: %s", exc)
                answer = "Error generating answer."
        else:
            answer = "No context retrieved."

        return {
            "question":    question,
            "answer":      answer.strip(),
            "contexts":    contexts,
            "chunk_ids":   chunk_ids,
            "ground_truth": ground_truth,
        }

    # ------------------------------------------------------------------
    # Individual metrics
    # ------------------------------------------------------------------

    def compute_faithfulness(self, answer: str, contexts: list) -> float:
        """
        LLM-graded measure of how faithfully *answer* is grounded in *contexts*.

        Prompts the LLM to score on 0-1. Returns 0.0 on any failure.
        """
        if not answer or not contexts:
            return 0.0

        combined = "\n\n---\n\n".join(contexts)[:_CONTEXT_CHAR_BUDGET]
        prompt = (
            "Rate how faithfully this answer is grounded in the provided context "
            "on a scale of 0 (completely unfaithful / hallucinated) to 1 (fully grounded).\n\n"
            f"ANSWER:\n{answer[:1000]}\n\n"
            f"CONTEXT:\n{combined}\n\n"
            "Return JSON with exactly these keys:\n"
            '  {"score": float, "reasoning": str}'
        )
        system = (
            "You are a rigorous factual-grounding evaluator. "
            "Score 1.0 only if every claim in the answer is explicitly supported by the context. "
            "Return only valid JSON."
        )
        try:
            result = self.llm.complete_json(prompt, system=system)
            score = float(result.get("score", 0.0))
            return max(0.0, min(1.0, score))
        except Exception as exc:
            log.warning("Faithfulness scoring failed: %s", exc)
            return 0.0

    def compute_answer_relevancy(self, question: str, answer: str) -> float:
        """
        Estimate answer relevancy by generating *_N_REVERSE_QUESTIONS* from the
        answer, embedding them, and computing mean cosine similarity with the
        original question embedding.

        Returns float in [0, 1]. Falls back to 0.5 on embedding/LLM failure.
        """
        if not question or not answer:
            return 0.0

        # Generate reverse questions
        gen_prompt = (
            f"Given this answer, generate {_N_REVERSE_QUESTIONS} different questions "
            f"that this answer could be responding to.\n\n"
            f"ANSWER: {answer[:800]}\n\n"
            f"Return JSON: {{\"questions\": [str, ...]}}"
        )
        gen_system = (
            "Generate concise, distinct questions. "
            "Return only valid JSON — no prose, no markdown fences."
        )
        try:
            result = self.llm.complete_json(gen_prompt, system=gen_system)
            gen_questions = result.get("questions", [])
        except Exception as exc:
            log.warning("Reverse question generation failed: %s", exc)
            return 0.5

        if not gen_questions:
            return 0.5

        # Embed original question + generated questions
        texts_to_embed = [question] + gen_questions[:_N_REVERSE_QUESTIONS]
        try:
            embeddings = self._embed(texts_to_embed)
        except Exception as exc:
            log.warning("Embedding for answer_relevancy failed: %s", exc)
            return 0.5

        if len(embeddings) < 2:
            return 0.5

        q_emb = embeddings[0]
        gen_embs = embeddings[1:]
        similarities = [_cosine(q_emb, g) for g in gen_embs if g]
        if not similarities:
            return 0.5
        return max(0.0, min(1.0, sum(similarities) / len(similarities)))

    def compute_context_precision(
        self,
        question: str,
        contexts: list,
        ground_truth: str,
    ) -> float:
        """
        Compute Mean Precision@k: for each context rank, ask the LLM whether
        it contributes to answering the question given the ground truth.

        Returns float in [0, 1].
        """
        if not contexts or not question:
            return 0.0

        relevance_flags: list[bool] = []
        system = (
            "You are a context relevance evaluator. "
            "Answer with valid JSON only: {\"relevant\": true|false, \"reasoning\": str}."
        )

        for ctx in contexts[:5]:
            prompt = (
                "Is this context passage relevant to answering the question, "
                "given the expected answer?\n\n"
                f"QUESTION: {question}\n\n"
                f"EXPECTED ANSWER: {ground_truth[:500]}\n\n"
                f"CONTEXT PASSAGE:\n{ctx[:1000]}\n\n"
                "Return JSON: {\"relevant\": true|false, \"reasoning\": str}"
            )
            try:
                result = self.llm.complete_json(prompt, system=system)
                relevance_flags.append(bool(result.get("relevant", False)))
            except Exception as exc:
                log.warning("Context precision LLM call failed: %s", exc)
                relevance_flags.append(False)

        if not relevance_flags:
            return 0.0

        # Mean Precision@k: average of cumulative precision at each relevant position
        cumulative_precision: list[float] = []
        n_relevant = 0
        for k, is_relevant in enumerate(relevance_flags, 1):
            if is_relevant:
                n_relevant += 1
                cumulative_precision.append(n_relevant / k)

        if not cumulative_precision:
            return 0.0
        return max(0.0, min(1.0, sum(cumulative_precision) / len(cumulative_precision)))

    def compute_context_recall(self, contexts: list, ground_truth: str) -> float:
        """
        What fraction of the ground-truth answer's statements are supported
        by the retrieved contexts?

        Prompts the LLM to enumerate ground-truth statements and classify
        each as covered / not covered by the context.

        Returns float in [0, 1].
        """
        if not contexts or not ground_truth:
            return 0.0

        combined = "\n\n---\n\n".join(contexts)[:_CONTEXT_CHAR_BUDGET]
        prompt = (
            "Break the GROUND TRUTH answer into individual atomic statements. "
            "For each statement, determine whether it is supported by the CONTEXT.\n\n"
            f"GROUND TRUTH: {ground_truth[:800]}\n\n"
            f"CONTEXT:\n{combined}\n\n"
            "Return JSON:\n"
            '  {"statements": [{"text": str, "supported": bool}, ...]}'
        )
        system = (
            "You are a rigorous factual-coverage evaluator. "
            "Be precise — mark a statement as supported only if the context "
            "explicitly contains that information. "
            "Return only valid JSON."
        )
        try:
            result = self.llm.complete_json(prompt, system=system)
            statements = result.get("statements", [])
        except Exception as exc:
            log.warning("Context recall LLM call failed: %s", exc)
            return 0.0

        if not statements:
            return 0.0

        n_supported = sum(1 for s in statements if s.get("supported", False))
        return max(0.0, min(1.0, n_supported / len(statements)))

    # ------------------------------------------------------------------
    # Full dataset evaluation
    # ------------------------------------------------------------------

    def evaluate_dataset(
        self,
        questions: list,
        search_fn: Callable,
        run_name: Optional[str] = None,
    ) -> dict:
        """
        Evaluate all *questions* using *search_fn* and compute RAGAS metrics.

        Parameters
        ----------
        questions:
            List of question dicts, each with at minimum a "question" key.
            Optionally includes "ground_truth".
        search_fn:
            Callable ``(query: str, top_k: int) -> list``.
        run_name:
            Optional human-readable name stored in rag.eval_runs.

        Returns
        -------
        dict:
            {faithfulness, answer_relevancy, context_precision,
             context_recall, n_questions, run_name}
        """
        if not questions:
            log.warning("evaluate_dataset: no questions provided")
            return {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0,
                "n_questions": 0,
                "run_name": run_name,
            }

        scores: dict[str, list[float]] = {
            "faithfulness":      [],
            "answer_relevancy":  [],
            "context_precision": [],
            "context_recall":    [],
        }

        t0 = time.time()
        log.info("Evaluating %d questions (run=%s)…", len(questions), run_name or "unnamed")

        for i, q_dict in enumerate(questions, 1):
            question    = q_dict.get("question", "")
            ground_truth = q_dict.get("ground_truth", "") or q_dict.get("answer", "")

            if not question:
                log.debug("Skipping empty question at index %d", i)
                continue

            log.info("  [%d/%d] %s…", i, len(questions), question[:70])

            # Run retrieval + answer generation
            try:
                bundle = self.run_query_for_eval(question, search_fn, ground_truth)
            except Exception as exc:
                log.error("run_query_for_eval failed for q%d: %s", i, exc)
                continue

            answer   = bundle["answer"]
            contexts = bundle["contexts"]

            # Compute each metric individually; failures default to 0.0
            try:
                faith = self.compute_faithfulness(answer, contexts)
                scores["faithfulness"].append(faith)
            except Exception as exc:
                log.warning("Faithfulness failed q%d: %s", i, exc)
                scores["faithfulness"].append(0.0)

            try:
                rel = self.compute_answer_relevancy(question, answer)
                scores["answer_relevancy"].append(rel)
            except Exception as exc:
                log.warning("Answer relevancy failed q%d: %s", i, exc)
                scores["answer_relevancy"].append(0.0)

            try:
                prec = self.compute_context_precision(question, contexts, ground_truth)
                scores["context_precision"].append(prec)
            except Exception as exc:
                log.warning("Context precision failed q%d: %s", i, exc)
                scores["context_precision"].append(0.0)

            try:
                recall = self.compute_context_recall(contexts, ground_truth)
                scores["context_recall"].append(recall)
            except Exception as exc:
                log.warning("Context recall failed q%d: %s", i, exc)
                scores["context_recall"].append(0.0)

        # ── Aggregate ─────────────────────────────────────────────────
        def _mean(lst: list) -> float:
            return round(sum(lst) / len(lst), 4) if lst else 0.0

        n_evaluated = len(scores["faithfulness"])
        summary = {
            "faithfulness":      _mean(scores["faithfulness"]),
            "answer_relevancy":  _mean(scores["answer_relevancy"]),
            "context_precision": _mean(scores["context_precision"]),
            "context_recall":    _mean(scores["context_recall"]),
            "n_questions":       n_evaluated,
            "run_name":          run_name,
        }

        elapsed = time.time() - t0
        log.info(
            "Evaluation complete in %.1fs — F=%.3f AR=%.3f CP=%.3f CR=%.3f",
            elapsed,
            summary["faithfulness"],
            summary["answer_relevancy"],
            summary["context_precision"],
            summary["context_recall"],
        )

        # ── Persist to rag.eval_runs ──────────────────────────────────
        self._persist_run(summary)

        return summary

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _persist_run(self, summary: dict) -> None:
        """Insert a row into rag.eval_runs (best-effort)."""
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO rag.eval_runs
                        (collection, run_name, faithfulness, answer_relevancy,
                         context_precision, context_recall, n_questions)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    self.collection,
                    summary.get("run_name"),
                    summary["faithfulness"],
                    summary["answer_relevancy"],
                    summary["context_precision"],
                    summary["context_recall"],
                    summary["n_questions"],
                ))
                run_id = cur.fetchone()[0]
            self.conn.commit()
            summary["run_id"] = run_id
            log.info("Eval run persisted with id=%d", run_id)
        except Exception as exc:
            self.conn.rollback()
            log.warning("Failed to persist eval run: %s", exc)
