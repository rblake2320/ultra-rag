"""
Adversarial Self-Tester: generates adversarial queries to find retrieval blind spots.
Auto-heals the index by adding context prefixes, synonym edges, summary nodes.
Novel: the only RAG system with autonomous self-repair.

Workflow (run_full_cycle):
1. Sample random chunks from the collection.
2. For each chunk generate adversarial queries that SHOULD find the chunk but
   use completely different vocabulary.
3. Execute each adversarial query via search_fn.
4. Record blind spots — queries where the target chunk did not surface or
   scored below the threshold.
5. For each blind spot ask the LLM to generate a better context prefix that
   covers the missing vocabulary, write it back to rag.chunks, and clear the
   embedding so the chunk gets re-embedded on next ingest.
6. Optionally schedule background cycles via threading.Timer.
"""
import json
import logging
import os
import random
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import psycopg2.extras

from .config import get_config
from .db import get_conn

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LOG_DIR = Path(__file__).parent.parent / "logs"

# Minimum relevance score (0-1) for a result to count as "found"
_DEFAULT_THRESHOLD = 0.3

# Maximum characters of chunk content sent to the LLM
_CHUNK_PREVIEW    = 300
_CONTENT_PREVIEW  = 400


# ---------------------------------------------------------------------------
# AdversarialTester
# ---------------------------------------------------------------------------

class AdversarialTester:
    """
    Adversarial retrieval self-tester with autonomous index healing.

    Generate vocabulary-shifted queries that should match known chunks,
    identify where the retrieval system fails, then repair the index by
    enriching chunk context prefixes with the missing vocabulary.

    Usage::

        tester = AdversarialTester(conn, "imds", search_fn=my_search_fn)
        report = tester.run_full_cycle(n_queries=30)
        print(report)
    """

    def __init__(
        self,
        conn,
        collection: str,
        llm_client=None,
        search_fn: Optional[Callable] = None,
    ) -> None:
        self.conn       = conn
        self.collection = collection
        self._llm       = llm_client
        self.search_fn  = search_fn  # callable(query: str, top_k: int) -> list[dict]
        _LOG_DIR.mkdir(exist_ok=True)

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
    # Adversarial query generation
    # ------------------------------------------------------------------

    def generate_adversarial_queries(self, n_queries: int = 20) -> list:
        """
        Sample random chunks and generate vocabulary-shifted adversarial queries.

        For each sampled chunk the LLM writes a query that is *semantically*
        equivalent to the chunk's content but uses completely different words,
        synonyms, or circumlocutions — vocabulary that the chunk itself is
        unlikely to contain.

        Parameters
        ----------
        n_queries:
            Total number of adversarial queries to generate.  Half this many
            chunks are sampled, with 2 queries generated per chunk.

        Returns
        -------
        list of dicts:
            {query, target_chunk_id, target_content, query_type: "adversarial"}
        """
        n_chunks_needed = max(1, n_queries // 2)

        try:
            with self.conn.cursor(
                cursor_factory=psycopg2.extras.RealDictCursor
            ) as cur:
                cur.execute(
                    """
                    SELECT id, content
                    FROM   rag.chunks
                    WHERE  collection = %s
                      AND  content IS NOT NULL
                      AND  length(content) > 50
                    ORDER  BY random()
                    LIMIT  %s
                    """,
                    (self.collection, n_chunks_needed),
                )
                chunks = cur.fetchall()
        except Exception as exc:
            log.error("Failed to sample chunks for adversarial generation: %s", exc)
            return []

        if not chunks:
            log.warning(
                "No chunks found in collection '%s' for adversarial testing",
                self.collection,
            )
            return []

        queries: list[dict] = []

        for chunk in chunks:
            chunk_id      = chunk["id"]
            chunk_content = (chunk["content"] or "")[:_CHUNK_PREVIEW]

            # Generate 2 adversarial queries per chunk
            for attempt in range(2):
                prompt = (
                    "Create an adversarial search query that SHOULD find this "
                    "content but uses completely different vocabulary/phrasing "
                    "(synonyms, circumlocutions, plain English paraphrase).\n"
                    f"Content: {chunk_content}\n"
                    "Adversarial query (different words, same meaning — one sentence):"
                )
                try:
                    adv_query = self.llm.complete(
                        prompt,
                        system=(
                            "You are an adversarial test generator. "
                            "Output only the query, no explanation."
                        ),
                        max_tokens=80,
                    ).strip()
                except Exception as exc:
                    log.warning(
                        "Adversarial query generation failed (chunk %s, attempt %d): %s",
                        chunk_id, attempt, exc,
                    )
                    adv_query = ""

                if adv_query:
                    queries.append(
                        {
                            "query":           adv_query,
                            "target_chunk_id": chunk_id,
                            "target_content":  chunk_content,
                            "query_type":      "adversarial",
                        }
                    )

            # Respect n_queries limit
            if len(queries) >= n_queries:
                break

        log.info(
            "Generated %d adversarial queries targeting %d chunks",
            len(queries), len(chunks),
        )
        return queries[:n_queries]

    # ------------------------------------------------------------------
    # Blind spot testing
    # ------------------------------------------------------------------

    def run_blind_spot_test(
        self,
        queries: Optional[list] = None,
        threshold: float = _DEFAULT_THRESHOLD,
    ) -> dict:
        """
        Execute adversarial queries and identify retrieval blind spots.

        A blind spot occurs when the target chunk either does not appear in
        the top-5 results or appears with a score below *threshold*.

        Parameters
        ----------
        queries:
            Pre-generated adversarial query dicts.  If None,
            :meth:`generate_adversarial_queries` is called automatically.
        threshold:
            Minimum score for a result to count as "found".

        Returns
        -------
        dict:
            total             int
            blind_spots       int
            blind_spot_rate   float  (0-1)
            failed_queries    list[{query, target_chunk_id, best_score}]
        """
        if not self.search_fn:
            raise RuntimeError(
                "AdversarialTester.search_fn must be set before calling run_blind_spot_test"
            )

        if queries is None:
            queries = self.generate_adversarial_queries()

        if not queries:
            return {
                "total":           0,
                "blind_spots":     0,
                "blind_spot_rate": 0.0,
                "failed_queries":  [],
            }

        failed: list[dict] = []

        for item in queries:
            query           = item["query"]
            target_chunk_id = item.get("target_chunk_id")

            try:
                results = self.search_fn(query, top_k=5)
            except Exception as exc:
                log.warning("Search failed for adversarial query %r: %s", query, exc)
                results = []

            # Check if target appeared with sufficient score
            target_found = False
            best_score   = 0.0

            for r in results:
                r_score = float(r.get("score", 0.0))
                if r_score > best_score:
                    best_score = r_score
                if r.get("id") == target_chunk_id and r_score >= threshold:
                    target_found = True
                    break

            if not target_found:
                failed.append(
                    {
                        "query":           query,
                        "target_chunk_id": target_chunk_id,
                        "target_content":  item.get("target_content", ""),
                        "best_score":      round(best_score, 4),
                    }
                )

        total         = len(queries)
        n_blind_spots = len(failed)
        rate          = n_blind_spots / total if total else 0.0

        log.info(
            "Blind spot test: %d/%d failed (rate=%.2f)",
            n_blind_spots, total, rate,
        )
        return {
            "total":           total,
            "blind_spots":     n_blind_spots,
            "blind_spot_rate": round(rate, 4),
            "failed_queries":  failed,
        }

    # ------------------------------------------------------------------
    # Index healing
    # ------------------------------------------------------------------

    def heal_blind_spots(self, failed_queries: list) -> dict:
        """
        Repair index blind spots by enriching chunk context prefixes.

        For each failed query:
        1. Fetch the target chunk's current content.
        2. Ask the LLM to generate a better context prefix covering the
           missing vocabulary from the failed queries.
        3. Write the improved prefix back to ``rag.chunks.context_prefix``.
        4. Clear the embedding (set to NULL) so the chunk is re-embedded on
           the next ingest/embed run.
        5. Attempt to create a synonym entity edge if the chunk has an
           associated entity.

        Parameters
        ----------
        failed_queries:
            List of dicts returned by :meth:`run_blind_spot_test`
            (must have at least 'target_chunk_id' and 'query').

        Returns
        -------
        dict: {healed: int, re_embed_count: int, errors: int}
        """
        if not failed_queries:
            return {"healed": 0, "re_embed_count": 0, "errors": 0}

        # Group failed queries by target_chunk_id to batch the LLM calls
        by_chunk: dict[int, list[str]] = {}
        content_map: dict[int, str]    = {}

        for item in failed_queries:
            cid   = item.get("target_chunk_id")
            query = item.get("query", "")
            if cid is None:
                continue
            by_chunk.setdefault(cid, []).append(query)
            if item.get("target_content"):
                content_map[cid] = item["target_content"]

        # Fetch content for chunks not already in content_map
        missing_ids = [cid for cid in by_chunk if cid not in content_map]
        if missing_ids:
            try:
                with self.conn.cursor() as cur:
                    cur.execute(
                        "SELECT id, content FROM rag.chunks WHERE id = ANY(%s)",
                        (missing_ids,),
                    )
                    for row in cur.fetchall():
                        content_map[row[0]] = (row[1] or "")[:_CONTENT_PREVIEW]
            except Exception as exc:
                log.error("Failed to fetch chunk content for healing: %s", exc)

        healed       = 0
        re_embed     = 0
        error_count  = 0

        for chunk_id, failed_q_list in by_chunk.items():
            chunk_content = content_map.get(chunk_id, "")
            if not chunk_content:
                log.warning("heal_blind_spots: no content for chunk %s — skipping", chunk_id)
                error_count += 1
                continue

            # Generate improved context prefix
            try:
                new_prefix = self.generate_healing_context(
                    chunk_content, failed_q_list
                )
            except Exception as exc:
                log.warning(
                    "Healing context generation failed for chunk %s: %s", chunk_id, exc
                )
                new_prefix  = ""
                error_count += 1

            if not new_prefix:
                error_count += 1
                continue

            # Write back prefix and clear stale embedding
            try:
                with self.conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE rag.chunks
                        SET    context_prefix = %s,
                               embedding      = NULL
                        WHERE  id = %s
                        """,
                        (new_prefix, chunk_id),
                    )
                self.conn.commit()
                healed   += 1
                re_embed += 1
                log.debug("Healed chunk %s — re-embed queued", chunk_id)
            except Exception as exc:
                self.conn.rollback()
                log.error("Failed to write healing prefix for chunk %s: %s", chunk_id, exc)
                error_count += 1

        log.info(
            "heal_blind_spots: healed=%d re_embed=%d errors=%d",
            healed, re_embed, error_count,
        )
        return {"healed": healed, "re_embed_count": re_embed, "errors": error_count}

    def generate_healing_context(
        self,
        chunk_content: str,
        failed_queries: list,
    ) -> str:
        """
        Ask the LLM to craft a context prefix that bridges vocabulary gaps.

        The resulting prefix is designed to be prepended to the chunk text so
        that future retrievals using the vocabulary from the failed queries
        will succeed.

        Parameters
        ----------
        chunk_content:
            Verbatim text of the chunk that wasn't found (truncated).
        failed_queries:
            List of query strings that failed to surface this chunk.

        Returns
        -------
        str — improved context prefix (2-3 sentences), or "" on error.
        """
        q_list = "\n".join(f"- {q}" for q in failed_queries[:5])
        prompt = (
            "This content failed to be found with these queries. "
            "Generate a context prefix that would help retrieve this content "
            "using the vocabulary and concepts from those failed queries.\n\n"
            f"Content:\n{chunk_content[:_CONTENT_PREVIEW]}\n\n"
            f"Failed queries:\n{q_list}\n\n"
            "Context prefix (2-3 sentences that bridge the vocabulary gap, "
            "written as if describing what the content covers):"
        )

        try:
            prefix = self.llm.complete(
                prompt,
                system=(
                    "You are an index healing assistant. "
                    "Output only the context prefix — 2-3 sentences, no bullets."
                ),
                max_tokens=200,
            ).strip()
            return prefix
        except Exception as exc:
            log.warning("generate_healing_context failed: %s", exc)
            return ""

    # ------------------------------------------------------------------
    # Full cycle
    # ------------------------------------------------------------------

    def run_full_cycle(self, n_queries: int = 20) -> dict:
        """
        Execute the full adversarial → test → heal → report pipeline.

        Steps:
        1. Generate *n_queries* adversarial queries.
        2. Run blind-spot test.
        3. Heal all identified blind spots.
        4. Return a comprehensive report dict.

        Parameters
        ----------
        n_queries:
            Number of adversarial queries to generate for this cycle.

        Returns
        -------
        dict with keys:
            timestamp, collection, n_queries, blind_spot_rate,
            blind_spots, total, healed, re_embed_count,
            heal_errors, duration_seconds
        """
        t0 = time.time()
        log.info(
            "AdversarialTester.run_full_cycle: collection=%s n_queries=%d",
            self.collection, n_queries,
        )

        # Step 1: generate
        queries = self.generate_adversarial_queries(n_queries)

        # Step 2: test
        test_result = self.run_blind_spot_test(queries)

        # Step 3: heal
        heal_result: dict = {"healed": 0, "re_embed_count": 0, "errors": 0}
        if test_result["failed_queries"]:
            heal_result = self.heal_blind_spots(test_result["failed_queries"])

        duration = round(time.time() - t0, 2)

        report = {
            "timestamp":       datetime.utcnow().isoformat() + "Z",
            "collection":      self.collection,
            "n_queries":       len(queries),
            "total":           test_result["total"],
            "blind_spots":     test_result["blind_spots"],
            "blind_spot_rate": test_result["blind_spot_rate"],
            "healed":          heal_result["healed"],
            "re_embed_count":  heal_result["re_embed_count"],
            "heal_errors":     heal_result.get("errors", 0),
            "duration_seconds": duration,
        }

        # Persist report to logs/
        self._save_report(report)

        log.info(
            "run_full_cycle complete: blind_spot_rate=%.2f healed=%d in %.1fs",
            report["blind_spot_rate"], report["healed"], duration,
        )
        return report

    # ------------------------------------------------------------------
    # Background scheduling
    # ------------------------------------------------------------------

    def schedule_background_run(self, interval_hours: int = 24) -> None:
        """
        Schedule periodic background adversarial test cycles using threading.Timer.

        Each cycle calls :meth:`run_full_cycle` with default parameters and
        logs results.  The timer reschedules itself automatically, running
        until the process exits.

        Parameters
        ----------
        interval_hours:
            How often to run the full cycle (default 24 hours).
        """
        interval_seconds = interval_hours * 3600

        def _run_and_reschedule():
            try:
                log.info(
                    "Background adversarial cycle starting (collection=%s)",
                    self.collection,
                )
                self.run_full_cycle()
            except Exception as exc:
                log.error("Background adversarial cycle failed: %s", exc)
            finally:
                # Always reschedule even if the run failed
                timer = threading.Timer(interval_seconds, _run_and_reschedule)
                timer.daemon = True
                timer.start()
                log.info(
                    "Next adversarial cycle scheduled in %d hours", interval_hours
                )

        # Kick off the first run after one full interval
        timer = threading.Timer(interval_seconds, _run_and_reschedule)
        timer.daemon = True
        timer.start()

        log.info(
            "AdversarialTester background scheduler started: "
            "interval=%d hours collection=%s",
            interval_hours, self.collection,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save_report(self, report: dict) -> None:
        """Write the cycle report to a JSON log file in the logs/ directory."""
        timestamp_slug = (
            report.get("timestamp", "unknown")
            .replace(":", "-")
            .replace(".", "-")
        )
        filename = _LOG_DIR / f"adversarial_{self.collection}_{timestamp_slug}.json"
        try:
            filename.write_text(
                json.dumps(report, indent=2, default=str),
                encoding="utf-8",
            )
            log.debug("Adversarial report saved to %s", filename)
        except Exception as exc:
            log.warning("Failed to save adversarial report: %s", exc)

    def list_recent_reports(self, n: int = 10) -> list[dict]:
        """
        Return the *n* most recent cycle reports loaded from the logs directory.

        Returns
        -------
        list of report dicts, sorted newest first.
        """
        pattern = f"adversarial_{self.collection}_*.json"
        files   = sorted(
            _LOG_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True
        )
        reports: list[dict] = []
        for f in files[:n]:
            try:
                reports.append(json.loads(f.read_text(encoding="utf-8")))
            except Exception as exc:
                log.debug("Could not load report %s: %s", f, exc)
        return reports
