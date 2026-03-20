#!/usr/bin/env python3
"""
Ultra RAG Ingest: full 8-stage pipeline.

Stages:
  1. parse        — Parse + chunk source documents (uses existing rag_ingest logic)
  2. embed        — Generate embeddings for all un-embedded chunks
  3. contextual   — LLM-generated situating context per chunk (Anthropic-style)
  4. parents      — Build parent chunks (small → large two-tier hierarchy)
  5. kg           — Extract entities and relationships into knowledge graph
  6. communities  — Leiden community detection + LLM summaries
  7. raptor       — RAPTOR hierarchical summary tree
  8. colbert      — Build ColBERT PLAID index for late-interaction retrieval

Usage:
  python ultra_ingest.py my-docs --stages all
  python ultra_ingest.py my-docs --stages kg,raptor
  python ultra_ingest.py my-docs --stages contextual --no-llm
  python ultra_ingest.py my-docs --stages parse,embed --batch-size 64
  python ultra_ingest.py my-docs --stages colbert
"""
import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Logging ──────────────────────────────────────────────────────────────────
logs_dir = Path(__file__).parent / "logs"
logs_dir.mkdir(exist_ok=True)
log_file = logs_dir / f"ultra_ingest_{datetime.now():%Y%m%d}.log"

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── Project root ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

# ---------------------------------------------------------------------------
# Stage registry
# ---------------------------------------------------------------------------

ALL_STAGES = ["parse", "embed", "contextual", "parents", "kg", "communities", "raptor", "colbert"]
LLM_STAGES = {"contextual", "kg", "communities", "raptor"}


def _parse_stages(stages_arg: str) -> list:
    """Resolve comma-separated stage names or 'all' to a sorted list."""
    if stages_arg.strip().lower() == "all":
        return list(ALL_STAGES)
    parts = [s.strip().lower() for s in stages_arg.split(",")]
    unknown = [p for p in parts if p not in ALL_STAGES]
    if unknown:
        log.error("Unknown stages: %s. Valid: %s", unknown, ALL_STAGES)
        sys.exit(1)
    # preserve canonical order
    return [s for s in ALL_STAGES if s in parts]


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

def _run_parse(collection: str, conn) -> dict:
    """Stage 1: parse + chunk using existing rag_ingest logic."""
    from rag_ingest import ingest_collection  # noqa: PLC0415
    from src.db import create_schema          # noqa: PLC0415
    create_schema(conn)
    summary = ingest_collection(collection, conn)
    return summary


def _run_embed(collection: str, conn) -> dict:
    """Stage 2: embed all un-embedded chunks."""
    from src.embedder import embed_collection  # noqa: PLC0415
    n = embed_collection(conn, collection)
    return {"chunks_embedded": n}


def _run_contextual(collection: str, conn, reprocess: bool) -> dict:
    """Stage 3: generate LLM situating context for each chunk."""
    from src.contextual import ContextualRetriever  # noqa: PLC0415
    cr = ContextualRetriever(conn, collection)
    n = cr.process_collection(reprocess=reprocess)
    return {"chunks_contextualised": n}


def _run_parents(collection: str, conn) -> dict:
    """Stage 4: build parent chunk hierarchy."""
    from src.parent_chunker import ParentChunker  # noqa: PLC0415
    pc = ParentChunker(conn, collection)
    stats = pc.process_collection()
    return stats


def _run_kg(collection: str, conn) -> dict:
    """Stage 5: extract KG entities + relationships, then build synonymy edges."""
    from src.kg_extractor import KGExtractor  # noqa: PLC0415
    from src.llm import LLMClient  # noqa: PLC0415
    from src.config import get_config  # noqa: PLC0415
    # Use fast model for KG extraction — qwen2.5:7b is 3× faster than 32b
    # and produces good-quality entity/relationship extraction
    _fast = get_config().get("llm", {}).get("fast_model", "qwen2.5:7b")
    extractor = KGExtractor(conn, collection, llm_client=LLMClient(model=_fast))
    stats = extractor.process_collection()
    edge_stats = extractor.build_synonymy_edges()
    return {**stats, "synonymy_edges": edge_stats}


def _run_communities(collection: str, conn) -> dict:
    """Stage 6: Leiden community detection + LLM summaries."""
    from src.kg_communities import CommunityDetector  # noqa: PLC0415
    detector = CommunityDetector(conn, collection)
    stats = detector.process_collection()
    return stats


def _run_raptor(collection: str, conn) -> dict:
    """Stage 7: build RAPTOR hierarchical summary tree."""
    try:
        from src.raptor import RAPTOR  # noqa: PLC0415
        raptor = RAPTOR(conn, collection)
        stats = raptor.build_tree()
        return stats
    except ImportError:
        log.warning("src.raptor not found — skipping RAPTOR stage")
        return {"skipped": True, "reason": "src.raptor module not available"}


def _run_colbert(collection: str, conn) -> dict:
    """Stage 8: build ColBERT PLAID index for late-interaction retrieval."""
    try:
        from src.colbert_retriever import ColBERTRetriever  # noqa: PLC0415
        retriever = ColBERTRetriever()
        stats = retriever.build_index(conn, collection)
        return stats
    except ImportError:
        log.warning("RAGatouille not installed — skipping ColBERT stage")
        log.warning("  Install with: pip install ragatouille>=0.0.8")
        return {"skipped": True, "reason": "ragatouille not installed"}
    except Exception as exc:
        log.warning("ColBERT stage failed: %s", exc)
        return {"skipped": True, "reason": str(exc)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Ultra RAG Ingest — 8-stage ingestion pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Stages: {', '.join(ALL_STAGES)}",
    )
    parser.add_argument(
        "collection",
        help="Collection name (must exist in config.yaml)",
    )
    parser.add_argument(
        "--stages",
        default="all",
        help="Comma-separated stages or 'all' (default: all). "
             f"Options: {', '.join(ALL_STAGES)}",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help=f"Skip LLM-dependent stages: {', '.join(sorted(LLM_STAGES))}",
    )
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="Re-process already-processed chunks (contextual stage)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for embedding and LLM calls (default: 50)",
    )
    args = parser.parse_args()

    stages = _parse_stages(args.stages)
    if args.no_llm:
        skipped = [s for s in stages if s in LLM_STAGES]
        stages  = [s for s in stages if s not in LLM_STAGES]
        if skipped:
            log.info("--no-llm: skipping stages: %s", skipped)

    if not stages:
        log.info("No stages to run.")
        return

    # Warn if prerequisite stages are missing from this run
    _PREREQS = {
        "embed":       {"parse"},
        "contextual":  {"parse", "embed"},
        "parents":     {"parse"},
        "kg":          {"parse"},
        "communities": {"parse", "kg"},
        "raptor":      {"parse", "embed"},
        "colbert":     {"parse"},
    }
    stages_set = set(stages)
    for stage, required in _PREREQS.items():
        if stage in stages_set:
            missing = required - stages_set
            if missing:
                log.warning(
                    "Stage '%s' depends on %s which are NOT in this run. "
                    "Ensure they completed in a prior run or add them with --stages.",
                    stage, sorted(missing),
                )

    log.info("=" * 65)
    log.info("Ultra RAG Ingest — collection: %s", args.collection)
    log.info("Stages: %s", stages)
    log.info("=" * 65)

    # Connect + ensure schema
    from src.db      import get_conn     # noqa: PLC0415
    from src.db_ultra import create_ultra_schema  # noqa: PLC0415

    conn = get_conn()
    try:
        create_ultra_schema(conn)

        all_stage_stats: dict[str, dict] = {}
        pipeline_start = time.time()

        for stage in stages:
            log.info("")
            log.info("─── Stage: %s ─────────────────────────────────────", stage.upper())
            t0 = time.time()
            stats: dict = {}

            try:
                if stage == "parse":
                    stats = _run_parse(args.collection, conn)

                elif stage == "embed":
                    stats = _run_embed(args.collection, conn)

                elif stage == "contextual":
                    stats = _run_contextual(args.collection, conn, args.reprocess)

                elif stage == "parents":
                    stats = _run_parents(args.collection, conn)

                elif stage == "kg":
                    stats = _run_kg(args.collection, conn)

                elif stage == "communities":
                    stats = _run_communities(args.collection, conn)

                elif stage == "raptor":
                    stats = _run_raptor(args.collection, conn)

                elif stage == "colbert":
                    stats = _run_colbert(args.collection, conn)

            except Exception as exc:
                log.error("Stage '%s' failed: %s", stage, exc, exc_info=True)
                stats = {"error": str(exc)}
                try:
                    conn.rollback()
                except Exception:
                    pass

            elapsed = time.time() - t0
            stats["elapsed_seconds"] = round(elapsed, 2)
            all_stage_stats[stage] = stats

            log.info(
                "  %s complete in %.1fs  %s",
                stage,
                elapsed,
                _fmt_stats(stats),
            )

        total_elapsed = time.time() - pipeline_start
        log.info("")
        log.info("=" * 65)
        log.info("ULTRA INGEST COMPLETE — %.1fs total", total_elapsed)
        for stage, stats in all_stage_stats.items():
            log.info("  %-15s %s", stage, _fmt_stats(stats))
        log.info("=" * 65)

    except KeyboardInterrupt:
        log.info("\nInterrupted by user")
    finally:
        conn.close()


def _fmt_stats(stats: dict) -> str:
    """Format a stats dict as a compact one-liner for logging."""
    if not stats:
        return "(no stats)"
    parts = []
    for k, v in stats.items():
        if k == "elapsed_seconds":
            continue
        if isinstance(v, (int, float)):
            parts.append(f"{k}={v:,}")
        elif isinstance(v, str) and len(v) < 40:
            parts.append(f"{k}={v}")
    return "  ".join(parts) if parts else str(stats)


if __name__ == "__main__":
    main()
