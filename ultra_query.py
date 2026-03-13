#!/usr/bin/env python3
"""
Ultra RAG Query: intelligent search with auto-routing and provenance.

Routes each query to the optimal retrieval strategy using LLM classification,
then applies reranking, CRAG quality evaluation, Self-RAG adaptive filtering,
and utility-score boosting before returning results.

Usage:
  python ultra_query.py "what are the key findings in chapter 3?" --collection my-docs
  python ultra_query.py "how does policy A relate to policy B?" --collection my-docs --strategy kg_local
  python ultra_query.py "what are the main themes across all documents?" --collection my-docs --strategy kg_global
  python ultra_query.py "explain the approval workflow" --collection my-docs --hyde
  python ultra_query.py "compliance AND reporting requirements" --collection my-docs --strategy compound
  python ultra_query.py "quarterly results" --collection my-docs --json
  python ultra_query.py "risk factors" --collection my-docs --provenance
"""
import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── Logging ───────────────────────────────────────────────────────────────────
logs_dir = Path(__file__).parent / "logs"
logs_dir.mkdir(exist_ok=True)
log_file = logs_dir / f"ultra_query_{datetime.now():%Y%m%d}.log"

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.WARNING,  # keep query output clean; set INFO for debugging
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── Project root ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

# ---------------------------------------------------------------------------
# Graceful module loader
# ---------------------------------------------------------------------------

def _try_import(module_path: str, class_name: str):
    """Import a class from a dotted module path; return None on ImportError."""
    try:
        import importlib  # noqa: PLC0415
        mod = importlib.import_module(module_path)
        return getattr(mod, class_name, None)
    except (ImportError, ModuleNotFoundError) as exc:
        log.debug("Could not import %s.%s: %s", module_path, class_name, exc)
        return None


# ---------------------------------------------------------------------------
# Strategy runners
# ---------------------------------------------------------------------------

def _search_hybrid(conn, query: str, collection: str, top_k: int) -> list:
    from src.search import search  # noqa: PLC0415
    return search(conn, query, collection, top_k=top_k)


def _search_kg_local(conn, query: str, collection: str, top_k: int) -> list:
    """Entity-centric KG neighbourhood + PPR traversal."""
    KGGraph = _try_import("src.kg_graph", "KGGraph")
    if KGGraph is None:
        log.warning("src.kg_graph not available; falling back to hybrid")
        return _search_hybrid(conn, query, collection, top_k)
    try:
        from src.config import get_config      # noqa: PLC0415
        from src.embedder import _embed_batch  # noqa: PLC0415
        cfg = get_config()["embedding"]
        embs = _embed_batch([query], cfg["ollama_url"], cfg["model"])
        query_emb = embs[0] if embs else None
        kg = KGGraph(conn, collection)
        seeds = kg.get_seed_entities(query, top_k=8, query_embedding=query_emb)
        if not seeds:
            return _search_hybrid(conn, query, collection, top_k)
        seed_ids = [s["id"] for s in seeds]
        # HippoRAG 2: pass per-seed similarity scores for non-uniform personalization
        seed_scores = {s["id"]: s.get("similarity", 1.0) for s in seeds}
        # CatRAG: pass query embedding for query-aware edge re-weighting
        ppr_ids = kg.ppr(
            seed_ids,
            query_embedding=query_emb,
            seed_scores=seed_scores,
        )
        chunks = kg.get_entity_chunks([e["id"] for e in ppr_ids[:top_k * 2]])
        return chunks[:top_k]
    except Exception as exc:
        log.warning("KG local search failed: %s; falling back to hybrid", exc)
        return _search_hybrid(conn, query, collection, top_k)


def _search_kg_global(conn, query: str, collection: str, top_k: int) -> list:
    """Community summary search for thematic / global queries."""
    try:
        from src.config import get_config      # noqa: PLC0415
        from src.embedder import _embed_batch  # noqa: PLC0415
        import psycopg2.extras                  # noqa: PLC0415
        cfg = get_config()["embedding"]
        embs = _embed_batch([query], cfg["ollama_url"], cfg["model"])
        if not embs:
            return _search_hybrid(conn, query, collection, top_k)
        q_emb_str = "[" + ",".join(str(x) for x in embs[0]) + "]"
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT id, title, summary, entity_count,
                       1 - (summary_embedding <=> %s::vector) AS score
                FROM   rag.communities
                WHERE  collection = %s
                  AND  summary_embedding IS NOT NULL
                ORDER  BY summary_embedding <=> %s::vector
                LIMIT  %s
            """, (q_emb_str, collection, q_emb_str, top_k * 2))
            community_rows = cur.fetchall()
    except Exception as exc:
        log.warning("Community search failed: %s; falling back to hybrid", exc)
        return _search_hybrid(conn, query, collection, top_k)

    if not community_rows:
        return _search_hybrid(conn, query, collection, top_k)

    # Blend community summary results with top hybrid chunks for concreteness
    hybrid = _search_hybrid(conn, query, collection, top_k)

    # Convert community rows to result-like dicts
    community_results = []
    for row in community_rows:
        if row.get("summary"):
            community_results.append({
                "id":           row["id"],
                "content":      row["summary"],
                "content_type": "community_summary",
                "context_prefix": row.get("title", ""),
                "chunk_metadata": {"entity_count": row.get("entity_count", 0)},
                "score":        float(row.get("score", 0.0)),
                "token_count":  None,
                "source":       "community",
            })

    # Merge: community summaries first, then top hybrid
    merged = community_results[:max(2, top_k // 2)] + hybrid
    return merged[:top_k]


def _search_multihop(conn, query: str, collection: str, top_k: int) -> list:
    """Decompose query into sub-queries, run each, then merge."""
    QueryDecomposer = _try_import("src.query_decomposer", "QueryDecomposer")
    if QueryDecomposer is None:
        log.warning("src.query_decomposer not available; falling back to hybrid")
        return _search_hybrid(conn, query, collection, top_k)
    try:
        decomposer = QueryDecomposer(conn, collection)
        sub_queries = decomposer.decompose(query)
        if not sub_queries:
            return _search_hybrid(conn, query, collection, top_k)

        sub_results_list = []
        for sq in sub_queries:
            sub_res = _search_hybrid(conn, sq, collection, top_k)
            sub_results_list.append(sub_res)

        merged = decomposer.merge_results(sub_results_list, query)
        return merged[:top_k]
    except Exception as exc:
        log.warning("Multi-hop search failed: %s; falling back to hybrid", exc)
        return _search_hybrid(conn, query, collection, top_k)


def _search_hyde(conn, query: str, collection: str, top_k: int) -> list:
    """Hypothetical Document Embedding search."""
    HyDERetriever = _try_import("src.hyde", "HyDERetriever")
    if HyDERetriever is None:
        log.warning("src.hyde not available; falling back to hybrid")
        return _search_hybrid(conn, query, collection, top_k)
    try:
        hyde = HyDERetriever(conn, collection)
        return hyde.search(query, top_k=top_k)
    except Exception as exc:
        log.warning("HyDE search failed: %s; falling back to hybrid", exc)
        return _search_hybrid(conn, query, collection, top_k)


def _search_compound(conn, query: str, collection: str, top_k: int) -> list:
    """Decompose compound queries, route each sub-query, then merge."""
    QueryDecomposer = _try_import("src.query_decomposer", "QueryDecomposer")
    QueryRouter     = _try_import("src.query_router",     "QueryRouter")

    if QueryDecomposer is None or QueryRouter is None:
        return _search_hybrid(conn, query, collection, top_k)

    try:
        decomposer = QueryDecomposer(conn, collection)
        router     = QueryRouter()
        sub_queries = decomposer.decompose(query)
        if not sub_queries:
            return _search_hybrid(conn, query, collection, top_k)

        sub_results_list = []
        for sq in sub_queries:
            sub_strategy = router.get_strategy(sq)
            sub_res = _run_strategy(conn, sq, collection, sub_strategy, top_k)
            sub_results_list.append(sub_res)

        merged = decomposer.merge_results(sub_results_list, query)
        return merged[:top_k]
    except Exception as exc:
        log.warning("Compound search failed: %s; falling back to hybrid", exc)
        return _search_hybrid(conn, query, collection, top_k)


def _run_strategy(conn, query: str, collection: str, strategy: str, top_k: int) -> list:
    """Dispatch to the appropriate strategy runner."""
    _DISPATCH = {
        "hybrid":    _search_hybrid,
        "kg_local":  _search_kg_local,
        "kg_global": _search_kg_global,
        "multi_hop": _search_multihop,
        "hyde":      _search_hyde,
        "compound":  _search_compound,
    }
    runner = _DISPATCH.get(strategy, _search_hybrid)
    return runner(conn, query, collection, top_k)


# ---------------------------------------------------------------------------
# Provenance builder
# ---------------------------------------------------------------------------

def _build_provenance_components(result: dict, rank: int) -> dict:
    """Extract score components from a result dict for provenance display."""
    return {
        "rank":         rank,
        "score":        result.get("score", 0.0),
        "rerank_score": result.get("rerank_score"),
        "utility_score": result.get("utility_score"),
        "source":       result.get("source", "hybrid"),
        "content_type": result.get("content_type", "text"),
    }


def _log_provenance_chain(conn, query_log_id: int, results: list) -> int:
    """
    Insert a provenance_chain + provenance_steps for the query.
    Returns chain id, or 0 on failure.
    """
    try:
        overall_confidence = (
            sum(r.get("score", 0.0) for r in results) / len(results)
            if results else 0.0
        )
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO rag.provenance_chains (query_log_id, overall_confidence)
                VALUES (%s, %s) RETURNING id
            """, (query_log_id, overall_confidence))
            chain_id = cur.fetchone()[0]

            for rank, result in enumerate(results, 1):
                chunk_id = result.get("id") if result.get("content_type") != "community_summary" else None
                score_components = _build_provenance_components(result, rank)
                cur.execute("""
                    INSERT INTO rag.provenance_steps
                        (chain_id, chunk_id, score_components, rank_position)
                    VALUES (%s, %s, %s, %s)
                """, (
                    chain_id,
                    chunk_id,
                    json.dumps(score_components),
                    rank,
                ))
        conn.commit()
        return chain_id
    except Exception as exc:
        log.warning("Provenance logging failed: %s", exc)
        try:
            conn.rollback()
        except Exception:
            pass
        return 0


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_ultra_query(
    conn,
    query: str,
    collection: str,
    top_k: int = 5,
    strategy_override: Optional[str] = None,
    no_rerank: bool = False,
    no_crag: bool = False,
) -> dict:
    """
    Execute the full Ultra RAG query pipeline.

    Returns
    -------
    dict with keys:
        results, strategy_used, quality_score, latency_ms, provenance_chain_id
    """
    from typing import Optional  # noqa: PLC0415 — already imported at module top
    t_start = time.time()

    # ── 1. Route ──────────────────────────────────────────────────────
    if strategy_override:
        strategy = strategy_override
        classification = {"query_type": "override", "confidence": 1.0, "reasoning": "manual"}
    else:
        try:
            from src.query_router import QueryRouter  # noqa: PLC0415
            router = QueryRouter()
            classification = router.classify(query)
            strategy = classification["strategy"]
        except Exception as exc:
            log.warning("QueryRouter failed: %s; defaulting to hybrid", exc)
            strategy = "hybrid"
            classification = {"query_type": "unknown", "confidence": 0.5, "reasoning": str(exc)}

    log.info("Strategy: %s  (type=%s, conf=%.2f)",
             strategy, classification.get("query_type"), classification.get("confidence", 0))

    # ── 2. Retrieve (strategy-specific) ───────────────────────────────
    try:
        results = _run_strategy(conn, query, collection, strategy, top_k * 3)
    except Exception as exc:
        log.error("Retrieval failed: %s", exc)
        results = []

    # ── 3. Record retrieval in memory ─────────────────────────────────
    chunk_ids = [r["id"] for r in results if r.get("id")]
    RetrievalMemory = _try_import("src.retrieval_memory", "RetrievalMemory")
    memory = None
    if RetrievalMemory:
        try:
            memory = RetrievalMemory(conn, collection)
            memory.record_retrieval(chunk_ids)
        except Exception as exc:
            log.debug("RetrievalMemory.record_retrieval failed: %s", exc)
            memory = None

    # ── 4. Rerank ─────────────────────────────────────────────────────
    if not no_rerank and results:
        Reranker = _try_import("src.reranker", "Reranker")
        if Reranker:
            try:
                reranker = Reranker()
                results = reranker.rerank(query, results, top_k=top_k * 2)
            except Exception as exc:
                log.debug("Reranker failed: %s; skipping", exc)
                results = results[:top_k * 2]
        else:
            results = results[:top_k * 2]
    else:
        results = results[:top_k * 2]

    # ── 5. CRAG quality check ─────────────────────────────────────────
    quality_score = 0.75  # optimistic default
    corrective_action = "none"
    if not no_crag and results:
        CRAGEvaluator = _try_import("src.corrective", "CRAGEvaluator")
        if CRAGEvaluator:
            try:
                def _base_search_fn(q, top_k_inner=5):
                    return _search_hybrid(conn, q, collection, top_k_inner)

                crag = CRAGEvaluator(conn, collection)
                crag_result = crag.corrective_pipeline(query, results, _base_search_fn)
                results = crag_result.get("results", results)
                quality_score = crag_result.get("quality_score", quality_score)
                corrective_action = crag_result.get("action", "none")
            except Exception as exc:
                log.debug("CRAG failed: %s; skipping", exc)

    # ── 6. Self-RAG adaptive filtering ────────────────────────────────
    SelfRAG = _try_import("src.self_rag", "SelfRAG")
    if SelfRAG and results:
        try:
            def _self_search_fn(q, top_k_inner=5):
                return _search_hybrid(conn, q, collection, top_k_inner)

            self_rag = SelfRAG()
            results = self_rag.adaptive_retrieve_and_filter(query, _self_search_fn, results)
        except Exception as exc:
            log.debug("SelfRAG failed: %s; skipping", exc)

    # ── 7. Utility score boost ────────────────────────────────────────
    if memory and results:
        try:
            results = memory.apply_utility_boost(results)
        except Exception as exc:
            log.debug("utility_boost failed: %s", exc)

    # Truncate to final top_k
    results = results[:top_k]

    # ── 8. Parent chunk expansion ─────────────────────────────────────
    ParentChunker = _try_import("src.parent_chunker", "ParentChunker")
    if ParentChunker:
        try:
            pc = ParentChunker(conn, collection)
            for r in results:
                cid = r.get("id")
                if cid:
                    parent_content = pc.get_parent_content(cid)
                    if parent_content:
                        r["parent_content"] = parent_content
        except Exception as exc:
            log.debug("Parent content expansion failed: %s", exc)

    # Attach provenance components to each result
    for rank, r in enumerate(results, 1):
        r["provenance_components"] = _build_provenance_components(r, rank)

    # ── 9. Log query ──────────────────────────────────────────────────
    latency_ms = int((time.time() - t_start) * 1000)
    query_log_id = 0
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO rag.query_log
                    (query_text, collection, strategy, quality_score,
                     latency_ms, corrective_action, result_count)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                query, collection, strategy,
                round(quality_score, 4),
                latency_ms,
                corrective_action,
                len(results),
            ))
            query_log_id = cur.fetchone()[0]
        conn.commit()
    except Exception as exc:
        log.warning("query_log insert failed: %s", exc)
        try:
            conn.rollback()
        except Exception:
            pass

    # ── 10. Provenance chain ──────────────────────────────────────────
    provenance_chain_id = 0
    if query_log_id and results:
        provenance_chain_id = _log_provenance_chain(conn, query_log_id, results)

    return {
        "results":              results,
        "strategy_used":        strategy,
        "quality_score":        quality_score,
        "corrective_action":    corrective_action,
        "latency_ms":           latency_ms,
        "query_log_id":         query_log_id,
        "provenance_chain_id":  provenance_chain_id,
        "classification":       classification,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_results(query_result: dict, show_provenance: bool = False) -> None:
    """Pretty-print results to stdout."""
    results       = query_result["results"]
    strategy      = query_result["strategy_used"]
    quality       = query_result["quality_score"]
    latency       = query_result["latency_ms"]
    classification = query_result.get("classification", {})

    print(f"\n{'='*72}")
    print(f"Query:     {_truncate(query_result.get('query', ''), 70)}")
    print(f"Strategy:  {strategy}  "
          f"(type={classification.get('query_type', '?')}, "
          f"conf={classification.get('confidence', 0):.2f})")
    print(f"Quality:   {quality:.3f}    Latency: {latency}ms    "
          f"Results: {len(results)}")
    print(f"{'='*72}")

    if not results:
        print("No results found.")
        return

    for i, r in enumerate(results, 1):
        ctype   = r.get("content_type", "text")
        score   = r.get("score", 0.0)
        tok     = r.get("token_count") or "?"
        ctx     = r.get("context_prefix", "") or ""
        meta    = r.get("chunk_metadata", {}) or {}
        sections = meta.get("sections", []) or meta.get("tags", [])
        rerank  = r.get("rerank_score")
        utility = r.get("utility_score")

        score_line = f"score={score:.5f}"
        if rerank is not None:
            score_line += f"  rerank={rerank:.5f}"
        if utility is not None:
            score_line += f"  utility={utility:.4f}"

        print(f"\n[{i}] {ctype.upper():<22} {score_line}  {tok}tok")
        if ctx:
            print(f"    Context: {_truncate(ctx, 80)}")
        if sections:
            print(f"    Sections: {', '.join(sections)}")

        content = r.get("content", "")
        for line in _truncate(content, 600).split("\n"):
            print(f"    {line}")

        if r.get("parent_content") and len(r["parent_content"]) > len(content):
            print(f"    [Parent: {_truncate(r['parent_content'], 200)}]")

        if show_provenance and r.get("provenance_components"):
            pc = r["provenance_components"]
            print(f"    Provenance: {pc}")

    print(f"\n{'='*72}")
    if show_provenance and query_result.get("provenance_chain_id"):
        print(f"Provenance chain id: {query_result['provenance_chain_id']}")


def _truncate(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n] + "…"


def main():
    parser = argparse.ArgumentParser(
        description="Ultra RAG Query — intelligent search with auto-routing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("query",      help="Search query")
    parser.add_argument("--collection", default="my-docs", help="Collection name")
    parser.add_argument("--top", "-k", type=int, default=5, help="Top results (default: 5)")
    parser.add_argument(
        "--strategy",
        choices=["hybrid", "kg_local", "kg_global", "multi_hop", "hyde", "compound"],
        help="Override auto-routing strategy",
    )
    parser.add_argument("--hyde",      action="store_true", help="Force HyDE strategy")
    parser.add_argument("--no-rerank", action="store_true", help="Skip reranking")
    parser.add_argument("--no-crag",   action="store_true", help="Skip CRAG quality check")
    parser.add_argument("--json",      action="store_true", help="Output raw JSON")
    parser.add_argument("--provenance",action="store_true", help="Show full provenance chain")
    args = parser.parse_args()

    strategy = args.strategy
    if args.hyde:
        strategy = "hyde"

    from src.db      import get_conn       # noqa: PLC0415
    from src.db_ultra import create_ultra_schema  # noqa: PLC0415

    conn = get_conn()
    try:
        create_ultra_schema(conn)

        result = run_ultra_query(
            conn,
            query=args.query,
            collection=args.collection,
            top_k=args.top,
            strategy_override=strategy,
            no_rerank=args.no_rerank,
            no_crag=args.no_crag,
        )
        result["query"] = args.query  # attach for display

        if args.json:
            out = {"results": [], "strategy_used": result["strategy_used"],
                   "quality_score": result["quality_score"],
                   "latency_ms": result["latency_ms"]}
            for r in result["results"]:
                d = {k: v for k, v in r.items() if k != "embedding"}
                out["results"].append(d)
            print(json.dumps(out, indent=2, default=str))
        else:
            _print_results(result, show_provenance=args.provenance)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
