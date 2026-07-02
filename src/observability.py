"""Lightweight query observability helpers."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class QueryTrace:
    """Collect per-stage timings and metadata for a single query."""

    started_at: float = field(default_factory=time.perf_counter)
    stages: list[dict[str, Any]] = field(default_factory=list)

    def mark(self, name: str, started_at: float, **metadata: Any) -> None:
        self.stages.append(
            {
                "stage": name,
                "latency_ms": int((time.perf_counter() - started_at) * 1000),
                **metadata,
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_ms": int((time.perf_counter() - self.started_at) * 1000),
            "stages": self.stages,
        }


def stage_timer() -> float:
    return time.perf_counter()


def recent_query_summary(conn, collection: str = "default", limit: int = 50) -> dict[str, Any]:
    """Return compact query-log telemetry for dashboards."""
    import psycopg2.extras

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT id, query_text, collection, strategy, quality_score,
                   latency_ms, corrective_action, result_count, created_at
            FROM rag.query_log
            WHERE collection = %s
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (collection, limit),
        )
        rows = [dict(r) for r in cur.fetchall()]

    latencies = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]
    qualities = [float(r["quality_score"]) for r in rows if r.get("quality_score") is not None]
    result_counts = [r["result_count"] for r in rows if r.get("result_count") is not None]

    return {
        "collection": collection,
        "count": len(rows),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0,
        "avg_quality_score": round(sum(qualities) / len(qualities), 4) if qualities else 0,
        "zero_result_count": sum(1 for value in result_counts if value == 0),
        "recent": rows,
    }
