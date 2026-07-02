from src.observability import QueryTrace, stage_timer


def test_query_trace_records_stage_timing():
    trace = QueryTrace()
    started = stage_timer()

    trace.mark("retrieve", started, result_count=3)
    payload = trace.as_dict()

    assert payload["total_ms"] >= 0
    assert payload["stages"][0]["stage"] == "retrieve"
    assert payload["stages"][0]["result_count"] == 3
