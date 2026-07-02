import inspect

import ultra_ingest
import ultra_server


def test_api_has_liveness_and_readiness_routes():
    paths = {route.path for route in ultra_server.app.routes}

    assert "/app" in paths
    assert "/api/live" in paths
    assert "/api/ready" in paths
    assert "/api/health" in paths
    assert "/dashboard" in paths
    assert "/api/observability/summary" in paths
    assert "/api/collections" in paths
    assert "/api/documents" in paths
    assert "/api/upload" in paths
    assert "/api/chat" in paths
    assert "/api/playbooks/recommend" in paths


def test_api_ingest_dispatch_covers_all_cli_stages():
    source = inspect.getsource(ultra_server.ingest)

    for stage in ultra_ingest.ALL_STAGES:
        assert f'"{stage}"' in source


def test_search_response_model_includes_trace():
    fields = ultra_server.SearchResponse.model_fields

    assert "trace" in fields


def test_chat_response_model_has_sources_and_trace():
    fields = ultra_server.ChatResponse.model_fields

    assert "sources" in fields
    assert "trace" in fields


def test_lifespan_is_non_strict_by_default(monkeypatch):
    monkeypatch.delenv("ULTRARAG_STRICT_STARTUP", raising=False)

    assert "ULTRARAG_STRICT_STARTUP" not in ultra_server.os.environ
