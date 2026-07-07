import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.mark.asyncio
async def test_startup_event_degrades_when_db_unavailable(monkeypatch):
    import ultra_server

    def raise_db_unavailable():
        raise RuntimeError("db unavailable")

    monkeypatch.setattr(ultra_server, "_get_conn", raise_db_unavailable)

    await ultra_server.startup_event()


@pytest.mark.asyncio
async def test_health_skips_pgvector_when_db_unavailable(monkeypatch):
    import ultra_server
    import src.config
    import httpx

    def raise_db_unavailable():
        raise RuntimeError("db unavailable")

    mock_ollama_response = MagicMock()
    mock_ollama_response.raise_for_status = MagicMock()

    monkeypatch.setattr(ultra_server, "_get_conn", raise_db_unavailable)
    monkeypatch.setattr(src.config, "get_config", lambda: {"llm": {"ollama_url": "http://ollama.test"}})
    monkeypatch.setattr(httpx, "get", lambda *args, **kwargs: mock_ollama_response)

    response = await ultra_server.health()

    assert response.status == "degraded"
    assert response.db.startswith("error:")
    assert response.pgvector == "skipped: db unavailable"
    assert response.ollama == "ok"
