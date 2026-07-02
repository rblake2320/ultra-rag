import pytest

from src import config


def test_optional_env_reference_defaults_when_unset(monkeypatch):
    monkeypatch.delenv("ULTRARAG_TEST_OPTIONAL", raising=False)

    resolved = config._resolve({"value": "${ULTRARAG_TEST_OPTIONAL:-fallback}"})

    assert resolved["value"] == "fallback"


def test_required_env_reference_raises_when_unset(monkeypatch):
    monkeypatch.delenv("ULTRARAG_TEST_REQUIRED", raising=False)

    with pytest.raises(ValueError, match="ULTRARAG_TEST_REQUIRED"):
        config._resolve({"value": "${ULTRARAG_TEST_REQUIRED}"})


def test_config_loads_without_optional_service_keys(monkeypatch):
    monkeypatch.delenv("RAG_DB_PASSWORD", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    config.reset_config_cache()

    cfg = config.get_config()

    assert cfg["db"]["password"] == ""
    assert cfg["search"]["nvidia_api_key"] == ""
