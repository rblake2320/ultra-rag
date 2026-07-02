from src.product import (
    extractive_answer,
    is_supported_upload,
    safe_collection_name,
    safe_upload_name,
)


def test_safe_collection_name_normalizes_workspace_names():
    assert safe_collection_name("Sales Docs 2026!") == "sales-docs-2026"
    assert safe_collection_name("   ") == "default"


def test_safe_upload_name_strips_paths_and_unsafe_chars():
    assert safe_upload_name("../My File!.PDF") == "My-File.pdf"


def test_supported_upload_uses_parser_registry():
    assert is_supported_upload("deck.pptx") is True
    assert is_supported_upload("malware.exe") is False


def test_extractive_answer_returns_source_snippets():
    answer = extractive_answer(
        "what changed",
        [{"content": "Ultra RAG added a consumer app.", "context_prefix": "Launch"}],
    )

    assert "what changed" in answer
    assert "Ultra RAG added" in answer


def test_extractive_answer_handles_empty_results():
    answer = extractive_answer("missing", [])

    assert "did not find" in answer
