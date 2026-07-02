# Ultra RAG Agent Instructions

Use these rules when working in this repository.

## First Action

- For codebase or architecture questions, check whether `graph-out/default/graph.json` exists.
- If it exists, query or inspect the code graph before broad file reads.
- If it does not exist and the user is asking about code structure, build it with:

```bash
python ultra_code_graph.py default --no-db --json
```

- For document retrieval or answer-quality questions, inspect the RAG pipeline first: `src/search.py`, `ultra_query.py`, `src/kg_graph.py`, and `src/provenance.py`.

## Core Workflows

- Build deterministic code graph artifacts:

```bash
python ultra_code_graph.py default --no-db --json
```

- Run the full non-LLM-safe ingestion subset:

```bash
python ultra_ingest.py default --stages all --no-llm
```

- Run tests:

```bash
pytest -q
```

- Run syntax verification:

```bash
python -m compileall src ingest.py query.py ultra_ingest.py ultra_query.py ultra_eval.py ultra_server.py ultra_code_graph.py
```

- Run the full local readiness gate:

```bash
python ultra_check.py
```

## Engineering Rules

- Keep deterministic extraction separate from LLM extraction.
- Preserve confidence labels on graph edges: `EXTRACTED`, `INFERRED`, `AMBIGUOUS`.
- Keep generated artifacts out of git unless they are deliberate examples. `graph-out/`, `documents/`, and `logs/` are generated.
- Prefer focused tests with local fixtures. Do not require Postgres, Ollama, NVIDIA, or external network for unit tests.
- If modifying config behavior, add or update tests in `tests/test_config.py`.
- If modifying code graph behavior, add or update tests in `tests/test_code_graph.py`.
- If modifying retrieval behavior, update `eval/golden_retrieval.jsonl` and tests in `tests/test_retrieval_eval.py`.
- A change is not green until `python ultra_check.py` passes, or `python ultra_check.py --no-docker` passes when Docker is unavailable.

## Current Architecture

- `src/code_graph.py`: deterministic Python/SQL graph extraction and artifact export.
- `ultra_code_graph.py`: direct CLI for portable graph artifacts.
- `ultra_ingest.py`: full ingestion, including the non-LLM `code_graph` stage.
- `src/search.py`: hybrid keyword/vector retrieval.
- `ultra_query.py`: strategy routing, reranking, CRAG/Self-RAG hooks, provenance.
- `src/db_ultra.py`: extended KG, provenance, eval, parent chunk, and utility schema.
