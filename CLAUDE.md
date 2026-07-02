# Ultra RAG Claude Code Guidance

Before reading many files, use the project graph when available.

## Query-First Rule

If `graph-out/default/graph.json` exists and the user asks about code structure, architecture, dependencies, or data flow, inspect the graph artifacts first:

- `graph-out/default/GRAPH_REPORT.md`
- `graph-out/default/graph.json`
- `graph-out/default/graph.html`

If the graph is missing, build it:

```bash
python ultra_code_graph.py default --no-db --json
```

Then answer using graph evidence plus targeted file reads.

## Build And Verify

Use these commands after changes:

```bash
pytest -q
python -m compileall src ingest.py query.py ultra_ingest.py ultra_query.py ultra_eval.py ultra_server.py ultra_code_graph.py
```

For deployment/config changes, also run:

```bash
docker compose config
```

For the full local readiness gate, run:

```bash
python ultra_check.py
```

## Do Not Break

- Unit tests must run without Postgres, Ollama, NVIDIA, or external API keys.
- `code_graph` must stay available under `--no-llm`.
- Optional env vars must use `${VAR:-default}` style in `config.yaml`.
- Generated files under `graph-out/`, `documents/`, and `logs/` should not be committed by default.
- Retrieval changes should keep `python ultra_retrieval_eval.py --golden eval/golden_retrieval.jsonl --fixture-results eval/fixture_results.json` green.
