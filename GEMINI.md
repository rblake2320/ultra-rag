# Ultra RAG Gemini Agent Guidance

Use the deterministic graph before broad context loading.

1. If `graph-out/default/graph.json` exists, inspect it or `graph-out/default/GRAPH_REPORT.md` first for codebase questions.
2. If it is missing, run:

```bash
python ultra_code_graph.py default --no-db --json
```

3. Use targeted file reads after graph inspection.
4. Verify changes with:

```bash
pytest -q
python -m compileall src ingest.py query.py ultra_ingest.py ultra_query.py ultra_eval.py ultra_server.py ultra_code_graph.py
```

Keep deterministic graph extraction, LLM KG extraction, and vector retrieval as distinct layers.

Before marking work complete, prefer:

```bash
python ultra_check.py
```

If Docker is unavailable, use `python ultra_check.py --no-docker` and state that Docker Compose syntax was skipped.
