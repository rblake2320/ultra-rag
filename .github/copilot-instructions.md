# Ultra RAG Copilot Instructions

Ultra RAG has three distinct retrieval layers:

- deterministic code graph extraction in `src/code_graph.py`
- LLM knowledge-graph extraction in `src/kg_extractor.py`
- vector/keyword RAG retrieval in `src/search.py` and `ultra_query.py`

For codebase questions, check `graph-out/default/graph.json` or `graph-out/default/GRAPH_REPORT.md` before broad source exploration. If missing, build it with:

```bash
python ultra_code_graph.py default --no-db --json
```

Before proposing changes as complete, verify:

```bash
pytest -q
python -m compileall src ingest.py query.py ultra_ingest.py ultra_query.py ultra_eval.py ultra_server.py ultra_code_graph.py
```

Keep generated directories out of commits unless explicitly requested: `graph-out/`, `documents/`, and `logs/`.

The preferred all-local gate is:

```bash
python ultra_check.py
```
