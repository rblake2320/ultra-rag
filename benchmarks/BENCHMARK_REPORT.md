# Ultra RAG Evidence Benchmark

This report contains measured results only. Unconfigured products are not ranked, scored, or treated as beaten.

Measured targets: 1
Unmeasured targets: 4
Overall gate: PASS

| Target | Class | Status | Evidence |
|---|---|---|---|
| Ultra RAG retrieval fixture | local-first RAG app | passed | {"cases": [{"mrr": 1.0, "ndcg_at_k": 1.0, "passed": true, "precision_at_k": 1.0, "query": "how do I build the deterministic code graph", "recall_at_k": 1.0, "relevant_ids": ["code_graph_cli", "code_graph_module"], "retrieved_ids": ["code_graph_cli", "code_graph_module"]}, {"mrr": 1.0, "ndcg_at_k": 1.0, "passed": true, "precision_at_k": 1.0, "query": "what protects optional service configuration", "recall_at_k": 1.0, "relevant_ids": ["config_optional_env"], "retrieved_ids": ["config_optional_env"]}, {"mrr": 1.0, "ndcg_at_k": 1.0, "passed": true, "precision_at_k": 1.0, "query": "how is retrieval quality measured", "recall_at_k": 1.0, "relevant_ids": ["retrieval_eval_metrics"], "retrieved_ids": ["retrieval_eval_metrics"]}], "mrr": 1.0, "n_cases": 3, "ndcg_at_k": 1.0, "passed": true, "precision_at_k": 1.0, "recall_at_k": 1.0} |
| AnythingLLM | self-hosted document chat app | not_configured | Set enabled=true and endpoint to a real AnythingLLM workspace chat API before making comparison claims. |
| Dify | AI app / knowledge-base platform | not_configured | Set enabled=true and endpoint to a real Dify app API backed by the same corpus before making comparison claims. |
| RAGFlow | document RAG product | not_configured | Set enabled=true and endpoint to a real RAGFlow chat API backed by the same corpus before making comparison claims. |
| Open WebUI | self-hosted chat UI with knowledge/RAG | not_configured | Set enabled=true and endpoint to a real Open WebUI pipeline/API backed by the same corpus before making comparison claims. |

## Required Comparison Areas

- workspace UI loads
- document upload is supported
- source-cited chat is supported
- retrieval quality is measured from golden queries
- latency and query traces are observable
- deployment config validates
- competitor claims require measured evidence

## How To Add Real Competitor Evidence

Configure a target in `benchmarks/targets.json` with either a command that returns JSON metrics or an HTTP chat endpoint.
Until that target runs successfully, it remains `not_configured` and no superiority claim is made.
