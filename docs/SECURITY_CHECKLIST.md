# Security Checklist

Ultra RAG follows the current OWASP API Security framing for API risk review.

## Required Controls

- Authentication: set `ULTRARAG_API_KEY` outside local development.
- Authorization: do not claim multi-tenant isolation until workspace RBAC is implemented.
- Object access: any endpoint that takes a document, collection, or provenance ID must be reviewed before public deployment.
- Resource limits: upload size, file type allowlist, query limits, and timeout behavior must be enforced at the reverse proxy or API layer.
- CORS: set `ULTRARAG_CORS_ORIGINS` to explicit origins.
- Secrets: keep `.env` out of git; use a secrets manager in hosted deployments.
- SSRF: do not add arbitrary URL ingestion without allowlists and network egress controls.
- Dependency drift: install from `pyproject.toml`/`requirements.txt` and run `python ultra_check.py`.
- API inventory: `/docs` and `/openapi.json` are useful locally; review public exposure before launch.

## Current Limitations

- API key authentication is implemented.
- Full user identity, roles, and workspace RBAC are not implemented.
- Upload file type allowlist is implemented through parser suffix support.
- Upload size enforcement should be added at reverse proxy or app layer before public internet exposure.
