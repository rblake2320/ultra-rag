# Ultra RAG System
World's most comprehensive RAG system combining LightRAG + HippoRAG + GraphRAG + RAPTOR + NVIDIA Blueprint + Anthropic Contextual Retrieval + HyDE + Self-RAG + CRAG.

## Stack
- PostgreSQL 16 + pgvector (HNSW indexes)
- Ollama (nomic-embed-text, deepseek-r1:32b, gemma3)
- FastAPI server on :8300
- 4,015 IMDS chunks, 768-dim embeddings

## Quick Start
```bash
python ultra_server.py   # API server :8300
python ultra_query.py "your question" --collection imds
```
