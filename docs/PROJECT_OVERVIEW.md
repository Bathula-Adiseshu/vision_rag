## Vision RAG: System Overview (Workflow + Architecture)

This document provides a full overview of the Vision RAG project: end-to-end workflow, architecture, components, data flow, configuration, deployment, and operational guidance.

---

### 1) Goals
- High-accuracy RAG over vision-heavy PDFs
- Robust ingestion (text + image awareness)
- Fast local embeddings via vLLM
- Scalable and observable retrieval using Milvus
- Simple API surface for ingestion, querying, and maintenance

---

### 2) High-Level Workflow
1. Ingestion Input: a PDF is supplied via multipart upload or a server-side path.
2. Extraction: each page is parsed with two independent text engines (PyMuPDF + PyPDF). Results are merged to maximize coverage. Page-level image presence is recorded in metadata.
3. Chunking: text is windowed with overlap; empty/low-signal chunks are dropped.
4. Embeddings: batched requests are sent to a local vLLM server. If the served model lacks `/v1/embeddings`, the client transparently falls back to `/pooling` with mean pooling.
5. Indexing: vectors + metadata are inserted into Milvus. Collection is dropped/recreated on ingest by default to ensure freshness.
6. Query: the user query is embedded; dense candidates are retrieved from Milvus; a hybrid reranker blends dense score with BM25 + fuzzy for relevance.
7. Answer: the top-k context is given to an LLM (OpenAI/Deepseek) to synthesize a concise answer. The response includes `images_in_hits` and `image_pages` to indicate vision context in use.

---

### 3) Architecture Overview

Components:
- FastAPI application (`app/main.py`, `app/api/routes.py`)
- Ingestion subsystem (`app/ingest/*`)
  - `pdf_loader.py`: merged extraction (PyMuPDF + PyPDF) and image bytes collection
  - `chunking.py`: overlapping sliding windows
- Embedding client (`app/core/embeddings.py`)
  - Async http client to vLLM at `EMBEDDINGS_BASE_URL`
  - Batch + truncate payloads; fall back to `/pooling` when needed
- Vector store (`app/vectorstores/milvus_store.py`)
  - Collection schema: `id` (string), `vector` (float vector, 2048-d), `text` (varchar), `metadata` (JSON)
  - Indexing and load-on-demand
- Retrieval & QA (`app/retrievers/*`, `app/core/llms.py`)
  - HybridRanker blends dense score with BM25 + fuzzy
  - LLM QA chain composes final answer
- Configuration (`app/core/config.py`)
  - Pydantic settings via `.env` (LLM provider, Milvus URI, embeddings host, dims, etc.)
- Logging (`app/utils/logging.py`)
  - Loguru-based; key ingestion/query metrics logged

Data Flow:
- PDF -> Extract (text merges + image flags) -> Chunk -> Embed (vLLM) -> Milvus upsert
- Query -> Embed -> Milvus search -> Hybrid re-rank -> LLM answer

---

### 4) Endpoints
- `GET /health`: liveness check
- `POST /api/ingest`:
  - Accepts: multipart form `file=@...` or `?path=...`
  - Behavior: clears collection by default and re-ingests; returns `inserted`, `chunks`, `pages`, `total_images`
  - Metadata stored: `page`, `source`, `has_image`, `num_images_on_page`
- `POST /api/query`:
  - Accepts: URL params (`?query=...&k=...`) or JSON body `{query, k}`
  - Returns: `answer`, `hits` (with text + metadata), `images_in_hits`, `image_pages`
- `POST /api/reindex`: drops collection (used when changing dims/models)
- `GET /api/stats`: basic stats placeholder

---

### 5) Configuration
All sensitive/runtime configuration is environment-driven.

Key settings (`.env`):
- `LLM_PROVIDER` (openai|deepseek)
- `OPENAI_API_KEY` / `DEEPSEEK_API_KEY`
- `EMBEDDINGS_BASE_URL` (default `http://localhost:8000`)
- `EMBEDDINGS_MODEL` (default `jinaai/jina-embeddings-v4-vllm-code`)
- `MILVUS_URI` (e.g., `localhost:19530`)
- `MILVUS_COLLECTION` (default `vision_rag_docs`)
- `MILVUS_DIM` (default `2048`)
- Retrieval tuning: `HYBRID_ALPHA`, BM25 params

---

### 6) Deployment
- Local:
  - Start Milvus (Docker standalone)
  - Start vLLM embeddings server
  - `./scripts/run.sh` to launch FastAPI
- Containerization:
  - Package the FastAPI app; link to Milvus and vLLM via env vars
  - Consider a Compose file to orchestrate app + Milvus + embeddings

Scaling:
- App: increase Uvicorn workers; horizontal autoscale behind a load balancer
- Milvus: move from standalone to distributed cluster for higher QPS and bigger corpora
- Embeddings: scale vLLM replicas (GPU) behind a reverse proxy; batch requests

---

### 7) Observability & Logging
- Ingestion logs: total pages, `total_images`, and chunk counts
- Query logs: `images_in_hits` and `image_pages` in the result set
- Add metrics via Prometheus and tracing via OpenTelemetry (optional)

---

### 8) Failure Modes & Remedies
- Embeddings 400: server lacks `/v1/embeddings`; client falls back to `/pooling`
- Milvus unreachable: check `MILVUS_URI`; ensure container is healthy and port is open
- Dim mismatch: update `MILVUS_DIM` to match the embedding model; call `/api/reindex`
- Empty content: if merged extraction yields no text, respond with 400 and guidance

---

### 9) Security & Hardening
- Keep API keys in `.env` or a secret manager
- Restrict CORS to trusted origins in production
- Use HTTPS for external traffic; isolate Milvus and vLLM on a private network
- Consider auth on write endpoints (`/api/ingest`, `/api/reindex`)

---

### 10) Evaluation Tips
- Validate retrieval quality with `images_in_hits` and `image_pages`
- Run spot queries covering text references to diagrams/figures
- Compare dense-only vs. hybrid (adjust `HYBRID_ALPHA`)

---

### 11) Roadmap Ideas
- Vision captioning: auto-generate captions for images where text is missing
- Rerankers: add cross-encoders or specialized rerank APIs
- Async pipelines: background ingestion jobs + progress reporting
- Caching: query response and embedding cache

---

### 12) Quick Commands
Ingest (clears previous index):
```
curl -s -F file=@./biology_textbook.pdf http://localhost:8080/api/ingest
```
Query:
```
curl -s -X POST "http://localhost:8080/api/query?query=What is photosynthesis?&k=6"
```
Drop index:
```
curl -s -X POST http://localhost:8080/api/reindex
```


