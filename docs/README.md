## Vision RAG

Production-ready Vision RAG system using LangChain, Milvus, and a local vLLM server for embeddings. Supports PDF text+image-aware ingestion, hybrid retrieval (dense + BM25 + fuzzy), and LLM QA with OpenAI or DeepSeek.

### Key Features
- Ingestion of PDFs with merged text extraction (PyMuPDF + PyPDF) and page-level image awareness.
- Embeddings via local vLLM server. Automatic fallback to `/pooling` when the model does not support `/v1/embeddings`.
- Milvus vector store (2048 dimensions) with automatic collection (re)creation and drop-on-ingest by default.
- Hybrid retrieval with dense + BM25 + fuzzy re-ranking for accuracy.
- FastAPI endpoints for health, ingest, query, reindex, and stats.

### Prerequisites
- Python 3.10+
- Milvus at `localhost:19530` (standalone OK)
- vLLM serving `jinaai/jina-embeddings-v4-vllm-code` at `http://localhost:8000`
- OpenAI or Deepseek API key in `.env`

### .env Example
```
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
# OPENAI_BASE_URL=https://api.openai.com/v1

# Or Deepseek
# LLM_PROVIDER=deepseek
# DEEPSEEK_API_KEY=...
# DEEPSEEK_BASE_URL=https://api.deepseek.com

EMBEDDINGS_BASE_URL=http://localhost:8000
EMBEDDINGS_MODEL=jinaai/jina-embeddings-v4-vllm-code

MILVUS_URI=localhost:19530
MILVUS_COLLECTION=vision_rag_docs
```

### Install
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Start Dependencies
- Milvus (Docker):
```
docker run -d --name milvus-standalone \
  -p 19530:19530 -p 9091:9091 \
  -e ETCD_USE_EMBED=true \
  -e MINIO_USE_EMBED=true \
  -e PULSAR_USE_EMBED=true \
  milvusdb/milvus:v2.5.5
```
- vLLM:
```
vllm serve jinaai/jina-embeddings-v4-vllm-code \
  --no-enable-chunked-prefill \
  --dtype float16 \
  --gpu-memory-utilization 0.8
```

### Run API
```
./scripts/run.sh
```
OpenAPI docs: http://localhost:8080/docs

### Ingest PDF (clears previous index)
```
curl -s -F file=@./biology_textbook.pdf http://localhost:8080/api/ingest
# or by path
curl -s -X POST "http://localhost:8080/api/ingest?path=$(pwd)/biology_textbook.pdf"
```
Example response includes pages and total_images for visibility.

### Query
URL params:
```
curl -s -X POST "http://localhost:8080/api/query?query=What is photosynthesis?&k=6"
```
JSON body:
```
curl -s http://localhost:8080/api/query \
  -H "Content-Type: application/json" \
  -d '{"query":"What is photosynthesis?","k":6}'
```
Response includes `images_in_hits` and `image_pages` to confirm vision-aware context presence.

### Reindex (drop collection)
```
curl -s -X POST http://localhost:8080/api/reindex
```

### Troubleshooting
- 503 embeddings: ensure vLLM is running at `EMBEDDINGS_BASE_URL`. If `/v1/embeddings` isn’t supported, the client falls back to `/pooling` automatically.
- Milvus dim mismatch: this project uses 2048-d embeddings. If you changed models, update `MILVUS_DIM` in `app/core/config.py` and call `/api/reindex`.
- File uploads require `python-multipart` (already in requirements). If you see a form parsing error, reinstall requirements and restart.

### Endpoints Summary
- `GET /health`
- `POST /api/ingest` (form `file=@...` or `?path=...`)
- `POST /api/query` (query params or JSON body)
- `POST /api/reindex`
- `GET /api/stats`


