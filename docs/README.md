## Vision RAG (LangChain + Milvus + Jina Embeddings)

### Prerequisites
- Python 3.10+
- Milvus running at `localhost:19530`
- Embedding server exposing `jinaai/jina-embeddings-v4-vllm-code` at `http://localhost:8000/v1/embeddings`
- OpenAI or Deepseek API key in `.env`

### .env example
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

MILVUS_URI=http://localhost:19530
MILVUS_COLLECTION=vision_rag_docs
```

### Install
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Run API
```
./scripts/run.sh
```

OpenAPI docs: http://localhost:8080/docs

### Ingest the provided PDF
Use the `/api/ingest` endpoint (defaults to `biology_textbook.pdf` in project root):
```
curl -X POST http://localhost:8080/api/ingest
```

### Query
```
curl -X POST "http://localhost:8080/api/query?query=What is photosynthesis?&k=6"
```

### Reindex (drop collection)
```
curl -X POST http://localhost:8080/api/reindex
```


