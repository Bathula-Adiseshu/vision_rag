## Building a Vision-First RAG with LangChain, Milvus and vLLM

Modern RAG needs more than text. Many PDFs are vision-heavy: figures, diagrams, screenshots. This project delivers an end-to-end Vision RAG with a clean operational workflow and pragmatic engineering choices that work on real documents.

### What We Optimized For
- Robust ingestion from messy PDFs (dual extractors: PyMuPDF + PyPDF)
- Accurate retrieval with hybrid signals (dense + BM25 + fuzzy)
- Fast local embeddings using vLLM (fallback to `/pooling` when needed)
- Low-friction operability: one FastAPI app with clear endpoints

### Technical Workflow (High-Level)
1) PDF arrives (upload or path). We parse each page and merge two text extractors, maximizing signal. We also detect images per page and carry that as metadata.
2) Text is chunked with overlap. Empty/low-signal chunks are dropped.
3) Embeddings are generated via a local vLLM server. If the served model doesn’t support `/v1/embeddings`, we transparently call `/pooling`.
4) Chunks are written to Milvus with JSON metadata (page, source, has_image). By default we clear the collection before new ingestion to avoid stale results.
5) Queries are embedded, Milvus returns dense candidates, we re-rank with BM25 + fuzzy to blend lexical and semantic evidence.
6) The top-k context feeds an LLM (OpenAI/Deepseek), which composes the final answer.

### Why Hybrid Retrieval?
Dense vectors are great at capturing semantics; BM25 + fuzzy hedge against exact terms, acronyms, and OCR oddities. The blend improves reliability on technical PDFs and scanned docs.

### Observability of Vision Signal
- We persist `has_image` and `num_images_on_page` into metadata.
- Query responses include `images_in_hits` and `image_pages`, so you can verify image-bearing pages influence results.

### API You Can Script
- Ingest: `POST /api/ingest` (form upload or `?path=`)
- Query: `POST /api/query` (URL params or JSON body)
- Reindex: `POST /api/reindex`
- Health/Stats: quick status and counts

### Practical Tips
- Keep embeddings local for speed; vLLM enables fast batch processing.
- Watch the vector dimension. This setup uses 2048-d; if you change models, reindex with the correct dim.
- Log and inspect `images_in_hits` during evaluation to validate vision impact on retrieval quality.

This stack gives you a dependable baseline for production Vision RAG: fast, observable, and resilient to real-world PDFs.


