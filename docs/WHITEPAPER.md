## Vision RAG: Technical Workflow and Operations

This paper describes the operational workflow of a production-ready Vision RAG system. It focuses on the end-to-end processing pipeline, retrieval methodology, and runtime behaviors. Architectural diagrams are intentionally omitted; we document only the technical workflow.

### Objectives
- Provide reliable QA over image-rich PDFs.
- Maintain high retrieval accuracy across mixed-quality extractions.
- Offer simple, scriptable interfaces for ingestion and querying.

### Workflow Overview
1) Input Acquisition
   - Accept PDF via multipart upload or a server-side path.
   - Cleanly drop prior index data before new ingestion (default behavior) to ensure freshness.

2) Text and Signal Extraction
   - Parse PDF pages with two independent extractors (PyMuPDF and PyPDF).
   - Merge results per page to maximize textual coverage.
   - Record image presence per page, including counts, as metadata.

3) Chunking
   - Create overlapping text windows to balance context and recall.
   - Remove empty/low-signal windows to avoid wasted embeddings.

4) Embedding Generation
   - Send batched requests to a local vLLM server.
   - If `/v1/embeddings` is not supported by the model, fall back to `/pooling` with mean pooling to obtain vectors.
   - Truncate overly long inputs and batch for throughput.

5) Indexing
   - Store vectors and metadata in Milvus (2048-d configuration).
   - Apply an index suitable for cosine similarity and load for querying.

6) Querying
   - Embed the user query through the same embeddings server.
   - Retrieve candidates from Milvus.
   - Re-rank candidates using BM25 and fuzzy matching blended with dense scores.

7) Answer Synthesis
   - Provide the top-k context to an LLM (OpenAI/Deepseek) with a conservative prompt.
   - Return answer along with hits and vision metrics.

### Vision Signal Observability
- Metadata includes `has_image` and `num_images_on_page`.
- Query responses report `images_in_hits` and `image_pages` to quantify visual-context presence among top candidates.

### Interfaces
- Ingest: `POST /api/ingest` (form `file=@...` or `?path=...`). Returns counts of chunks, pages, and total images.
- Query: `POST /api/query` (URL params or JSON body). Returns answer, hits, and image metrics.
- Reindex: `POST /api/reindex` to drop the collection.
- Health/Stats endpoints for operational checks.

### Operational Practices
- Keep embeddings server local to minimize latency.
- Reindex when changing embedding models or dimensions.
- Monitor ingestion logs for page counts and total images; monitor query logs for image metrics in hits.

### Conclusion
The described workflow ensures robust handling of real-world PDFs, efficient retrieval via hybrid scoring, and transparent visibility into when and how visual context contributes to results.


