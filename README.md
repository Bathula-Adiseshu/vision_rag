# Multimodal Vision RAG System

A state-of-the-art multimodal Retrieval-Augmented Generation (RAG) system with advanced vision capabilities, dynamic model selection, and AI-enhanced image understanding. Built with LangChain, Milvus, and supporting multiple LLM and embedding providers.

## Key Features

- **Dynamic Model Selection**: Configure LLM and embedding providers via environment variables
- **AI-Enhanced Image Understanding**: Uses vision-capable LLMs for detailed image descriptions
- **True Multimodal RAG**: Images are properly embedded and included in LLM context with URLs
- **Generic Content Processing**: Smart chunking preserves structured content without hardcoded detection
- **Vision-Capable Responses**: Uses GPT-4o for image-aware query answering when enabled
- **Fallback Architecture**: OpenAI embeddings as fallback for Jina embedding service
- **Image Storage & Serving**: Local image storage with HTTP serving endpoints
- **Hybrid Search**: Advanced search combining dense vectors, sparse BM25, and multimodal content
- **Production Ready**: Comprehensive error handling, logging, and monitoring

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   PDF Input     │───▶│  PDF Processor   │───▶│  Text Chunks    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │ Image Extractor │    │ Text Embeddings │
                       └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │Image Embeddings │    │  Milvus Store   │
                       └─────────────────┘    └─────────────────┘
                                │                        │
                                └────────┬───────────────┘
                                         ▼
                                ┌─────────────────┐
                                │ Hybrid Search   │
                                └─────────────────┘
                                         │
                                         ▼
                                ┌─────────────────┐
                                │ DeepSeek LLM    │
                                └─────────────────┘
```

## Components

### Core Components

1. **MultimodalEmbeddingService**: Unified service supporting Jina and OpenAI embeddings with fallback
2. **ImageStorageService**: Local image storage with URL generation and HTTP serving
3. **MilvusVectorStore**: Advanced vector store with hybrid and multimodal search capabilities
4. **MultimodalPDFProcessor**: Smart PDF processing with AI-enhanced image descriptions
5. **MultimodalRAGPipeline**: Complete pipeline with vision-capable LLM integration
6. **Dynamic LLM Selection**: Support for DeepSeek, OpenAI, and other providers via configuration

### API Endpoints

- `POST /api/ingest` - Ingest PDF documents with multimodal processing
- `POST /api/query` - Query with support for text, image, and hybrid search
- `GET /api/status` - Comprehensive system status and metrics
- `GET /api/images/{filename}` - Serve stored images with proper URLs
- `DELETE /api/clear` - Clear all data from vector store and image storage
- `GET /api/health` - Health check with component status

## Installation

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

2. **Configure Environment**:
```bash
cp .env.example .env
# Edit .env with your API keys and service URLs
```

3. **Required Services**:
   - Milvus vector database
   - Jina embedding service
   - DeepSeek API key

## Configuration

The system supports dynamic configuration via environment variables. Copy `.env.example` to `.env` and configure:

### LLM Provider Selection
```env
# Primary LLM Provider (deepseek or openai)
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your_deepseek_api_key
OPENAI_API_KEY=your_openai_api_key
```

### Embedding Provider Selection
```env
# Embedding Provider (jina or openai)
EMBEDDING_PROVIDER=jina
JINA_EMBEDDING_URL=http://your_vm:8000/v1/embeddings
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
```

### Vision Capabilities
```env
# Enable vision features
USE_VISION_EMBEDDINGS=true
USE_VISION_LLM=true
MAX_IMAGES_IN_CONTEXT=3
```

### Image Storage
```env
# Image storage and serving
IMAGE_STORAGE_PATH=./uploads/images
IMAGE_BASE_URL=http://localhost:8000/api/images
```

### Vector Store
```env
# Milvus Configuration
MILVUS_HOST=your_vm_ip
MILVUS_PORT=19530
MILVUS_COLLECTION_NAME=vision_rag_collection
```

## Usage

### Starting the Server

```bash
python start_server.py --host 0.0.0.0 --port 8080
```

### API Usage

**Ingest a Document**:
```bash
curl -X POST "http://localhost:8080/api/ingest" \
  -F "file=@attention_all_you_need.pdf" \
  -F "request={\"clear_existing\": true}"
```

**Query the System**:
```bash
curl -X POST "http://localhost:8080/api/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the transformer architecture?",
    "search_type": "hybrid",
    "k": 5,
    "include_images": true
  }'
```

### Python Usage

```python
from app.core.rag_pipeline import rag_pipeline

# Ingest document
result = await rag_pipeline.ingest_document(
    file_path="attention_all_you_need.pdf",
    clear_existing=True
)

# Query system
response = await rag_pipeline.query(
    query="What is the transformer architecture?",
    search_type="hybrid",
    k=5
)
```

## Testing

Run comprehensive tests:

```bash
python test_multimodal_rag.py
```

This will:
- Test system status and component availability
- Ingest the attention paper
- Run various text and multimodal queries
- Test different search types
- Test edge cases and error handling
- Generate a detailed test report

## Search Types

1. **Hybrid Search**: Combines dense vector search with sparse BM25 search
2. **Similarity Search**: Dense vector search only
3. **Multimodal Search**: Searches across both text and image content

## Advanced Multimodal Capabilities

### AI-Enhanced Image Understanding
- **Vision LLM Descriptions**: Uses GPT-4o to generate detailed, context-aware image descriptions
- **Smart Content Detection**: Generic approach preserves tables, figures, equations without hardcoding
- **Image Storage & URLs**: Local storage with HTTP serving and proper URL generation
- **Multimodal Embeddings**: Support for both Jina and OpenAI multimodal embeddings

### Vision-Capable Query Processing
- **Image Context Appending**: Images included in LLM context with descriptions and URLs
- **Vision LLM Integration**: GPT-4o processes actual images alongside text for comprehensive answers
- **Fallback Architecture**: Graceful degradation to text-only responses when vision unavailable
- **Dynamic Provider Selection**: Switch between embedding and LLM providers via configuration

### Content Processing
- **Generic Chunking**: Preserves structured content (tables, figures, algorithms) intelligently
- **Enhanced Metadata**: Rich metadata including image dimensions, URLs, and AI descriptions
- **Batch Processing**: Efficient batch processing for embeddings and image analysis

## Performance Optimizations

- **Batch Processing**: Embeddings are processed in batches for efficiency
- **Image Compression**: Images are compressed to reduce storage requirements
- **Hybrid Search**: Combines multiple search strategies for better results
- **Caching**: LRU caching for configuration and embeddings
- **Async Processing**: Fully asynchronous pipeline for better performance

## Error Handling

- Comprehensive error handling throughout the pipeline
- Graceful fallbacks for failed components
- Detailed error logging and reporting
- Input validation and sanitization

## Monitoring

- System status endpoint for health monitoring
- Comprehensive logging with configurable levels
- Collection statistics and metrics
- Test suite for continuous validation

## Development

### Project Structure

```
app/
├── core/
│   ├── config.py                # Dynamic configuration with provider selection
│   ├── multimodal_embeddings.py # Unified embedding service (Jina + OpenAI)
│   ├── vectorstore.py           # Advanced Milvus with multimodal search
│   ├── pdf_processor.py         # Smart PDF processing with AI descriptions
│   ├── rag_pipeline.py          # Vision-capable RAG pipeline
│   └── llms.py                  # Multi-provider LLM support
├── api/
│   ├── models.py                # Comprehensive Pydantic models
│   └── routes.py                # FastAPI with image serving endpoint
├── ingest/
│   ├── chunking.py              # Generic smart chunking
│   └── pdf_loader.py            # Enhanced PDF loading
└── main.py                      # FastAPI application

uploads/images/                  # Local image storage
test_multimodal_rag.py          # Comprehensive test suite
start_server.py                 # Server startup script
.env.example                    # Complete configuration template
```

### System Architecture

#### Dynamic Provider Selection
The system supports multiple providers for both embeddings and LLMs:

**Embedding Providers:**
- **Jina**: Primary multimodal embeddings via local/remote service
- **OpenAI**: Fallback with `text-embedding-3-large` model

**LLM Providers:**
- **DeepSeek**: Primary provider with `deepseek-chat` model
- **OpenAI**: Fallback and vision-capable with `gpt-4o` model

#### Vision Pipeline
1. **Image Extraction**: Extract images from PDF pages
2. **AI Description**: Generate detailed descriptions using vision LLM
3. **Image Storage**: Save locally with URL generation
4. **Embedding**: Create embeddings from descriptions
5. **Vector Storage**: Store in Milvus with rich metadata
6. **Query Processing**: Include images in LLM context for answers

### Adding New Features

1. **New LLM Provider**: Add to `llms.py` and update `config.py`
2. **New Embedding Provider**: Extend `multimodal_embeddings.py`
3. **New Search Type**: Add to `vectorstore.py` hybrid search methods
4. **New Document Type**: Extend processors in `pdf_processor.py`
5. **New API Endpoint**: Add to `routes.py` with image serving support

## Troubleshooting

### Common Issues

1. **Embeddings Service Unavailable**: Check Jina service URL and status
2. **Milvus Connection Failed**: Verify Milvus host and port configuration
3. **LLM API Errors**: Check API keys and rate limits
4. **Memory Issues**: Reduce batch sizes or image quality settings

### Debug Mode

Enable debug logging:
```env
LOG_LEVEL=DEBUG
```

### Test Individual Components

```python
# Test multimodal embeddings
from app.core.multimodal_embeddings import embedding_service
text_result = await embedding_service.embed_texts(["test text"])
image_result = await embedding_service.embed_images([image_bytes])

# Test vector store with multimodal content
from app.core.vectorstore import vector_store
await vector_store.initialize_collection()
docs = await vector_store.add_documents_with_embeddings(documents)

# Test PDF processing with AI descriptions
from app.core.pdf_processor import pdf_processor
result = await pdf_processor.process_and_embed_pdf("test.pdf")

# Test image storage
from app.core.multimodal_embeddings import image_storage
image_info = image_storage.save_image(image_bytes, "source", 1, 0)
```

### Environment Variables Reference

See `.env.example` for complete configuration options:

- **Provider Selection**: `LLM_PROVIDER`, `EMBEDDING_PROVIDER`
- **Vision Features**: `USE_VISION_LLM`, `USE_VISION_EMBEDDINGS`
- **Image Handling**: `IMAGE_STORAGE_PATH`, `MAX_IMAGES_IN_CONTEXT`
- **Fallback Configuration**: OpenAI keys for embedding/LLM fallback
- **Performance Tuning**: Chunk sizes, batch sizes, image quality

## Key Improvements in This Version

### ✅ Completed Enhancements

1. **Removed Hardcoded Logic**: Generic content processing without hardcoded table/image detection
2. **Dynamic Model Selection**: Configure embedding and LLM providers via `.env`
3. **Proper Image Context**: Images embedded and appended to LLM inference with URLs
4. **AI-Enhanced Descriptions**: Vision LLM generates detailed image descriptions
5. **OpenAI Fallback**: Multimodal embedding fallback to OpenAI API
6. **Vision LLM Integration**: GPT-4o processes actual images for comprehensive answers
7. **Image Storage Service**: Local storage with HTTP serving endpoints

### Production Readiness

- **Robust Error Handling**: Comprehensive fallback mechanisms
- **Flexible Configuration**: Environment-based provider selection
- **Scalable Architecture**: Modular design supporting multiple providers
- **Rich Metadata**: Enhanced image and document metadata
- **Performance Optimized**: Batch processing and intelligent chunking

## Contributing

1. Follow the existing modular architecture patterns
2. Add comprehensive tests for new providers or features
3. Update `.env.example` for new configuration options
4. Ensure fallback mechanisms work properly
5. Test with multiple provider combinations

## License

This project is licensed under the MIT License.