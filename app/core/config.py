from functools import lru_cache
from typing import Optional, Dict, Any, List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False)

    # API
    api_prefix: str = "/api"
    app_name: str = "Multimodal Vision RAG"
    app_version: str = "1.0.0"

    # LLM Configuration
    llm_provider: str = Field(default="deepseek", description="LLM provider: deepseek, openai, or claude")
    llm_model: str = Field(default="deepseek-chat", description="Specific LLM model name")

    # DeepSeek Configuration
    deepseek_api_key: Optional[str] = Field(default=None, description="DeepSeek API key")
    deepseek_base_url: str = Field(default="https://api.deepseek.com", description="DeepSeek API base URL")
    deepseek_model: str = Field(default="deepseek-chat", description="DeepSeek model name")

    # Additional LLM Models
    claude_api_key: Optional[str] = Field(default=None, description="Claude API key")
    claude_model: str = Field(default="claude-3-sonnet-20240229", description="Claude model name")

    # OpenAI Configuration (fallback)
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    openai_base_url: str = Field(default="https://api.openai.com/v1", description="OpenAI API base URL")
    openai_model: str = Field(default="gpt-4o", description="OpenAI model name")

    # Embedding Provider and Model Selection
    embedding_provider: str = Field(default="jina", description="Embedding provider: jina or openai")
    text_embedding_model: str = Field(default="jinaai/jina-embeddings-v3", description="Text embedding model name")
    vision_embedding_model: str = Field(default="jinaai/jina-clip-v2", description="Vision embedding model name")
    openai_embedding_model: str = Field(default="text-embedding-3-large", description="OpenAI embedding model")
    use_vision_embeddings: bool = Field(default=True, description="Use vision embeddings for images")

    # Jina Embeddings Configuration
    jina_embedding_url: str = Field(default="http://localhost:8000/v1/embeddings", description="Jina embedding service URL")
    jina_text_model: str = Field(default="jinaai/jina-embeddings-v3", description="Jina text embedding model")
    jina_vision_model: str = Field(default="jinaai/jina-clip-v2", description="Jina vision embedding model")

    # OpenAI Vision Models
    openai_vision_model: str = Field(default="gpt-4o", description="OpenAI vision model for image analysis")

    # Image Storage Configuration
    image_storage_path: str = Field(default="./uploads/images", description="Path to store extracted images")
    image_base_url: str = Field(default="http://localhost:8080/images", description="Base URL for serving images")

    # Milvus Configuration
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_user: Optional[str] = None
    milvus_password: Optional[str] = None
    milvus_db_name: str = "multimodal_rag"
    milvus_collection_name: str = "vision_rag_collection"
    milvus_timeout: int = 30

    # Vector dimensions
    milvus_dense_dim: int = 768  # Jina text embeddings
    milvus_sparse_dim: int = 768  # BM25 sparse embeddings
    milvus_vision_dim: int = 768  # Jina vision embeddings

    # Milvus Index Parameters
    milvus_index_params: Dict[str, Any] = {
        "dense": {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 200}
        },
        "sparse": {
            "index_type": "SPARSE_INVERTED_INDEX",
            "metric_type": "IP"
        }
    }

    # Milvus Search Parameters
    milvus_search_params: Dict[str, Any] = {
        "dense": {"metric_type": "COSINE", "params": {"ef": 100}},
        "sparse": {"metric_type": "IP", "params": {}}
    }

    # Hybrid Search Configuration
    milvus_k: int = 10
    fetch_k: int = 20
    milvus_ranker_type: str = "rrf"  # reciprocal rank fusion
    milvus_ranker_params: Dict[str, Any] = {"k": 60}
    milvus_dense_ranker_params: Dict[str, Any] = {"k": 60}
    milvus_sparse_ranker_params: Dict[str, Any] = {"k": 60}

    # Hybrid Search Weights
    enable_hybrid: bool = True
    bm25_k1: float = 1.6
    bm25_b: float = 0.75
    hybrid_alpha: float = 0.6  # weight for dense vs sparse

    # Multimodal Processing
    enable_vision: bool = True
    max_image_size: int = 1024  # max image dimension
    image_quality: int = 85  # JPEG quality for compression
    extract_images_from_pdf: bool = True
    
    # Text Processing
    chunk_size: int = 800
    chunk_overlap: int = 100
    min_chunk_size: int = 50
    max_chunks_per_document: int = 1000

    # Image Processing
    image_chunk_strategy: str = "page_based"  # or "region_based"
    max_images_per_page: int = 10
    image_description_prompt: str = "Describe this image in detail, focusing on key visual elements, text content, diagrams, charts, and any relevant information."

    # RAG Configuration
    max_context_length: int = 4000
    temperature: float = 0.1
    max_tokens: int = 1000
    top_k_retrieval: int = 5
    similarity_threshold: float = 0.7

    # Application Settings
    debug: bool = Field(default=False, description="Debug mode")
    log_level: str = Field(default="INFO", description="Logging level")
    enable_vision: bool = Field(default=True, description="Enable vision capabilities")
    enable_hybrid: bool = Field(default=True, description="Enable hybrid search")
    embedding_dimension: int = Field(default=2048, description="Embedding vector dimension")
    use_vision_llm: bool = Field(default=True, description="Use vision-capable LLM for image queries")
    max_images_in_context: int = Field(default=5, description="Maximum images to include in LLM context")

    # Runtime
    enable_tracing: bool = False
    data_dir: str = "data"
    upload_dir: str = "uploads"
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration based on selected provider and model"""
        if self.llm_provider == "deepseek":
            return {
                "provider": "deepseek",
                "api_key": self.deepseek_api_key,
                "base_url": self.deepseek_base_url,
                "model": self.llm_model or self.deepseek_model
            }
        elif self.llm_provider == "openai":
            return {
                "provider": "openai",
                "api_key": self.openai_api_key,
                "base_url": self.openai_base_url,
                "model": self.llm_model or self.openai_model
            }
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
    
    def get_embedding_config(self) -> Dict[str, Any]:
        """Get embedding configuration based on selected provider and models"""
        if self.embedding_provider == "jina":
            return {
                "provider": "jina",
                "url": self.jina_embedding_url,
                "text_model": self.text_embedding_model or self.jina_text_model,
                "vision_model": self.vision_embedding_model or self.jina_vision_model
            }
        elif self.embedding_provider == "openai":
            return {
                "provider": "openai",
                "api_key": self.openai_api_key,
                "base_url": self.openai_base_url,
                "text_model": self.text_embedding_model or self.openai_embedding_model,
                "vision_model": self.openai_vision_model
            }
        else:
            raise ValueError(f"Unsupported embedding provider: {self.embedding_provider}")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


# Global settings instance
settings = get_settings()


