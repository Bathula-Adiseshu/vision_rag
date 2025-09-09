from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False)

    # API
    api_prefix: str = "/api"
    app_name: str = "Vision RAG"
    app_version: str = "0.1.0"

    # LLM Providers
    llm_provider: str = "openai"  # or "deepseek"
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None
    deepseek_api_key: Optional[str] = None
    deepseek_base_url: Optional[str] = None

    # Embeddings service (Jina vLLM)
    embeddings_base_url: str = "http://localhost:8000"
    embeddings_model: str = "jinaai/jina-embeddings-v4-vllm-code"
    embeddings_timeout_seconds: float = 30.0

    # Milvus
    milvus_uri: str = "http://localhost:19530"
    milvus_user: Optional[str] = None
    milvus_password: Optional[str] = None
    milvus_collection: str = "vision_rag_docs"
    milvus_dim: int = 2048
    milvus_index_type: str = "IVF_FLAT"
    milvus_metric_type: str = "COSINE"
    milvus_nlist: int = 2048

    # Hybrid Search
    enable_hybrid: bool = True
    bm25_k1: float = 1.6
    bm25_b: float = 0.75
    hybrid_alpha: float = 0.6  # weight for dense vs sparse

    # Ingestion
    chunk_size: int = 800
    chunk_overlap: int = 100

    # Runtime
    log_level: str = "INFO"
    enable_tracing: bool = False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


