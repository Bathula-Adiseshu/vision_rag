from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    """Request model for document ingestion"""
    file_path: Optional[str] = Field(None, description="Path to the PDF file to ingest")
    clear_existing: bool = Field(False, description="Whether to clear existing data before ingestion")


class IngestResponse(BaseModel):
    """Response model for document ingestion"""
    success: bool
    message: str
    file_path: Optional[str] = None
    document_ids: Optional[List[Union[str, int]]] = None  # Allow both string and int IDs
    processing_stats: Optional[Dict[str, Any]] = None
    collection_stats: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class QueryRequest(BaseModel):
    """Request model for querying the RAG system"""
    query: str = Field(..., description="The question or query to ask")
    search_type: str = Field("hybrid", description="Type of search: 'hybrid', 'similarity', or 'multimodal'")
    k: Optional[int] = Field(None, description="Number of results to retrieve")
    include_images: bool = Field(True, description="Whether to include image references in the response")
    content_type_filter: Optional[str] = Field(None, description="Filter by content type: 'text' or 'image'")


class SourceInfo(BaseModel):
    """Information about a source document"""
    index: int
    type: str  # "text" or "image"
    page: Optional[int] = None
    source: Optional[str] = None
    content: Optional[str] = None
    description: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None


class QueryResponse(BaseModel):
    """Response model for RAG queries"""
    success: bool
    query: str
    answer: str
    sources: List[SourceInfo]
    context_used: str
    search_type: str
    num_sources: int
    num_images: int
    error: Optional[str] = None
    message: Optional[str] = None


class SystemStatusResponse(BaseModel):
    """Response model for system status"""
    system_status: str
    components: Optional[Dict[str, Any]] = None
    configuration: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ClearDataResponse(BaseModel):
    """Response model for clearing data"""
    success: bool
    message: str
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    version: str


class StatusResponse(BaseModel):
    """Response model for system status"""
    status: str
    version: str
    llm_provider: str
    multimodal_enabled: bool
    hybrid_search_enabled: bool
