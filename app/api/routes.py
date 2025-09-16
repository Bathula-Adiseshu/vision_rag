from typing import Any, Dict, List, Optional
import os
import tempfile
import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException, UploadFile, File, Body, status, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import logging
import asyncio
import os

from app.core.config import settings
from app.core.rag_pipeline import rag_pipeline
from app.api.models import (
    IngestRequest, IngestResponse, QueryRequest, QueryResponse,
    SystemStatusResponse, ClearDataResponse, HealthResponse, SourceInfo, StatusResponse
)
from loguru import logger


router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: Optional[UploadFile] = File(None),
    request: Optional[IngestRequest] = Body(None),
    clear_existing: bool = Query(False)
) -> IngestResponse:
    """
    Ingest a PDF document into the multimodal RAG system
    """
    try:
        # Determine file path
        file_path = None
        temp_file = None
        
        if file is not None:
            # Handle uploaded file
            if not file.filename or not file.filename.lower().endswith('.pdf'):
                raise HTTPException(
                    status_code=400, 
                    detail="Only PDF files are supported"
                )
            
            # Save uploaded file to temporary location
            suffix = os.path.splitext(file.filename)[1] or ".pdf"
            fd, temp_file = tempfile.mkstemp(prefix="ingest_", suffix=suffix)
            
            try:
                with os.fdopen(fd, "wb") as out:
                    content = await file.read()
                    out.write(content)
                file_path = temp_file
            except Exception as e:
                if temp_file and os.path.exists(temp_file):
                    os.remove(temp_file)
                raise HTTPException(
                    status_code=400, 
                    detail=f"Failed to save uploaded file: {e}"
                )
        
        elif request and request.file_path:
            # Handle file path from request
            file_path = request.file_path
            if not os.path.exists(file_path):
                raise HTTPException(
                    status_code=404, 
                    detail=f"File not found: {file_path}"
                )
        
        else:
            raise HTTPException(
                status_code=400, 
                detail="Either upload a file or provide a file_path"
            )
        
        # Get clear_existing flag
        clear_existing = request.clear_existing if request else False
        
        # Process document
        result = await rag_pipeline.ingest_document(
            file_path=file_path,
            clear_existing=clear_existing
        )

        # Convert document IDs to strings if they exist
        if result.get('document_ids'):
            result['document_ids'] = [str(doc_id) for doc_id in result['document_ids']]

        return IngestResponse(
            success=True,
            message=f"Successfully ingested document: {file_path}",
            file_path=file_path,
            **result
        )

    except Exception as e:
        logger.error(f"Ingestion endpoint error: {str(e)}")
        return IngestResponse(
            success=False,
            message="Failed to ingest document",
            error=str(e)
        )


@router.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest) -> QueryResponse:
    """
    Query the multimodal RAG system
    """
    try:
        # Validate query
        if not request.query or not request.query.strip():
            raise HTTPException(
                status_code=422, 
                detail="Query text is required and cannot be empty"
            )
        
        # Process query
        result = await rag_pipeline.query(
            query=request.query,
            search_type=request.search_type,
            k=request.k,
            include_images=request.include_images,
            content_type_filter=request.content_type_filter
        )
        
        # Convert sources to response format
        sources = []
        for source in result.get("sources", []):
            sources.append(SourceInfo(**source))
        
        if result.get("success"):
            # Use safe defaults to prevent KeyError when upstream fails to populate fields
            return QueryResponse(
                success=True,
                query=result.get("query", request.query),
                answer=result.get("answer", ""),
                sources=sources,
                context_used=result.get("context_used", ""),
                search_type=result.get("search_type", request.search_type),
                num_sources=int(result.get("num_sources", len(sources))),
                num_images=int(result.get("num_images", 0))
            )
        else:
            return QueryResponse(
                success=False,
                query=request.query,
                answer="",
                sources=[],
                context_used="",
                search_type=request.search_type,
                num_sources=0,
                num_images=0,
                error=result.get("error"),
                message=result.get("message")
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query endpoint error: {e}")
        return QueryResponse(
            success=False,
            query=request.query,
            answer="",
            sources=[],
            context_used="",
            search_type=request.search_type,
            num_sources=0,
            num_images=0,
            error=str(e),
            message="Internal server error during query processing"
        )


@router.get("/status", response_model=SystemStatusResponse)
async def get_system_status() -> SystemStatusResponse:
    """
    Get comprehensive system status
    """
    try:
        result = await rag_pipeline.get_system_status()
        
        if result.get("system_status") == "operational":
            return SystemStatusResponse(
                system_status=result["system_status"],
                components=result.get("components"),
                configuration=result.get("configuration")
            )
        else:
            return SystemStatusResponse(
                system_status=result.get("system_status", "error"),
                error=result.get("error")
            )
    
    except Exception as e:
        logger.error(f"Status endpoint error: {e}")
        return SystemStatusResponse(
            success=True,
            status="operational",
            timestamp=datetime.utcnow(),
            **status
        )
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Status check failed: {str(e)}"
        )


@router.get("/images/{filename}")
async def serve_image(filename: str):
    """
    Serve stored images
    """
    try:
        image_path = os.path.join(settings.image_storage_path, filename)
        if not os.path.exists(image_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Image not found"
            )
        return FileResponse(image_path)
    except Exception as e:
        logger.error(f"Image serving failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to serve image: {str(e)}"
        )


@router.delete("/clear", response_model=ClearDataResponse)
async def clear_all_data() -> ClearDataResponse:
    """
    Clear all data from the RAG system
    """
    try:
        result = await rag_pipeline.clear_all_data()
        
        return ClearDataResponse(
            success=result["success"],
            message=result["message"],
            error=result.get("error")
        )
    
    except Exception as e:
        logger.error(f"Clear data endpoint error: {e}")
        return ClearDataResponse(
            success=False,
            message="Failed to clear data",
            error=str(e)
        )


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Simple health check endpoint
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version=settings.app_version
    )


# Legacy endpoints for backward compatibility
@router.post("/reindex")
async def reindex_endpoint() -> Dict[str, Any]:
    """Legacy endpoint - redirects to clear data"""
    try:
        result = await rag_pipeline.clear_all_data()
        return {"status": "cleared" if result["success"] else "failed"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@router.get("/stats")
async def stats_endpoint() -> Dict[str, Any]:
    """Legacy endpoint - redirects to system status"""
    try:
        result = await rag_pipeline.get_system_status()
        return {
            "status": result.get("system_status", "unknown"),
            "collection_stats": result.get("components", {}).get("vector_store", {}).get("collection_stats", {})
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


