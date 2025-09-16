import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import settings


# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Multimodal Vision RAG API")
    
    # Create necessary directories
    os.makedirs(settings.data_dir, exist_ok=True)
    os.makedirs(settings.upload_dir, exist_ok=True)
    
    # Initialize components (optional - they initialize lazily)
    try:
        from app.core.rag_pipeline import rag_pipeline
        status = await rag_pipeline.get_system_status()
        logger.info(f"System status: {status.get('system_status', 'unknown')}")
    except Exception as e:
        logger.warning(f"Could not check system status during startup: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Multimodal Vision RAG API")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Advanced multimodal RAG system with vision capabilities",
        openapi_url=f"{settings.api_prefix}/openapi.json",
        docs_url=f"{settings.api_prefix}/docs",
        redoc_url=f"{settings.api_prefix}/redoc",
        lifespan=lifespan
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routes
    app.include_router(router, prefix=settings.api_prefix)
    
    @app.get("/")
    async def root():
        """Root endpoint"""
        return {
            "message": "Multimodal Vision RAG API",
            "version": settings.app_version,
            "docs": f"{settings.api_prefix}/docs"
        }
    
    
    return app


app = create_app()


