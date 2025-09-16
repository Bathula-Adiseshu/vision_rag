#!/usr/bin/env python3
"""
Startup script for the Multimodal Vision RAG API server
"""

import uvicorn
import argparse
import logging
from pathlib import Path

# Add the app directory to Python path
import sys
sys.path.insert(0, str(Path(__file__).parent / "app"))

from app.core.config import settings


def main():
    parser = argparse.ArgumentParser(description="Start the Multimodal Vision RAG API server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    
    args = parser.parse_args()
    
    # Configure logging
    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "root": {
            "level": settings.log_level,
            "handlers": ["default"],
        },
    }
    
    print(f"Starting {settings.app_name} v{settings.app_version}")
    print(f"Server will be available at: http://{args.host}:{args.port}")
    print(f"API documentation: http://{args.host}:{args.port}{settings.api_prefix}/docs")
    print(f"LLM Provider: {settings.llm_provider}")
    print(f"Multimodal enabled: {settings.enable_vision}")
    print(f"Hybrid search enabled: {settings.enable_hybrid}")
    
    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_config=log_config,
        access_log=True
    )


if __name__ == "__main__":
    main()
