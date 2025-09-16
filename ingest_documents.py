#!/usr/bin/env python3
"""
Document ingestion script for the multimodal RAG system
"""
import asyncio
import logging
from pathlib import Path
from app.core.rag_pipeline import rag_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def ingest_all_documents():
    """Ingest all PDF documents in the project directory"""
    
    # List of PDF files to ingest
    pdf_files = [
        "attention_all_you_need.pdf"
        # "biology_textbook.pdf", 
        # "HCTRA_CSR_Manual.pdf"
    ]
    
    logger.info("Starting document ingestion process...")
    
    for pdf_file in pdf_files:
        pdf_path = Path(pdf_file)
        if pdf_path.exists():
            logger.info(f"Ingesting: {pdf_file}")
            try:
                # Force recreate collection on first document to update schema
                clear_existing = pdf_file == pdf_files[0]
                result = await rag_pipeline.ingest_document(
                    file_path=str(pdf_path),
                    clear_existing=clear_existing
                )
                logger.info(f"Successfully ingested {pdf_file}: {result}")
            except Exception as e:
                logger.error(f"Failed to ingest {pdf_file}: {e}")
        else:
            logger.warning(f"File not found: {pdf_file}")
    
    logger.info("Document ingestion completed!")

if __name__ == "__main__":
    asyncio.run(ingest_all_documents())
