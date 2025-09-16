#!/usr/bin/env python3
"""
Reset the Milvus collection with updated schema
"""
import asyncio
import logging
from pymilvus import connections, utility
from app.core.config import settings
from app.core.vectorstore import vector_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def reset_collection():
    """Drop and recreate the collection with updated schema"""
    
    try:
        # Connect to Milvus
        connections.connect(
            alias="default",
            host=settings.milvus_host,
            port=settings.milvus_port,
            user=settings.milvus_user,
            password=settings.milvus_password,
            db_name=settings.milvus_db_name
        )
        
        # Drop collection if it exists
        if utility.has_collection(settings.milvus_collection_name):
            utility.drop_collection(settings.milvus_collection_name)
            logger.info(f"Dropped existing collection: {settings.milvus_collection_name}")
        
        # Initialize with new schema
        await vector_store.initialize_collection(force_recreate=True)
        logger.info("Collection reset successfully!")
        
    except Exception as e:
        logger.error(f"Failed to reset collection: {e}")

if __name__ == "__main__":
    asyncio.run(reset_collection())
