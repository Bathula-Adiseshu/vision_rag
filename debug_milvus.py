#!/usr/bin/env python3
"""
Debug script to test Milvus connection and document storage
"""
import asyncio
import logging
from app.core.config import settings
from app.core.vectorstore import vector_store
from app.core.multimodal_embeddings import embedding_service
from langchain.schema import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_milvus_connection():
    """Test basic Milvus connection and operations"""
    print("🔍 Testing Milvus connection and document storage...")
    
    try:
        # Test connection setup
        print(f"📡 Connecting to Milvus at {settings.milvus_host}:{settings.milvus_port}")
        connection_ok = vector_store._setup_connection()
        print(f"✅ Connection setup: {connection_ok}")
        
        # Test collection initialization
        print("🏗️ Initializing collection...")
        init_ok = await vector_store.initialize_collection(force_recreate=True)
        print(f"✅ Collection initialization: {init_ok}")
        
        # Get collection stats
        stats = vector_store.get_collection_stats()
        print(f"📊 Collection stats: {stats}")
        
        # Test embedding service
        print("🔤 Testing embedding service...")
        test_texts = ["This is a test document", "Another test document"]
        embeddings = await embedding_service.embed_texts(test_texts)
        print(f"✅ Embeddings generated: {len(embeddings)} vectors of dimension {len(embeddings[0]) if embeddings else 0}")
        
        # Test document addition
        print("📄 Testing document addition...")
        test_docs = [
            Document(
                page_content="This is a test document about transformers",
                metadata={"content_type": "text", "source": "test", "page": 1}
            ),
            Document(
                page_content="Another test document about attention mechanisms", 
                metadata={"content_type": "text", "source": "test", "page": 2}
            )
        ]
        
        doc_ids = await vector_store.add_documents_with_embeddings(test_docs)
        print(f"✅ Documents added: {len(doc_ids)} IDs: {doc_ids}")
        
        # Check collection stats after adding documents
        stats_after = vector_store.get_collection_stats()
        print(f"📊 Collection stats after adding docs: {stats_after}")
        
        # Test search
        print("🔍 Testing search...")
        results = await vector_store.similarity_search("transformer", k=2)
        print(f"✅ Search results: {len(results)} documents found")
        for i, doc in enumerate(results):
            print(f"  {i+1}. {doc.page_content[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(test_milvus_connection())
