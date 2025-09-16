#!/usr/bin/env python3
"""
Debug script to check image document storage and retrieval
"""
import asyncio
import os
from dotenv import load_dotenv

load_dotenv(override=True)

from app.core.vectorstore import vector_store
from app.core.config import settings

async def debug_image_storage():
    """Debug image document storage in vector database"""
    print("🔍 DEBUGGING IMAGE STORAGE IN VECTOR DATABASE")
    print("=" * 50)
    
    # Initialize vector store
    await vector_store.initialize_collection()
    
    # Get collection stats
    stats = vector_store.get_collection_stats()
    print(f"📊 Collection stats: {stats}")
    
    # Search for image documents specifically
    print("\n🖼️ SEARCHING FOR IMAGE DOCUMENTS...")
    image_docs = await vector_store.similarity_search(
        query="transformer architecture diagram",
        k=10,
        expr='content_type == "image"'
    )
    
    print(f"Found {len(image_docs)} image documents")
    
    for i, doc in enumerate(image_docs):
        print(f"\n📄 Image Document {i+1}:")
        print(f"  Content: {doc.page_content[:100]}...")
        print(f"  Metadata keys: {list(doc.metadata.keys())}")
        
        # Check specific metadata fields
        for key in ['image_path', 'image_url', 'filepath', 'source', 'page']:
            value = doc.metadata.get(key)
            print(f"  {key}: {value}")
        
        # Check if image file exists
        image_path = doc.metadata.get('image_path')
        if image_path:
            exists = os.path.exists(image_path)
            print(f"  File exists: {exists}")
        else:
            print(f"  ❌ No image_path in metadata")
            
            # Try to construct path from other metadata
            source = doc.metadata.get('source', '')
            page = doc.metadata.get('page', 0)
            
            # Check uploads directory
            uploads_dir = "uploads/images"
            if os.path.exists(uploads_dir):
                files = [f for f in os.listdir(uploads_dir) 
                        if f.startswith(f"{os.path.basename(source)}_page_{page}")]
                print(f"  Possible files: {files}")
    
    # Also search for all documents to see the difference
    print("\n📚 SEARCHING ALL DOCUMENTS...")
    all_docs = await vector_store.similarity_search(
        query="transformer",
        k=5
    )
    
    for i, doc in enumerate(all_docs):
        content_type = doc.metadata.get('content_type', 'unknown')
        print(f"  Doc {i+1}: {content_type} - {doc.page_content[:50]}...")
        if content_type == 'image':
            print(f"    image_path: {doc.metadata.get('image_path')}")

if __name__ == "__main__":
    asyncio.run(debug_image_storage())
