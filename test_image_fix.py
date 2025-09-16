#!/usr/bin/env python3
"""
Test script to validate image path resolution fixes
"""
import asyncio
import os
from dotenv import load_dotenv

load_dotenv(override=True)
os.environ['LLM_PROVIDER'] = 'openai'
os.environ['USE_VISION_LLM'] = 'true'
# Force embeddings to use OpenAI to avoid Jina server timeouts during tests
os.environ['EMBEDDING_PROVIDER'] = 'openai'

from app.core.rag_pipeline import rag_pipeline

async def test_image_fix():
    """Test the image path resolution fix"""
    print("🔧 TESTING IMAGE PATH RESOLUTION FIX")
    print("=" * 50)
    
    # Ingest document
    print("📄 Ingesting document...")
    result = await rag_pipeline.ingest_document(
        "attention_all_you_need.pdf",
        clear_existing=True
    )
    print(f"✅ {result['message']}")
    
    # Test image query
    print("\n🖼️ Testing image query...")
    result = await rag_pipeline.query(
        query="Show me the transformer architecture diagram",
        search_type="hybrid",
        k=10,
        include_images=True
    )
    
    print(f"📊 Total sources: {len(result.get('sources', []))}")
    print(f"🖼️ Vision LLM used: {result.get('used_vision_llm', False)}")
    print(f"🖼️ Images in context: {result.get('images_in_context', 0)}")
    
    # Check image sources
    image_sources = [s for s in result.get('sources', []) if s.get('content_type') == 'image']
    print(f"🖼️ Image sources found: {len(image_sources)}")
    
    for i, img_src in enumerate(image_sources):
        print(f"  {i+1}. Page {img_src.get('page')}: {img_src.get('content', '')[:50]}...")
    
    print(f"\n🤖 Answer: {result.get('answer', 'No answer')[:200]}...")
    
    # Test table query
    print("\n📊 Testing table query...")
    result = await rag_pipeline.query(
        query="Show me performance comparison tables",
        search_type="hybrid",
        k=10,
        include_images=True
    )
    
    print(f"📊 Total sources: {len(result.get('sources', []))}")
    print(f"🖼️ Vision LLM used: {result.get('used_vision_llm', False)}")
    print(f"🖼️ Images in context: {result.get('images_in_context', 0)}")
    print(f"🤖 Answer: {result.get('answer', 'No answer')[:200]}...")

if __name__ == "__main__":
    asyncio.run(test_image_fix())
