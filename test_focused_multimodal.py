#!/usr/bin/env python3
"""
Focused test script for multimodal RAG with proper API key loading
"""
import asyncio
import os
import json
import logging
from dotenv import load_dotenv
from app.core.rag_pipeline import rag_pipeline
from app.core.config import settings

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_multimodal_rag():
    """Test multimodal RAG with image and table queries"""
    print("🔍 FOCUSED MULTIMODAL RAG TEST")
    print("=" * 50)
    
    # Print configuration
    print(f"🔧 LLM Provider: {settings.llm_provider}")
    print(f"🔧 OpenAI API Key: {settings.openai_api_key[:10]}..." if settings.openai_api_key else "❌ No OpenAI key")
    print(f"🔧 DeepSeek API Key: {settings.deepseek_api_key[:10]}..." if settings.deepseek_api_key else "❌ No DeepSeek key")
    print(f"🔧 Vision LLM: {settings.use_vision_llm}")
    print(f"🔧 Embedding Provider: {settings.embedding_provider}")
    
    # Test 1: Ingest document
    print("\n📄 STEP 1: INGESTING DOCUMENT...")
    result = await rag_pipeline.ingest_document(
        "attention_all_you_need.pdf",
        clear_existing=True
    )
    print(f"✅ Ingestion: {result['message']}")
    print(f"📊 Documents: {result['processing_stats']}")
    
    # Test 2: Image-based query
    print("\n🖼️ STEP 2: TESTING IMAGE QUERY...")
    image_query = "Show me the transformer architecture diagram and explain what you see"
    
    result = await rag_pipeline.query(
        query=image_query,
        search_type="hybrid",
        k=8,
        include_images=True
    )
    
    sources = result.get('sources', [])
    image_sources = [s for s in sources if s.get('content_type') == 'image']
    
    print(f"📊 Total sources: {len(sources)}")
    print(f"🖼️ Image sources: {len(image_sources)}")
    print(f"👁️ Vision LLM used: {result.get('used_vision_llm', False)}")
    print(f"🖼️ Images in context: {result.get('images_in_context', 0)}")
    print(f"✅ Success: {result.get('success', False)}")
    
    if image_sources:
        print("\n🔍 IMAGE SOURCES FOUND:")
        for i, img in enumerate(image_sources[:3]):
            print(f"  {i+1}. Page {img.get('page')}: {img.get('content', '')[:80]}...")
            print(f"     Path: {img.get('image_path', 'N/A')}")
            print(f"     URL: {img.get('image_url', 'N/A')}")
    
    print(f"\n🤖 ANSWER ({len(result.get('answer', ''))} chars):")
    print(f"{result.get('answer', 'No answer')[:300]}...")
    
    # Test 3: Table-based query
    print("\n📊 STEP 3: TESTING TABLE QUERY...")
    table_query = "What are the performance metrics and BLEU scores in the results table?"
    
    result = await rag_pipeline.query(
        query=table_query,
        search_type="hybrid",
        k=5,
        include_images=True
    )
    
    print(f"📊 Sources found: {len(result.get('sources', []))}")
    print(f"🤖 Answer: {result.get('answer', 'No answer')[:200]}...")
    
    # Test 4: Architecture query with vision
    print("\n🏗️ STEP 4: TESTING ARCHITECTURE QUERY...")
    arch_query = "Describe the multi-head attention mechanism using the diagram"
    
    result = await rag_pipeline.query(
        query=arch_query,
        search_type="multimodal",
        k=6,
        include_images=True
    )
    
    print(f"📊 Sources: {len(result.get('sources', []))}")
    print(f"👁️ Vision LLM: {result.get('used_vision_llm', False)}")
    print(f"🖼️ Images in context: {result.get('images_in_context', 0)}")
    print(f"🤖 Answer: {result.get('answer', 'No answer')[:200]}...")
    
    # Save results
    test_results = {
        'configuration': {
            'llm_provider': settings.llm_provider,
            'has_openai_key': bool(settings.openai_api_key and not settings.openai_api_key.startswith('your_')),
            'has_deepseek_key': bool(settings.deepseek_api_key and not settings.deepseek_api_key.startswith('your_')),
            'vision_enabled': settings.use_vision_llm,
            'embedding_provider': settings.embedding_provider
        },
        'tests': {
            'image_query': {
                'query': image_query,
                'total_sources': len(sources),
                'image_sources': len(image_sources),
                'vision_llm_used': result.get('used_vision_llm', False),
                'images_in_context': result.get('images_in_context', 0),
                'success': result.get('success', False),
                'answer_length': len(result.get('answer', ''))
            }
        }
    }
    
    with open('focused_test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\n💾 Results saved to: focused_test_results.json")
    print("✅ Test complete!")

if __name__ == "__main__":
    asyncio.run(test_multimodal_rag())
