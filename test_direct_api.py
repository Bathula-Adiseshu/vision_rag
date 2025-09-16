#!/usr/bin/env python3
"""
Direct test with explicit environment loading and API validation
"""
import asyncio
import os
import json
import logging
from dotenv import load_dotenv

# Force load environment variables
load_dotenv(override=True)

# Set environment variables directly
os.environ['LLM_PROVIDER'] = 'openai'
os.environ['USE_VISION_LLM'] = 'true'
os.environ['ENABLE_VISION'] = 'true'
os.environ['EMBEDDING_PROVIDER'] = 'openai'

from app.core.rag_pipeline import rag_pipeline
from app.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_with_direct_config():
    """Test multimodal RAG with direct configuration"""
    print("🔍 DIRECT API TEST WITH EXPLICIT CONFIG")
    print("=" * 50)
    
    # Print actual environment variables
    print(f"🔧 ENV LLM_PROVIDER: {os.getenv('LLM_PROVIDER')}")
    print(f"🔧 ENV OPENAI_API_KEY: {os.getenv('OPENAI_API_KEY', 'NOT_SET')[:15]}...")
    print(f"🔧 ENV DEEPSEEK_API_KEY: {os.getenv('DEEPSEEK_API_KEY', 'NOT_SET')[:15]}...")
    
    # Print settings values
    print(f"🔧 Settings LLM Provider: {settings.llm_provider}")
    print(f"🔧 Settings OpenAI Key: {settings.openai_api_key[:15] if settings.openai_api_key else 'None'}...")
    print(f"🔧 Settings DeepSeek Key: {settings.deepseek_api_key[:15] if settings.deepseek_api_key else 'None'}...")
    print(f"🔧 Settings Vision LLM: {settings.use_vision_llm}")
    
    # Test OpenAI API directly
    print("\n🧪 TESTING OPENAI API DIRECTLY...")
    try:
        import openai
        client = openai.AsyncOpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            base_url="https://api.openai.com/v1"
        )
        
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello, this is a test."}],
            max_tokens=50
        )
        print(f"✅ OpenAI API test successful: {response.choices[0].message.content}")
        
    except Exception as e:
        print(f"❌ OpenAI API test failed: {e}")
    
    # Test document ingestion
    print("\n📄 TESTING DOCUMENT INGESTION...")
    result = await rag_pipeline.ingest_document(
        "attention_all_you_need.pdf",
        clear_existing=True
    )
    print(f"✅ Ingestion: {result['message']}")
    
    # Test simple query first
    print("\n💬 TESTING SIMPLE TEXT QUERY...")
    result = await rag_pipeline.query(
        query="What is the transformer architecture?",
        search_type="hybrid",
        k=5,
        include_images=False
    )
    print(f"📊 Sources: {len(result.get('sources', []))}")
    print(f"🤖 Answer: {result.get('answer', 'No answer')[:100]}...")
    
    # Test image query with explicit image search
    print("\n🖼️ TESTING IMAGE QUERY...")
    result = await rag_pipeline.query(
        query="Show me the transformer architecture diagram",
        search_type="hybrid",
        k=10,
        include_images=True
    )
    
    sources = result.get('sources', [])
    image_sources = [s for s in sources if s.get('content_type') == 'image']
    
    print(f"📊 Total sources: {len(sources)}")
    print(f"🖼️ Image sources: {len(image_sources)}")
    
    if image_sources:
        print("\n🔍 IMAGE SOURCES DETAILS:")
        for i, img in enumerate(image_sources):
            print(f"  {i+1}. Page {img.get('page')}: {img.get('content', '')}")
            print(f"     Metadata: {img.get('metadata', {})}")
            
            # Check actual image files
            possible_paths = [
                img.get('image_path'),
                img.get('metadata', {}).get('image_path'),
                f"uploads/images/attention_all_you_need.pdf_page_{img.get('page')}_img_0_*.jpg"
            ]
            
            for path in possible_paths:
                if path and os.path.exists(path):
                    print(f"     ✅ Found image at: {path}")
                    break
            else:
                print(f"     ❌ No image file found")
    
    print(f"👁️ Vision LLM used: {result.get('used_vision_llm', False)}")
    print(f"🖼️ Images in context: {result.get('images_in_context', 0)}")
    print(f"🤖 Answer: {result.get('answer', 'No answer')[:200]}...")
    
    # Check image files directly
    print("\n📁 CHECKING IMAGE FILES...")
    image_dir = "uploads/images"
    if os.path.exists(image_dir):
        image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        print(f"Found {len(image_files)} image files:")
        for img_file in image_files[:5]:
            print(f"  - {img_file}")
    else:
        print("❌ Image directory not found")
    
    print("\n✅ Direct test complete!")

if __name__ == "__main__":
    asyncio.run(test_with_direct_config())
