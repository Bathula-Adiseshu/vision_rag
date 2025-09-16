#!/usr/bin/env python3
"""
Comprehensive test for multimodal RAG system fixes
Tests image context appending, model selection, and query accuracy
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Add the app directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from app.core.rag_pipeline import rag_pipeline
from app.core.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MultimodalRAGTester:
    """Comprehensive tester for multimodal RAG system"""
    
    def __init__(self):
        self.test_results = {
            "system_status": {},
            "ingestion_tests": {},
            "query_tests": {},
            "image_tests": {},
            "model_selection_tests": {},
            "errors": []
        }
    
    async def run_all_tests(self):
        """Run comprehensive test suite"""
        logger.info("🚀 Starting comprehensive multimodal RAG tests...")
        
        try:
            # Test 1: System Status
            await self.test_system_status()
            
            # Test 2: Document Ingestion
            await self.test_document_ingestion()
            
            # Test 3: Text Queries
            await self.test_text_queries()
            
            # Test 4: Image-based Queries
            await self.test_image_queries()
            
            # Test 5: Model Selection
            await self.test_model_selection()
            
            # Test 6: Vision LLM Integration
            await self.test_vision_llm()
            
            # Generate report
            self.generate_report()
            
        except Exception as e:
            logger.error(f"Test suite failed: {e}")
            self.test_results["errors"].append(str(e))
    
    async def test_system_status(self):
        """Test system status and component availability"""
        logger.info("📊 Testing system status...")
        
        try:
            status = await rag_pipeline.get_system_status()
            self.test_results["system_status"] = status
            
            # Check critical components
            checks = {
                "llm_available": status.get("llm_status") == "available",
                "vector_store_connected": status.get("vector_store_status") == "connected",
                "embedding_service_ready": status.get("embedding_status") == "ready"
            }
            
            logger.info(f"System checks: {checks}")
            self.test_results["system_status"]["checks"] = checks
            
        except Exception as e:
            logger.error(f"System status test failed: {e}")
            self.test_results["errors"].append(f"System status: {e}")
    
    async def test_document_ingestion(self):
        """Test document ingestion with multimodal content"""
        logger.info("📄 Testing document ingestion...")
        
        test_files = [
            "attention_all_you_need.pdf"
            # "biology_textbook.pdf"
        ]
        
        for file_path in test_files:
            if os.path.exists(file_path):
                try:
                    logger.info(f"Ingesting {file_path}...")
                    result = await rag_pipeline.ingest_document(
                        file_path=file_path,
                        clear_existing=(file_path == test_files[0])  # Clear only for first file
                    )
                    
                    self.test_results["ingestion_tests"][file_path] = {
                        "success": result.get("success", False),
                        "documents_count": len(result.get("document_ids", [])),
                        "stats": result.get("processing_stats", {}),
                        "collection_stats": result.get("collection_stats", {})
                    }
                    
                    logger.info(f"✅ {file_path}: {result.get('message', 'Success')}")
                    
                except Exception as e:
                    logger.error(f"❌ Ingestion failed for {file_path}: {e}")
                    self.test_results["errors"].append(f"Ingestion {file_path}: {e}")
            else:
                logger.warning(f"⚠️ Test file not found: {file_path}")
    
    async def test_text_queries(self):
        """Test text-based queries"""
        logger.info("🔍 Testing text queries...")
        
        text_queries = [
            "What is the transformer architecture?",
            "How does attention mechanism work?",
            "What are the key components of the model?",
            "Explain the encoder-decoder structure"
        ]
        
        for query in text_queries:
            try:
                logger.info(f"Querying: {query}")
                result = await rag_pipeline.query(
                    query=query,
                    search_type="hybrid",
                    k=5,
                    include_images=False
                )
                
                self.test_results["query_tests"][query] = {
                    "success": result.get("success", False),
                    "answer_length": len(result.get("answer", "")),
                    "sources_count": len(result.get("sources", [])),
                    "search_type": result.get("search_type"),
                    "has_answer": bool(result.get("answer", "").strip())
                }
                
                logger.info(f"✅ Query successful: {len(result.get('sources', []))} sources")
                
            except Exception as e:
                logger.error(f"❌ Text query failed: {e}")
                self.test_results["errors"].append(f"Text query '{query}': {e}")
    
    async def test_image_queries(self):
        """Test image-based queries - the critical test"""
        logger.info("🖼️ Testing image-based queries...")
        
        image_queries = [
            "Show me diagrams from the document",
            "What figures are available?",
            "Describe any charts or tables",
            "What visual elements are in the document?",
            "Find images with architectural diagrams"
        ]
        
        for query in image_queries:
            try:
                logger.info(f"Image query: {query}")
                result = await rag_pipeline.query(
                    query=query,
                    search_type="multimodal",
                    k=10,
                    include_images=True
                )
                
                # Count image sources
                image_sources = [s for s in result.get("sources", []) if s.get("type") == "image"]
                
                self.test_results["image_tests"][query] = {
                    "success": result.get("success", False),
                    "total_sources": len(result.get("sources", [])),
                    "image_sources": len(image_sources),
                    "answer_length": len(result.get("answer", "")),
                    "has_image_content": any("image" in result.get("answer", "").lower() for _ in [1]),
                    "image_details": image_sources[:3]  # First 3 image sources
                }
                
                logger.info(f"✅ Image query: {len(image_sources)} images found")
                
                # Log image details for verification
                for img_src in image_sources[:2]:
                    logger.info(f"   📸 Image: Page {img_src.get('page')}, {img_src.get('description', 'No description')[:100]}...")
                
            except Exception as e:
                logger.error(f"❌ Image query failed: {e}")
                self.test_results["errors"].append(f"Image query '{query}': {e}")
    
    async def test_model_selection(self):
        """Test model selection configuration"""
        logger.info("⚙️ Testing model selection...")
        
        try:
            # Test LLM config
            llm_config = settings.get_llm_config
            embedding_config = settings.get_embedding_config
            
            self.test_results["model_selection_tests"] = {
                "llm_provider": settings.llm_provider,
                "llm_model": settings.llm_model,
                "embedding_provider": settings.embedding_provider,
                "text_embedding_model": settings.text_embedding_model,
                "vision_embedding_model": settings.vision_embedding_model,
                "use_vision_llm": settings.use_vision_llm,
                "use_vision_embeddings": settings.use_vision_embeddings
            }
            
            logger.info(f"✅ LLM Provider: {settings.llm_provider} ({settings.llm_model})")
            logger.info(f"✅ Embedding Provider: {settings.embedding_provider}")
            logger.info(f"✅ Vision Features: LLM={settings.use_vision_llm}, Embeddings={settings.use_vision_embeddings}")
            
        except Exception as e:
            logger.error(f"❌ Model selection test failed: {e}")
            self.test_results["errors"].append(f"Model selection: {e}")
    
    async def test_vision_llm(self):
        """Test vision LLM integration with actual images"""
        logger.info("👁️ Testing vision LLM integration...")
        
        if not settings.use_vision_llm or not settings.openai_api_key:
            logger.warning("⚠️ Vision LLM not enabled or OpenAI key not available")
            return
        
        vision_queries = [
            "What do you see in the transformer architecture diagram?",
            "Describe the visual elements in the figures",
            "Analyze any charts or graphs in the document"
        ]
        
        for query in vision_queries:
            try:
                logger.info(f"Vision query: {query}")
                result = await rag_pipeline.query(
                    query=query,
                    search_type="multimodal",
                    k=5,
                    include_images=True
                )
                
                # Check if vision LLM was actually used
                image_sources = [s for s in result.get("sources", []) if s.get("type") == "image"]
                
                self.test_results["image_tests"][f"vision_{query}"] = {
                    "success": result.get("success", False),
                    "used_vision_llm": len(image_sources) > 0 and settings.use_vision_llm,
                    "image_count": len(image_sources),
                    "answer_quality": len(result.get("answer", "")) > 100
                }
                
                logger.info(f"✅ Vision query processed with {len(image_sources)} images")
                
            except Exception as e:
                logger.error(f"❌ Vision LLM test failed: {e}")
                self.test_results["errors"].append(f"Vision LLM '{query}': {e}")
    
    def generate_report(self):
        """Generate comprehensive test report"""
        logger.info("📋 Generating test report...")
        
        report = {
            "test_summary": {
                "timestamp": str(asyncio.get_event_loop().time()),
                "total_errors": len(self.test_results["errors"]),
                "system_healthy": len(self.test_results["errors"]) == 0
            },
            "configuration": {
                "llm_provider": settings.llm_provider,
                "llm_model": settings.llm_model,
                "embedding_provider": settings.embedding_provider,
                "vision_enabled": settings.use_vision_llm,
                "vision_embeddings": settings.use_vision_embeddings
            },
            "detailed_results": self.test_results
        }
        
        # Save report
        with open("multimodal_test_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print summary
        print("\n" + "="*80)
        print("🎯 MULTIMODAL RAG TEST REPORT")
        print("="*80)
        
        print(f"📊 System Status: {'✅ HEALTHY' if report['test_summary']['system_healthy'] else '❌ ISSUES'}")
        print(f"🔧 LLM: {settings.llm_provider} ({settings.llm_model})")
        print(f"🔗 Embeddings: {settings.embedding_provider}")
        print(f"👁️ Vision LLM: {'✅ ENABLED' if settings.use_vision_llm else '❌ DISABLED'}")
        
        if self.test_results["ingestion_tests"]:
            print(f"\n📄 Document Ingestion:")
            for file, result in self.test_results["ingestion_tests"].items():
                status = "✅" if result["success"] else "❌"
                print(f"   {status} {file}: {result['documents_count']} documents")
        
        if self.test_results["image_tests"]:
            print(f"\n🖼️ Image Query Results:")
            image_query_count = len([t for t in self.test_results["image_tests"].values() if t.get("image_sources", 0) > 0])
            print(f"   Queries with images found: {image_query_count}/{len(self.test_results['image_tests'])}")
        
        if self.test_results["errors"]:
            print(f"\n❌ Errors ({len(self.test_results['errors'])}):")
            for error in self.test_results["errors"][:5]:  # Show first 5 errors
                print(f"   • {error}")
        
        print(f"\n📁 Full report saved to: multimodal_test_report.json")
        print("="*80)

async def main():
    """Main test runner"""
    tester = MultimodalRAGTester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())
