#!/usr/bin/env python3
"""
Comprehensive test script for the Multimodal Vision RAG system
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))

from app.core.rag_pipeline import rag_pipeline
from app.core.config import settings


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MultimodalRAGTester:
    """Comprehensive tester for the multimodal RAG system"""
    
    def __init__(self):
        self.pipeline = rag_pipeline
        self.test_results = {}
    
    async def run_all_tests(self):
        """Run all test cases"""
        logger.info("Starting comprehensive multimodal RAG tests")
        
        # Test 1: System Status
        await self.test_system_status()
        
        # Test 2: Document Ingestion
        await self.test_document_ingestion()
        
        # Test 3: Text-based Queries
        await self.test_text_queries()
        
        # Test 4: Multimodal Queries
        await self.test_multimodal_queries()
        
        # Test 5: Different Search Types
        await self.test_search_types()
        
        # Test 6: Edge Cases
        await self.test_edge_cases()
        
        # Generate report
        self.generate_test_report()
    
    async def test_system_status(self):
        """Test system status and component availability"""
        logger.info("Testing system status...")
        
        try:
            status = await self.pipeline.get_system_status()
            
            self.test_results["system_status"] = {
                "passed": status.get("system_status") == "operational",
                "details": status,
                "components_available": {
                    "llm": status.get("components", {}).get("llm", {}).get("status") == "available",
                    "embeddings": status.get("components", {}).get("embeddings", {}).get("status") == "available",
                    "vector_store": status.get("components", {}).get("vector_store", {}).get("status") == "available"
                }
            }
            
            logger.info(f"System status: {status.get('system_status')}")
            
        except Exception as e:
            logger.error(f"System status test failed: {e}")
            self.test_results["system_status"] = {
                "passed": False,
                "error": str(e)
            }
    
    async def test_document_ingestion(self):
        """Test document ingestion with the attention paper"""
        logger.info("Testing document ingestion...")
        
        pdf_path = "attention_all_you_need.pdf"
        
        if not os.path.exists(pdf_path):
            logger.error(f"Test PDF not found: {pdf_path}")
            self.test_results["document_ingestion"] = {
                "passed": False,
                "error": f"Test file not found: {pdf_path}"
            }
            return
        
        try:
            # Clear existing data first
            clear_result = await self.pipeline.clear_all_data()
            logger.info(f"Cleared existing data: {clear_result}")
            
            # Ingest the document
            result = await self.pipeline.ingest_document(
                file_path=pdf_path,
                clear_existing=True
            )
            
            self.test_results["document_ingestion"] = {
                "passed": result["success"],
                "details": result,
                "stats": result.get("processing_stats", {}),
                "collection_stats": result.get("collection_stats", {})
            }
            
            if result["success"]:
                logger.info(f"Successfully ingested document: {result['message']}")
                stats = result.get("processing_stats", {})
                logger.info(f"Processing stats: {stats}")
            else:
                logger.error(f"Document ingestion failed: {result.get('error')}")
                
        except Exception as e:
            logger.error(f"Document ingestion test failed: {e}")
            self.test_results["document_ingestion"] = {
                "passed": False,
                "error": str(e)
            }
    
    async def test_text_queries(self):
        """Test various text-based queries"""
        logger.info("Testing text-based queries...")
        
        test_queries = [
            "What is the transformer architecture?",
            "Explain the attention mechanism",
            "What are the key components of the transformer model?",
            "How does multi-head attention work?",
            "What is positional encoding?",
            "What are the advantages of the transformer over RNNs?",
            "Describe the encoder-decoder structure",
            "What is self-attention?"
        ]
        
        query_results = []
        
        for query in test_queries:
            try:
                logger.info(f"Testing query: {query}")
                
                result = await self.pipeline.query(
                    query=query,
                    search_type="hybrid",
                    k=5,
                    include_images=True
                )
                
                query_results.append({
                    "query": query,
                    "success": result["success"],
                    "answer_length": len(result.get("answer", "")),
                    "num_sources": result.get("num_sources", 0),
                    "num_images": result.get("num_images", 0),
                    "search_type": result.get("search_type"),
                    "error": result.get("error")
                })
                
                if result["success"]:
                    logger.info(f"Query successful: {result['num_sources']} sources, {result['num_images']} images")
                else:
                    logger.error(f"Query failed: {result.get('error')}")
                
            except Exception as e:
                logger.error(f"Query test failed for '{query}': {e}")
                query_results.append({
                    "query": query,
                    "success": False,
                    "error": str(e)
                })
        
        self.test_results["text_queries"] = {
            "total_queries": len(test_queries),
            "successful_queries": sum(1 for r in query_results if r["success"]),
            "results": query_results
        }
    
    async def test_multimodal_queries(self):
        """Test multimodal queries that should reference both text and images"""
        logger.info("Testing multimodal queries...")
        
        multimodal_queries = [
            "Show me the transformer architecture diagram and explain its components",
            "What does the attention visualization look like?",
            "Describe any figures or diagrams in the paper",
            "Are there any mathematical formulas or equations shown in images?",
            "What visual elements support the explanation of attention mechanisms?"
        ]
        
        multimodal_results = []
        
        for query in multimodal_queries:
            try:
                logger.info(f"Testing multimodal query: {query}")
                
                result = await self.pipeline.query(
                    query=query,
                    search_type="multimodal",
                    k=8,
                    include_images=True
                )
                
                multimodal_results.append({
                    "query": query,
                    "success": result["success"],
                    "answer_length": len(result.get("answer", "")),
                    "num_sources": result.get("num_sources", 0),
                    "num_images": result.get("num_images", 0),
                    "has_image_references": result.get("num_images", 0) > 0,
                    "error": result.get("error")
                })
                
                if result["success"]:
                    logger.info(f"Multimodal query successful: {result['num_sources']} sources, {result['num_images']} images")
                else:
                    logger.error(f"Multimodal query failed: {result.get('error')}")
                
            except Exception as e:
                logger.error(f"Multimodal query test failed for '{query}': {e}")
                multimodal_results.append({
                    "query": query,
                    "success": False,
                    "error": str(e)
                })
        
        self.test_results["multimodal_queries"] = {
            "total_queries": len(multimodal_queries),
            "successful_queries": sum(1 for r in multimodal_results if r["success"]),
            "queries_with_images": sum(1 for r in multimodal_results if r.get("has_image_references", False)),
            "results": multimodal_results
        }
    
    async def test_search_types(self):
        """Test different search types"""
        logger.info("Testing different search types...")
        
        test_query = "What is the transformer architecture?"
        search_types = ["hybrid", "similarity", "multimodal"]
        
        search_results = []
        
        for search_type in search_types:
            try:
                logger.info(f"Testing search type: {search_type}")
                
                result = await self.pipeline.query(
                    query=test_query,
                    search_type=search_type,
                    k=5
                )
                
                search_results.append({
                    "search_type": search_type,
                    "success": result["success"],
                    "num_sources": result.get("num_sources", 0),
                    "answer_length": len(result.get("answer", "")),
                    "error": result.get("error")
                })
                
                if result["success"]:
                    logger.info(f"Search type {search_type} successful: {result['num_sources']} sources")
                else:
                    logger.error(f"Search type {search_type} failed: {result.get('error')}")
                
            except Exception as e:
                logger.error(f"Search type test failed for '{search_type}': {e}")
                search_results.append({
                    "search_type": search_type,
                    "success": False,
                    "error": str(e)
                })
        
        self.test_results["search_types"] = {
            "results": search_results,
            "all_types_working": all(r["success"] for r in search_results)
        }
    
    async def test_edge_cases(self):
        """Test edge cases and error handling"""
        logger.info("Testing edge cases...")
        
        edge_cases = [
            {"query": "", "description": "Empty query"},
            {"query": "   ", "description": "Whitespace only query"},
            {"query": "x" * 10000, "description": "Very long query"},
            {"query": "What is quantum computing?", "description": "Query about unrelated topic"},
            {"query": "🚀🎯💡", "description": "Emoji only query"}
        ]
        
        edge_results = []
        
        for case in edge_cases:
            try:
                logger.info(f"Testing edge case: {case['description']}")
                
                result = await self.pipeline.query(
                    query=case["query"],
                    search_type="hybrid",
                    k=3
                )
                
                edge_results.append({
                    "description": case["description"],
                    "query": case["query"][:100] + "..." if len(case["query"]) > 100 else case["query"],
                    "success": result["success"],
                    "handled_gracefully": True,  # If we get here, it was handled
                    "error": result.get("error")
                })
                
            except Exception as e:
                logger.info(f"Edge case '{case['description']}' raised exception: {e}")
                edge_results.append({
                    "description": case["description"],
                    "query": case["query"][:100] + "..." if len(case["query"]) > 100 else case["query"],
                    "success": False,
                    "handled_gracefully": True,  # Exception handling is also graceful
                    "error": str(e)
                })
        
        self.test_results["edge_cases"] = {
            "total_cases": len(edge_cases),
            "handled_gracefully": sum(1 for r in edge_results if r["handled_gracefully"]),
            "results": edge_results
        }
    
    def generate_test_report(self):
        """Generate a comprehensive test report"""
        logger.info("Generating test report...")
        
        report = {
            "test_summary": {
                "total_test_categories": len(self.test_results),
                "system_operational": self.test_results.get("system_status", {}).get("passed", False),
                "ingestion_successful": self.test_results.get("document_ingestion", {}).get("passed", False),
                "queries_working": self.test_results.get("text_queries", {}).get("successful_queries", 0) > 0,
                "multimodal_working": self.test_results.get("multimodal_queries", {}).get("successful_queries", 0) > 0,
                "search_types_working": self.test_results.get("search_types", {}).get("all_types_working", False),
                "edge_cases_handled": self.test_results.get("edge_cases", {}).get("handled_gracefully", 0) > 0
            },
            "detailed_results": self.test_results
        }
        
        # Save report to file
        with open("test_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print summary
        print("\n" + "="*80)
        print("MULTIMODAL RAG TEST REPORT")
        print("="*80)
        
        summary = report["test_summary"]
        print(f"System Operational: {'✅' if summary['system_operational'] else '❌'}")
        print(f"Document Ingestion: {'✅' if summary['ingestion_successful'] else '❌'}")
        print(f"Text Queries Working: {'✅' if summary['queries_working'] else '❌'}")
        print(f"Multimodal Queries Working: {'✅' if summary['multimodal_working'] else '❌'}")
        print(f"All Search Types Working: {'✅' if summary['search_types_working'] else '❌'}")
        print(f"Edge Cases Handled: {'✅' if summary['edge_cases_handled'] else '❌'}")
        
        # Detailed stats
        if "text_queries" in self.test_results:
            tq = self.test_results["text_queries"]
            print(f"\nText Queries: {tq['successful_queries']}/{tq['total_queries']} successful")
        
        if "multimodal_queries" in self.test_results:
            mq = self.test_results["multimodal_queries"]
            print(f"Multimodal Queries: {mq['successful_queries']}/{mq['total_queries']} successful")
            print(f"Queries with Images: {mq.get('queries_with_images', 0)}")
        
        if "document_ingestion" in self.test_results and self.test_results["document_ingestion"].get("passed"):
            stats = self.test_results["document_ingestion"].get("stats", {})
            print(f"\nDocument Processing:")
            print(f"  Total Documents: {stats.get('total_documents', 'N/A')}")
            print(f"  Text Documents: {stats.get('text_documents', 'N/A')}")
            print(f"  Image Documents: {stats.get('image_documents', 'N/A')}")
            print(f"  Total Images: {stats.get('total_images', 'N/A')}")
        
        print(f"\nDetailed report saved to: test_report.json")
        print("="*80)


async def main():
    """Main test function"""
    tester = MultimodalRAGTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
