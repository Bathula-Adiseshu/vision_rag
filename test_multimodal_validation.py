#!/usr/bin/env python3
"""
Comprehensive validation script for multimodal RAG with image and table queries
"""
import asyncio
import json
import logging
from typing import List, Dict, Any
from app.core.rag_pipeline import rag_pipeline
from app.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultimodalRAGValidator:
    """Validate multimodal RAG system with specific focus on image and table queries"""
    
    def __init__(self):
        self.test_results = {}
        
    async def run_comprehensive_validation(self):
        """Run comprehensive validation tests"""
        print("🔍 MULTIMODAL RAG VALIDATION - ATTENTION TRANSFORMER PAPER")
        print("=" * 70)
        
        # First, ingest the document
        await self.ingest_document()
        
        # Test image-based queries
        await self.test_image_queries()
        
        # Test table-based queries  
        await self.test_table_queries()
        
        # Test architecture diagram queries
        await self.test_architecture_queries()
        
        # Test multimodal context validation
        await self.validate_multimodal_context()
        
        # Generate comprehensive report
        self.generate_validation_report()
        
    async def ingest_document(self):
        """Ingest the attention paper"""
        print("\n📄 INGESTING ATTENTION_ALL_YOU_NEED.PDF...")
        
        result = await rag_pipeline.ingest_document(
            "attention_all_you_need.pdf",
            clear_existing=True
        )
        
        print(f"✅ Ingestion: {result['message']}")
        print(f"📊 Stats: {result['processing_stats']}")
        self.test_results['ingestion'] = result
        
    async def test_image_queries(self):
        """Test queries specifically targeting images in the transformer paper"""
        print("\n🖼️ TESTING IMAGE-BASED QUERIES...")
        
        image_queries = [
            "Show me the transformer architecture diagram",
            "What does the multi-head attention visualization look like?",
            "Display the encoder-decoder structure diagram",
            "Show me any performance comparison charts or graphs",
            "What figures illustrate the attention mechanism?",
            "Find diagrams showing the model architecture",
            "Show me visual representations of the transformer model"
        ]
        
        image_results = {}
        
        for query in image_queries:
            print(f"\n🔍 Query: {query}")
            
            result = await rag_pipeline.query(
                query=query,
                search_type="hybrid",
                k=10,
                include_images=True
            )
            
            # Analyze result for image content
            image_count = len([s for s in result.get('sources', []) if s.get('content_type') == 'image'])
            total_sources = len(result.get('sources', []))
            
            print(f"   📊 Sources: {total_sources} total, {image_count} images")
            print(f"   🤖 Answer length: {len(result.get('answer', ''))}")
            print(f"   ✅ Success: {result.get('success', False)}")
            
            # Check if images were actually used in LLM context
            used_vision_llm = result.get('used_vision_llm', False)
            images_in_context = result.get('images_in_context', 0)
            
            print(f"   👁️ Vision LLM used: {used_vision_llm}")
            print(f"   🖼️ Images in context: {images_in_context}")
            
            if result.get('sources'):
                print("   📋 Image sources found:")
                for i, source in enumerate(result['sources'][:3]):
                    if source.get('content_type') == 'image':
                        print(f"      {i+1}. Page {source.get('page')}: {source.get('content', '')[:100]}...")
            
            image_results[query] = {
                'total_sources': total_sources,
                'image_sources': image_count,
                'answer_length': len(result.get('answer', '')),
                'used_vision_llm': used_vision_llm,
                'images_in_context': images_in_context,
                'success': result.get('success', False),
                'answer': result.get('answer', '')[:200] + "..." if len(result.get('answer', '')) > 200 else result.get('answer', '')
            }
        
        self.test_results['image_queries'] = image_results
        
    async def test_table_queries(self):
        """Test queries targeting tables and numerical data"""
        print("\n📊 TESTING TABLE-BASED QUERIES...")
        
        table_queries = [
            "Show me performance comparison tables",
            "What are the BLEU scores in the results table?",
            "Display training time comparisons",
            "Show me the model size and parameter counts",
            "What are the performance metrics in tabular format?",
            "Find tables showing experimental results",
            "Show me computational complexity comparisons"
        ]
        
        table_results = {}
        
        for query in table_queries:
            print(f"\n🔍 Query: {query}")
            
            result = await rag_pipeline.query(
                query=query,
                search_type="hybrid", 
                k=8,
                include_images=True
            )
            
            # Look for table-related content
            sources = result.get('sources', [])
            table_indicators = ['table', 'score', 'metric', 'comparison', 'result', 'performance']
            table_relevant_sources = 0
            
            for source in sources:
                content = source.get('content', '').lower()
                if any(indicator in content for indicator in table_indicators):
                    table_relevant_sources += 1
            
            print(f"   📊 Total sources: {len(sources)}")
            print(f"   📋 Table-relevant sources: {table_relevant_sources}")
            print(f"   🤖 Answer: {result.get('answer', '')[:150]}...")
            
            table_results[query] = {
                'total_sources': len(sources),
                'table_relevant_sources': table_relevant_sources,
                'answer_length': len(result.get('answer', '')),
                'success': result.get('success', False),
                'answer': result.get('answer', '')[:200] + "..." if len(result.get('answer', '')) > 200 else result.get('answer', '')
            }
        
        self.test_results['table_queries'] = table_results
        
    async def test_architecture_queries(self):
        """Test queries about transformer architecture with visual elements"""
        print("\n🏗️ TESTING ARCHITECTURE DIAGRAM QUERIES...")
        
        arch_queries = [
            "Explain the transformer architecture using the diagram",
            "How does the encoder-decoder structure work based on the visual?",
            "Describe the multi-head attention mechanism from the figure",
            "What does the positional encoding diagram show?",
            "Explain the feed-forward network structure visually"
        ]
        
        arch_results = {}
        
        for query in arch_queries:
            print(f"\n🔍 Query: {query}")
            
            result = await rag_pipeline.query(
                query=query,
                search_type="multimodal",
                k=8,
                include_images=True
            )
            
            # Check multimodal integration
            image_count = len([s for s in result.get('sources', []) if s.get('content_type') == 'image'])
            text_count = len([s for s in result.get('sources', []) if s.get('content_type') == 'text'])
            
            print(f"   📊 Text sources: {text_count}, Image sources: {image_count}")
            print(f"   👁️ Vision LLM: {result.get('used_vision_llm', False)}")
            print(f"   🤖 Answer quality: {'Good' if len(result.get('answer', '')) > 100 else 'Short'}")
            
            arch_results[query] = {
                'text_sources': text_count,
                'image_sources': image_count,
                'used_vision_llm': result.get('used_vision_llm', False),
                'answer_length': len(result.get('answer', '')),
                'success': result.get('success', False)
            }
        
        self.test_results['architecture_queries'] = arch_results
        
    async def validate_multimodal_context(self):
        """Validate that images are properly appended to LLM context"""
        print("\n🔍 VALIDATING MULTIMODAL CONTEXT INTEGRATION...")
        
        # Test with a specific query that should trigger vision LLM
        test_query = "Describe in detail what you see in the transformer architecture diagram"
        
        print(f"Test query: {test_query}")
        
        result = await rag_pipeline.query(
            query=test_query,
            search_type="hybrid",
            k=5,
            include_images=True
        )
        
        # Detailed analysis
        sources = result.get('sources', [])
        image_sources = [s for s in sources if s.get('content_type') == 'image']
        
        print(f"\n📊 CONTEXT VALIDATION RESULTS:")
        print(f"   Total sources retrieved: {len(sources)}")
        print(f"   Image sources found: {len(image_sources)}")
        print(f"   Vision LLM used: {result.get('used_vision_llm', False)}")
        print(f"   Images in context: {result.get('images_in_context', 0)}")
        
        if image_sources:
            print(f"\n🖼️ IMAGE SOURCES DETAILS:")
            for i, img_source in enumerate(image_sources):
                print(f"   {i+1}. Page {img_source.get('page')}: {img_source.get('content', '')}")
                print(f"      Image path: {img_source.get('image_path', 'N/A')}")
                print(f"      Image URL: {img_source.get('image_url', 'N/A')}")
        
        print(f"\n🤖 LLM RESPONSE:")
        print(f"   {result.get('answer', 'No answer generated')}")
        
        self.test_results['context_validation'] = {
            'total_sources': len(sources),
            'image_sources_count': len(image_sources),
            'used_vision_llm': result.get('used_vision_llm', False),
            'images_in_context': result.get('images_in_context', 0),
            'answer': result.get('answer', ''),
            'image_sources': [
                {
                    'page': img.get('page'),
                    'content': img.get('content', ''),
                    'image_path': img.get('image_path'),
                    'image_url': img.get('image_url')
                }
                for img in image_sources
            ]
        }
        
    def generate_validation_report(self):
        """Generate comprehensive validation report"""
        print("\n" + "=" * 70)
        print("📋 MULTIMODAL RAG VALIDATION REPORT")
        print("=" * 70)
        
        # Summary statistics
        total_image_queries = len(self.test_results.get('image_queries', {}))
        successful_image_queries = sum(1 for r in self.test_results.get('image_queries', {}).values() if r['image_sources'] > 0)
        
        total_table_queries = len(self.test_results.get('table_queries', {}))
        successful_table_queries = sum(1 for r in self.test_results.get('table_queries', {}).values() if r['table_relevant_sources'] > 0)
        
        print(f"\n📊 SUMMARY STATISTICS:")
        print(f"   Image queries with images found: {successful_image_queries}/{total_image_queries} ({successful_image_queries/total_image_queries*100:.1f}%)")
        print(f"   Table queries with relevant content: {successful_table_queries}/{total_table_queries} ({successful_table_queries/total_table_queries*100:.1f}%)")
        
        # Context validation summary
        context_val = self.test_results.get('context_validation', {})
        print(f"   Vision LLM integration: {'✅ Working' if context_val.get('used_vision_llm') else '❌ Not working'}")
        print(f"   Images in LLM context: {context_val.get('images_in_context', 0)}")
        
        # Save detailed report
        with open('multimodal_validation_report.json', 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        print(f"\n💾 Detailed report saved to: multimodal_validation_report.json")
        
        # Recommendations
        print(f"\n🔧 RECOMMENDATIONS:")
        if successful_image_queries < total_image_queries * 0.8:
            print("   - Improve image embedding and retrieval accuracy")
        if not context_val.get('used_vision_llm'):
            print("   - Fix vision LLM integration for image context")
        if context_val.get('images_in_context', 0) == 0:
            print("   - Fix image path resolution for LLM context")
        
        print("\n✅ Validation complete!")

async def main():
    """Run the validation"""
    validator = MultimodalRAGValidator()
    await validator.run_comprehensive_validation()

if __name__ == "__main__":
    asyncio.run(main())
