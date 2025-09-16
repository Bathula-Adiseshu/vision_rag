import logging
from typing import List, Dict, Any, Optional, Tuple
import asyncio
import os
import base64
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config import settings
from app.core.multimodal_embeddings import embedding_service
from app.core.vectorstore import vector_store
from app.core.pdf_processor import pdf_processor
from app.core.llms import get_llm


logger = logging.getLogger(__name__)


class MultimodalRAGPipeline:
    """
    Complete multimodal RAG pipeline integrating all components
    """
    
    def __init__(self):
        try:
            self.llm = get_llm()
        except Exception as e:
            logger.warning(f"LLM initialization failed: {e}")
            self.llm = None
        self.vector_store = vector_store
        self.pdf_processor = pdf_processor
        self.embeddings_client = embedding_service
    
    async def ingest_document(
        self, 
        file_path: str, 
        clear_existing: bool = False
    ) -> Dict[str, Any]:
        """
        Ingest a document (PDF) into the RAG system
        """
        try:
            logger.info(f"Starting document ingestion: {file_path}")
            
            # Initialize vector store
            if clear_existing:
                logger.info("Clearing existing collection...")
                await self.vector_store.initialize_collection(force_recreate=True)
            else:
                # Always ensure collection is initialized
                await self.vector_store.initialize_collection(force_recreate=False)
            
            # Process PDF
            result = await self.pdf_processor.process_and_embed_pdf(file_path)
            
            # Add documents to vector store with proper embeddings
            document_ids = await self.vector_store.add_documents_with_embeddings(result["all_documents"])
            
            # Get collection stats
            stats = self.vector_store.get_collection_stats()
            
            ingestion_result = {
                "success": True,
                "file_path": file_path,
                "document_ids": document_ids,
                "processing_stats": result["stats"],
                "collection_stats": stats,
                "message": f"Successfully ingested {len(result['all_documents'])} documents"
            }
            
            logger.info(f"Document ingestion complete: {ingestion_result['message']}")
            return ingestion_result
            
        except Exception as e:
            logger.error(f"Document ingestion failed: {e}")
            return {
                "success": False,
                "file_path": file_path,
                "error": str(e),
                "message": f"Failed to ingest document: {str(e)}"
            }
    
    async def query(
        self,
        query: str,
        search_type: str = "hybrid",
        k: int = 5,
        include_images: bool = True,
        content_type_filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process a query using the RAG pipeline"""
        try:
            logger.info(f"Processing query: {query}...")

            # Convert content type filter to Milvus expression
            filter_expr = None
            if content_type_filter:
                filter_expr = self._build_filter_expression(content_type_filter)

            # Perform search
            results = await self.vector_store.hybrid_search(
                query=query,
                k=k,
                expr=filter_expr,
                include_images=include_images
            )

            if not results:
                return {
                    "success": True,
                    "query": query,
                    "answer": "I couldn't find any relevant information to answer your question.",
                    "sources": [],
                    "context_used": "",
                    "search_type": search_type
                }
            
            # Prepare context
            context_parts = []
            sources = []
            image_references = []
            
            # Process retrieved documents and extract image references
            for i, doc in enumerate(results):
                metadata = doc.metadata
                content_type = metadata.get("content_type", "text")
                
                if content_type == "text":
                    context_parts.append(f"[Source {i+1}] {doc.page_content}")
                    sources.append({
                        "index": i+1,
                        "type": "text",
                        "page": metadata.get("page"),
                        "source": metadata.get("source"),
                        "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                        "description": None,
                        "width": None,
                        "height": None
                    })
                elif content_type == "image" and include_images:
                    # Extract image information from metadata
                    description = doc.page_content
                    image_path = metadata.get("image_path")
                    image_url = metadata.get("image_url")
                    
                    # Resolve an image path if not present in metadata
                    resolved_path = None
                    if image_path and os.path.exists(image_path):
                        resolved_path = image_path
                    else:
                        # Try common alternatives and pattern search in uploads/images
                        candidates = []
                        alt_filepath = metadata.get("filepath")
                        if alt_filepath:
                            candidates.append(alt_filepath)
                        if image_url:
                            candidates.append(image_url.replace('http://localhost:8000/api/images/', 'uploads/images/'))
                        source_name = os.path.basename(metadata.get("source", ""))
                        page_num = metadata.get("page")
                        image_dir = 'uploads/images'
                        if source_name and page_num is not None and os.path.exists(image_dir):
                            for f in os.listdir(image_dir):
                                if f.startswith(f"{source_name}_page_{page_num}_img_"):
                                    candidates.append(os.path.join(image_dir, f))
                        for cand in candidates:
                            if cand and os.path.exists(cand):
                                resolved_path = cand
                                break
                        if not resolved_path:
                            logger.warning(f"Image path could not be resolved for source={metadata.get('source')} page={metadata.get('page')}")
                    
                    # Add to context with description
                    context_parts.append(f"[Image {i+1}] {description}")
                    
                    # Create image reference for LLM with actual or resolved image path
                    image_ref = {
                        "index": i+1,
                        "page": metadata.get("page"),
                        "source": metadata.get("source"),
                        "description": description,
                        "width": metadata.get("width"),
                        "height": metadata.get("height"),
                        "image_url": image_url,
                        "image_path": resolved_path,
                    }
                    
                    if resolved_path:
                        image_references.append(image_ref)
                        logger.info(f"Added image reference: {resolved_path}")

                    sources.append({
                        "index": i+1,
                        "type": "image",
                        "page": metadata.get("page"),
                        "source": metadata.get("source"),
                        "content": description,
                        "description": description,
                        "width": metadata.get("width"),
                        "height": metadata.get("height"),
                        "image_url": image_url
                    })
            
            context = "\n\n".join(context_parts)

            # Fallback: if no image references resolved but images are requested, try to find plausible images from uploads
            if include_images and not image_references:
                try:
                    image_dir = 'uploads/images'
                    if os.path.exists(image_dir):
                        # Prefer images belonging to the same source if available in text sources
                        preferred_sources = list({s.get('source') for s in sources if s.get('source')})
                        candidates = []
                        for fname in os.listdir(image_dir):
                            if not fname.endswith('.jpg'):
                                continue
                            if preferred_sources and not any(os.path.basename(ps or '') in fname for ps in preferred_sources):
                                continue
                            candidates.append(os.path.join(image_dir, fname))
                        # If still empty, take any images
                        if not candidates:
                            candidates = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')]
                        # Use up to max_images_in_context
                        for idx, path in enumerate(sorted(candidates)[:settings.max_images_in_context]):
                            image_references.append({
                                "index": idx + 1,
                                "page": None,
                                "source": None,
                                "description": "Auto-discovered image from document",
                                "width": None,
                                "height": None,
                                "image_url": None,
                                "image_path": path,
                            })
                        if image_references:
                            logger.info(f"Fallback discovered {len(image_references)} images from uploads directory")
                except Exception as e:
                    logger.warning(f"Fallback image discovery failed: {e}")
            
            # Generate answer using LLM
            answer = await self._generate_answer(query, context, image_references)
            used_vision = bool(image_references and settings.use_vision_llm and settings.openai_api_key)
            
            result = {
                "success": True,
                "query": query,
                "answer": answer,
                "sources": sources,
                "context_used": context[:1000] + "..." if len(context) > 1000 else context,
                "search_type": search_type,
                "num_sources": len(sources),
                "num_images": len(image_references),
                "used_vision_llm": used_vision,
                "images_in_context": len(image_references)
            }
            
            logger.info(f"Query processed successfully: {len(sources)} sources, {len(image_references)} images")
            return result
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {
                "success": False,
                "query": query,
                "error": str(e),
                "message": f"Failed to process query: {str(e)}"
            }
    
    async def _generate_answer(
        self,
        query: str,
        context: str,
        image_references: List[Dict[str, Any]]
    ) -> str:
        """Generate answer using vision-capable LLM with image context"""
        if not self.llm:
            return "LLM service is not available."
        
        try:
            # Check if we have images and vision LLM is enabled
            if image_references and settings.use_vision_llm and settings.openai_api_key:
                logger.info(f"Using vision LLM with {len(image_references)} images")
                return await self._generate_vision_answer(query, context, image_references)
            else:
                logger.info(f"Using text-only LLM (images: {len(image_references)}, vision_enabled: {settings.use_vision_llm})")
                return await self._generate_text_answer(query, context, image_references)
                
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return f"I encountered an error while generating the answer: {str(e)}"
    
    async def _generate_vision_answer(
        self,
        query: str,
        context: str,
        image_references: List[Dict[str, Any]]
    ) -> str:
        """Generate answer using vision LLM with actual images"""
        try:
            import openai
            client = openai.AsyncOpenAI(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url
            )
            
            # Prepare messages with text and images
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant that can analyze both text and images. Provide comprehensive answers based on the provided context and images."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Query: {query}\n\nContext: {context}\n\nPlease analyze the provided images along with the text context to answer the query comprehensively."}
                    ]
                }
            ]
            
            # Resolve image paths for vision LLM from references
            image_paths = []
            for ref in image_references:
                # Start with any directly provided path
                candidates = [
                    ref.get('image_path'),
                    ref.get('image_url', '').replace('http://localhost:8000/api/images/', 'uploads/images/') if ref.get('image_url') else None
                ]
                # Also try pattern-based search using source/page
                source_file = ref.get('source', '')
                page = ref.get('page')
                if source_file and page is not None:
                    image_dir = 'uploads/images'
                    if os.path.exists(image_dir):
                        source_name = os.path.basename(source_file)
                        for f in os.listdir(image_dir):
                            if f.startswith(f"{source_name}_page_{page}_img_"):
                                candidates.append(os.path.join(image_dir, f))
                found = False
                for path in candidates:
                    if path and os.path.exists(path):
                        image_paths.append(path)
                        found = True
                        break
                if not found:
                    logger.warning(f"No image file found for source: {source_file}, page: {page}")
            
            # Add images to the message (limit to max_images_in_context)
            image_count = 0
            for img_path in image_paths[:settings.max_images_in_context]:
                try:
                    if img_path and os.path.exists(img_path):
                        with open(img_path, 'rb') as f:
                            img_bytes = f.read()
                        img_b64 = base64.b64encode(img_bytes).decode('utf-8')
                        
                        messages[1]["content"].append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_b64}"
                            }
                        })
                        image_count += 1
                        logger.info(f"Successfully loaded image from: {img_path}")
                    else:
                        logger.warning(f"Image path does not exist: {img_path}")
                except Exception as e:
                    logger.warning(f"Failed to load image for vision LLM: {e}")
            print("messages:", messages["content"])
            response = await client.chat.completions.create(
                model=settings.openai_vision_model,
                messages=messages,
                max_tokens=1500
            )
            
            logger.info(f"Vision LLM processed {image_count} images for query: {query[:50]}...")
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Vision answer generation failed: {e}")
            # Fallback to OpenAI if DeepSeek fails
            if settings.openai_api_key and not settings.openai_api_key.startswith('your_'):
                try:
                    logger.info("Falling back to OpenAI for vision LLM")
                    import openai
                    openai_client = openai.AsyncOpenAI(
                        api_key=settings.openai_api_key,
                        base_url=settings.openai_base_url
                    )
                    
                    response = await openai_client.chat.completions.create(
                        model=settings.openai_vision_model,
                        messages=messages,
                        max_tokens=1500
                    )
                    
                    logger.info(f"OpenAI Vision LLM processed images")
                    return response.choices[0].message.content
                    
                except Exception as openai_e:
                    logger.error(f"OpenAI vision fallback failed: {openai_e}")
            
            # Final fallback to text-only answer
            return await self._generate_text_answer(query, context, image_references)
    
    async def _generate_text_answer(
        self,
        query: str,
        context: str,
        image_references: List[Dict[str, Any]]
    ) -> str:
        """Generate answer using text-only LLM"""
        try:
            # Prepare prompt with context and image descriptions
            prompt_parts = [
                f"Query: {query}",
                f"Context: {context}"
            ]
            
            if image_references:
                image_info = "\n".join([
                    f"Image {ref['index']}: {ref['description']} (Page {ref['page']})"
                    for ref in image_references
                ])
                prompt_parts.append(f"Images: {image_info}")
            
            prompt_parts.append(
                "Please provide a comprehensive answer based on the context and any referenced images. "
                "If images are mentioned, incorporate their content into your response."
            )
            
            prompt = "\n\n".join(prompt_parts)
            
            response = await self.llm.ainvoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
            
        except Exception as e:
            logger.error(f"Text answer generation failed: {e}")
            return f"I encountered an error while generating the answer: {str(e)}"
    
    async def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status
        """
        try:
            # Vector store stats
            collection_stats = self.vector_store.get_collection_stats()
            
            # LLM status
            llm_status = "available" if self.llm else "unavailable"
            
            # Embeddings status
            embeddings_status = "available"
            try:
                test_embedding = await self.embeddings_client.embed_query("test")
                if not test_embedding:
                    embeddings_status = "unavailable"
            except:
                embeddings_status = "unavailable"
            
            return {
                "system_status": "operational",
                "components": {
                    "llm": {
                        "status": llm_status,
                        "provider": settings.llm_provider,
                        "model": settings.deepseek_model if settings.llm_provider == "deepseek" else settings.openai_model
                    },
                    "embeddings": {
                        "status": embeddings_status,
                        "service_url": settings.jina_embedding_url,
                        "text_model": settings.jina_text_model,
                        "vision_model": settings.jina_vision_model
                    },
                    "vector_store": {
                        "status": "available" if collection_stats.get("exists") else "unavailable",
                        "collection_stats": collection_stats
                    }
                },
                "configuration": {
                    "multimodal_enabled": settings.enable_vision,
                    "hybrid_search_enabled": settings.enable_hybrid,
                    "chunk_size": settings.chunk_size,
                    "top_k_retrieval": settings.top_k_retrieval
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {
                "system_status": "error",
                "error": str(e)
            }
    
    async def clear_all_data(self) -> Dict[str, Any]:
        """
        Clear all data from the system
        """
        try:
            success = await self.vector_store.delete_collection()
            return {
                "success": success,
                "message": "All data cleared successfully" if success else "Failed to clear data"
            }
        except Exception as e:
            logger.error(f"Failed to clear data: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _build_filter_expression(self, content_type_filter: Dict[str, Any]) -> str:
        """Convert content type filter to Milvus expression"""
        expressions = []
        for key, value in content_type_filter.items():
            if isinstance(value, (list, tuple)):
                expressions.append(f"metadata.{key} in {value}")
            else:
                expressions.append(f"metadata.{key} == '{value}'")
        return " && ".join(expressions) if expressions else None


# Global RAG pipeline instance
rag_pipeline = MultimodalRAGPipeline()
