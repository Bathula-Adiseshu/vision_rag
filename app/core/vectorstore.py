import logging
from typing import List, Dict, Any, Optional, Tuple
import threading
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from langchain_milvus import Milvus, BM25BuiltInFunction
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

from app.core.config import settings
from app.core.multimodal_embeddings import embedding_service


logger = logging.getLogger(__name__)


class MultimodalEmbeddings(Embeddings):
    """
    LangChain-compatible embeddings wrapper for multimodal content
    """
    
    def __init__(self):
        self.client = embedding_service
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        import asyncio
        return asyncio.run(self.client.embed_texts(texts))
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single text query"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            embeddings = loop.run_until_complete(embedding_service.embed_texts([text]))
            return embeddings[0] if embeddings else []
        except RuntimeError:
            # If no event loop is running, create a new one
            embeddings = asyncio.run(embedding_service.embed_texts([text]))
            return embeddings[0] if embeddings else []


class MilvusVectorStore:
    """
    Advanced Milvus vector store with hybrid search capabilities
    """
    
    def __init__(self):
        self.embeddings = MultimodalEmbeddings()
        self.vector_store: Optional[Milvus] = None
        self._lock = threading.Lock()
        self.collection_name = settings.milvus_collection_name
        self._setup_connection()
        self._collection = None
    
    def _setup_connection(self) -> bool:
        """Setup Milvus database connection"""
        try:
            # Remove existing connections
            if connections.has_connection("default"):
                connections.disconnect("default")
            
            # Connect to Milvus
            connections.connect(
                alias="default",
                host=settings.milvus_host,
                port=settings.milvus_port,
                user=settings.milvus_user,
                password=settings.milvus_password,
                timeout=settings.milvus_timeout
            )
            
            # Create or use database
            from pymilvus import db
            if settings.milvus_db_name not in db.list_database():
                db.create_database(settings.milvus_db_name)
            db.using_database(settings.milvus_db_name)
            
            logger.info(f"Connected to Milvus database: {settings.milvus_db_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup Milvus connection: {e}")
            return False
    
    def _create_collection_schema(self) -> CollectionSchema:
        """Create collection schema for multimodal documents"""
        fields = [
            FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=100),
            FieldSchema(name="dense", dtype=DataType.FLOAT_VECTOR, dim=settings.embedding_dimension),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="metadata", dtype=DataType.JSON)
        ]
        
        return CollectionSchema(
            fields=fields,
            description="Multimodal RAG collection with hybrid search support"
        )
    
    async def initialize_collection(self, force_recreate: bool = False) -> bool:
        """Initialize or recreate the Milvus collection"""
        try:
            with self._lock:
                if force_recreate and utility.has_collection(settings.milvus_collection_name):
                    collection = Collection(settings.milvus_collection_name)
                    collection.drop()
                    logger.info(f"Dropped existing collection: {settings.milvus_collection_name}")
                
                # Create vector store with proper dimension configuration
                self.vector_store = Milvus(
                    embedding_function=self.embeddings,
                    collection_name=settings.milvus_collection_name,
                    connection_args={
                        "host": settings.milvus_host,
                        "port": settings.milvus_port,
                        "user": settings.milvus_user,
                        "password": settings.milvus_password,
                        "db_name": settings.milvus_db_name,
                        "timeout": settings.milvus_timeout
                    },
                    consistency_level="Strong",
                    vector_field="vector",
                    text_field="text",
                    auto_id=True,
                    # Enable dynamic metadata fields (LangChain Milvus will store document metadata)
                    enable_dynamic_field=True
                )
                
                logger.info(f"Initialized collection: {settings.milvus_collection_name}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to initialize collection: {e}")
            return False
    
    async def add_documents(
        self, 
        documents: List[Document], 
        images: Optional[List[bytes]] = None,
        image_metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """
        Add multimodal documents to the vector store
        """
        if not self.vector_store:
            await self.initialize_collection()
        
        try:
            # Add text documents
            text_ids = []
            if documents:
                text_ids = await self.vector_store.aadd_documents(documents)
                logger.info(f"Added {len(documents)} text documents")
            
            # Add images if provided
            image_ids = []
            if images and settings.enable_vision:
                image_ids = await self._add_images(images, image_metadata or [])
                logger.info(f"Added {len(images)} images")
            
            return text_ids + image_ids
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise
    
    async def add_documents_with_embeddings(
        self,
        documents: List[Document]
    ) -> List[str]:
        """Add documents with proper embeddings based on content type"""
        try:
            text_docs = []
            image_docs = []
            
            # Separate text and image documents
            for doc in documents:
                if doc.metadata.get('content_type') == 'image':
                    image_docs.append(doc)
                else:
                    text_docs.append(doc)
            
            doc_ids = []
            
            # Process text documents
            if text_docs:
                logger.info(f"Processing {len(text_docs)} text documents")
                text_ids = await self.vector_store.aadd_documents(text_docs)
                doc_ids.extend(text_ids)
                logger.info(f"Added {len(text_docs)} text documents with IDs: {text_ids[:3]}...")
            
            # Process image documents
            if image_docs:
                logger.info(f"Processing {len(image_docs)} image documents")
                image_ids = await self.vector_store.aadd_documents(image_docs)
                doc_ids.extend(image_ids)
                logger.info(f"Added {len(image_docs)} image documents with IDs: {image_ids[:3]}...")
            
            logger.info(f"Successfully added {len(text_docs)} text documents and {len(image_docs)} image documents. Total IDs: {len(doc_ids)}")
            return doc_ids
            
        except Exception as e:
            logger.error(f"Failed to add documents with embeddings: {e}")
            return []
    
    async def hybrid_search(
        self,
        query: str,
        k: Optional[int] = None,
        expr: Optional[str] = None,
        include_images: bool = True,
        content_type_filter: Optional[str] = None,
        **kwargs
    ) -> List[Document]:
        """
        Perform hybrid search - includes both text and image results
        """
        if content_type_filter:
            # If specific content type is requested, use that filter
            expr = f'content_type == "{content_type_filter}"'
            return await self.similarity_search(query, k, expr, **kwargs)
        
        # Get both text and image results
        results = []
        k = k or settings.milvus_k
        
        if include_images:
            # Split k between text and images
            text_k = max(1, int(k * 0.7))  # 70% for text
            image_k = max(1, k - text_k)   # 30% for images
            
            # Search text content
            text_results = await self.similarity_search(
                query=query,
                k=text_k,
                expr='content_type == "text"',
                **kwargs
            )
            
            # Search image content
            image_results = await self.similarity_search(
                query=query,
                k=image_k,
                expr='content_type == "image"',
                **kwargs
            )
            
            # Combine results - interleave text and images
            max_len = max(len(text_results), len(image_results))
            for i in range(max_len):
                if i < len(text_results):
                    results.append(text_results[i])
                if i < len(image_results):
                    results.append(image_results[i])
        else:
            # Only search text content
            text_results = await self.similarity_search(
                query=query,
                k=k,
                expr='content_type == "text"',
                **kwargs
            )
            results.extend(text_results)
        
        return results[:k]
    
    async def similarity_search(
        self,
        query: str,
        k: Optional[int] = None,
        expr: Optional[str] = None,
        **kwargs
    ) -> List[Document]:
        """
        Perform similarity search using dense vectors only
        """
        if not self.vector_store:
            await self.initialize_collection()
        
        try:
            k = k or settings.milvus_k
            
            results = await self.vector_store.asimilarity_search(
                query=query,
                k=k,
                expr=expr,
                **kwargs
            )
            
            logger.debug(f"Similarity search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []
    
    async def multimodal_search(
        self,
        text_query: Optional[str] = None,
        image_query: Optional[bytes] = None,
        k: Optional[int] = None,
        content_type_filter: Optional[str] = None,
        include_images: bool = True,
        **kwargs
    ) -> List[Document]:
        """
        Perform multimodal search across text and images
        """
        if not text_query and not image_query:
            raise ValueError("Either text_query or image_query must be provided")
        
        # Use hybrid search which now handles multimodal content
        return await self.hybrid_search(
            query=text_query or "multimodal content",
            k=k,
            include_images=include_images,
            content_type_filter=content_type_filter,
            **kwargs
        )
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            from pymilvus import utility
            if not utility.has_collection(settings.milvus_collection_name):
                return {"exists": False}
            
            collection = Collection(settings.milvus_collection_name)
            collection.load()
            
            return {
                "exists": True,
                "num_entities": collection.num_entities,
                "collection_name": settings.milvus_collection_name,
                "schema": str(collection.schema)
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"exists": False, "error": str(e)}
    
    async def delete_collection(self) -> bool:
        """Delete the collection"""
        try:
            from pymilvus import utility
            if utility.has_collection(settings.milvus_collection_name):
                collection = Collection(settings.milvus_collection_name)
                collection.drop()
                logger.info(f"Deleted collection: {settings.milvus_collection_name}")
            
            self.vector_store = None
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            return False
    
    def _get_collection(self) -> Optional[Collection]:
        """Get or create Milvus collection"""
        if self._collection is None:
            if utility.has_collection(self.collection_name):
                self._collection = Collection(self.collection_name)
                self._collection.load()
            else:
                logger.warning(f"Collection {self.collection_name} does not exist")
                return None
        return self._collection


# Global vector store instance
vector_store = MilvusVectorStore()
