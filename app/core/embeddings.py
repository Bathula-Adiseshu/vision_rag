from typing import Any, Dict, List, Optional, Union
import httpx
import base64
import io
from PIL import Image
import logging

from app.core.config import settings


logger = logging.getLogger(__name__)


class MultimodalEmbeddingsClient:
    """
    Multimodal embeddings client supporting both text and image embeddings via Jina API
    """
    
    def __init__(self):
        self.jina_url = settings.jina_embedding_url
        self.text_model = settings.jina_text_model
        self.vision_model = settings.jina_vision_model
        self.timeout = settings.embeddings_timeout_seconds
        self.batch_size = settings.embeddings_batch_size
        self.max_chars = settings.embeddings_max_chars
        
    async def _make_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make HTTP request to Jina embedding service"""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(self.jina_url, json=payload)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                logger.error(f"Embedding API error: {str(e)}")
                raise RuntimeError(f"Embedding API error: {str(e)}")
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed text chunks using Jina text embedding model
        """
        if not texts:
            return []
            
        # Truncate texts to avoid oversized payloads
        safe_texts = [
            text[:self.max_chars] if isinstance(text, str) and len(text) > self.max_chars else text 
            for text in texts
        ]
        
        vectors: List[List[float]] = []
        
        # Process in batches
        for i in range(0, len(safe_texts), self.batch_size):
            batch = safe_texts[i:i + self.batch_size]
            payload = {
                "input": batch,
                "model": self.text_model,
                "encoding_format": "float"
            }
            
            try:
                response = await self._make_request(payload)
                batch_vectors = self._extract_embeddings(response)
                vectors.extend(batch_vectors)
                logger.debug(f"Text embeddings batch {i//self.batch_size}: {len(batch)} -> {len(batch_vectors)}")
            except Exception as e:
                logger.error(f"Failed to embed text batch {i//self.batch_size}: {e}")
                # Add zero vectors as fallback
                vectors.extend([[0.0] * settings.milvus_dense_dim for _ in batch])
                
        return vectors
    
    async def embed_images(self, images: List[bytes]) -> List[List[float]]:
        """
        Embed images using Jina vision embedding model
        """
        if not images:
            return []
            
        vectors: List[List[float]] = []
        
        # Process images in batches
        for i in range(0, len(images), self.batch_size):
            batch = images[i:i + self.batch_size]
            
            # Convert images to base64
            encoded_images = []
            for img_bytes in batch:
                try:
                    # Resize and compress image if needed
                    processed_img = self._process_image(img_bytes)
                    encoded_img = base64.b64encode(processed_img).decode('utf-8')
                    encoded_images.append(f"data:image/jpeg;base64,{encoded_img}")
                except Exception as e:
                    logger.error(f"Failed to process image: {e}")
                    # Skip invalid images
                    continue
            
            if not encoded_images:
                continue
                
            payload = {
                "input": encoded_images,
                "model": self.vision_model,
                "encoding_format": "float"
            }
            
            try:
                response = await self._make_request(payload)
                batch_vectors = self._extract_embeddings(response)
                vectors.extend(batch_vectors)
                logger.debug(f"Image embeddings batch {i//self.batch_size}: {len(encoded_images)} -> {len(batch_vectors)}")
            except Exception as e:
                logger.error(f"Failed to embed image batch {i//self.batch_size}: {e}")
                # Add zero vectors as fallback
                vectors.extend([[0.0] * settings.milvus_vision_dim for _ in encoded_images])
                
        return vectors
    
    def _process_image(self, image_bytes: bytes) -> bytes:
        """
        Process image: resize and compress if needed
        """
        try:
            # Open image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize if too large
            max_size = settings.max_image_size
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            # Save as JPEG with compression
            output = io.BytesIO()
            image.save(output, format='JPEG', quality=settings.image_quality, optimize=True)
            return output.getvalue()
            
        except Exception as e:
            logger.error(f"Image processing error: {e}")
            return image_bytes  # Return original if processing fails
    
    def _extract_embeddings(self, response: Dict[str, Any]) -> List[List[float]]:
        """
        Extract embeddings from various response formats
        """
        embeddings = []
        
        if isinstance(response, dict):
            # OpenAI-style response
            if "data" in response and isinstance(response["data"], list):
                for item in response["data"]:
                    if isinstance(item, dict) and "embedding" in item:
                        embeddings.append(item["embedding"])
            # Direct embeddings response
            elif "embeddings" in response and isinstance(response["embeddings"], list):
                embeddings = response["embeddings"]
            # Nested data structure
            elif "data" in response and isinstance(response["data"], dict):
                if "embeddings" in response["data"]:
                    embeddings = response["data"]["embeddings"]
        
        return embeddings
    
    async def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text
        """
        result = await self.embed_texts([text])
        return result[0] if result else [0.0] * settings.milvus_dense_dim
    
    async def embed_multimodal_content(
        self, 
        texts: Optional[List[str]] = None, 
        images: Optional[List[bytes]] = None
    ) -> Dict[str, List[List[float]]]:
        """
        Embed both text and image content
        """
        result = {}
        
        if texts:
            result["text_embeddings"] = await self.embed_texts(texts)
        
        if images:
            result["image_embeddings"] = await self.embed_images(images)
            
        return result


# Global embeddings client instance
embeddings_client = MultimodalEmbeddingsClient()


