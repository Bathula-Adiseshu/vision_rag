"""
Multimodal Embedding Service supporting multiple providers
"""
import asyncio
import base64
import os
import uuid
from typing import List, Optional, Dict, Any, Union
import httpx
import openai
from PIL import Image
import io
import logging

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class MultimodalEmbeddingService:
    """Unified embedding service supporting multiple providers"""
    
    def __init__(self):
        self.settings = settings
        self._openai_client = None
        self._setup_clients()
    
    def _setup_clients(self):
        """Setup embedding clients based on configuration"""
        if self.settings.openai_api_key:
            self._openai_client = openai.AsyncOpenAI(
                api_key=self.settings.openai_api_key,
                base_url=self.settings.openai_base_url
            )
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts using the configured provider with fallback"""
        try:
            if self.settings.embedding_provider == "jina":
                return await self._embed_texts_jina(texts)
            elif self.settings.embedding_provider == "openai":
                return await self._embed_texts_openai(texts)
            else:
                raise ValueError(f"Unsupported embedding provider: {self.settings.embedding_provider}")
        except Exception as e:
            logger.error(f"Primary embedding provider failed: {e}")
            # Always fallback to OpenAI if primary fails
            if self.settings.embedding_provider != "openai":
                logger.info("Falling back to OpenAI embeddings")
                return await self._embed_texts_openai(texts)
            else:
                raise e
    
    async def embed_images(self, images: List[bytes]) -> List[List[float]]:
        """Embed images using the configured provider"""
        try:
            if self.settings.embedding_provider == "jina":
                return await self._embed_images_jina(images)
            elif self.settings.embedding_provider == "openai":
                return await self._embed_images_openai(images)
            else:
                raise ValueError(f"Unsupported embedding provider: {self.settings.embedding_provider}")
        except Exception as e:
            logger.error(f"Image embedding failed: {e}")
            # Fallback to OpenAI if Jina fails
            if self.settings.embedding_provider != "openai":
                logger.info("Falling back to OpenAI for image embeddings")
                return await self._fallback_to_openai_images(images)
            raise
    
    async def _embed_texts_openai(self, texts: List[str]) -> List[List[float]]:
        """Embed texts using OpenAI"""
        try:
            response = await self._openai_client.embeddings.create(
                input=texts,
                model=self.settings.openai_embedding_model
            )
            return [embedding.embedding for embedding in response.data]
        except Exception as e:
            logger.error(f"OpenAI text embedding failed: {e}")
            # Fallback to Jina
            return await self._embed_texts_jina(texts)
    
    async def _embed_images_openai(self, images: List[bytes]) -> List[List[float]]:
        """Embed images using OpenAI (fallback to text descriptions)"""
        try:
            # For OpenAI, we'll use vision model to describe images then embed descriptions
            descriptions = []
            for img_bytes in images:
                # Convert to base64
                img_b64 = base64.b64encode(img_bytes).decode('utf-8')
                
                # Use vision model to describe image
                response = await self._openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Describe this image in detail for search and retrieval purposes. Include any text, diagrams, charts, or visual elements."},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{img_b64}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=300
                )
                descriptions.append(response.choices[0].message.content)
            
            # Embed the descriptions
            return await self._embed_texts_openai(descriptions)
            
        except Exception as e:
            logger.error(f"OpenAI image embedding failed: {e}")
            # Fallback to Jina
            return await self._embed_images_jina(images)
    
    async def _embed_texts_jina(self, texts: List[str]) -> List[List[float]]:
        """Embed texts using Jina"""
        try:
            # Add retry logic for Jina service
            for attempt in range(3):
                try:
                    async with httpx.AsyncClient(timeout=30.0) as client:
                        response = await client.post(
                            self.settings.jina_embedding_url,
                            json={
                                "input": texts,
                                "model": self.settings.jina_text_model
                            },
                            timeout=60.0  # Increased timeout
                        )
                        response.raise_for_status()
                        result = response.json()
                        
                        embeddings = [item["embedding"] for item in result["data"]]
                        logger.info(f"Generated {len(embeddings)} text embeddings via Jina (attempt {attempt + 1})")
                        return embeddings
                        
                except Exception as retry_e:
                    logger.warning(f"Jina embedding attempt {attempt + 1} failed: {retry_e}")
                    if attempt < 2:  # Don't sleep on last attempt
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
            
            # All retries failed
            raise Exception("All Jina embedding attempts failed")
            
        except Exception as e:
            logger.error(f"Jina text embedding failed after retries: {e}")
            # Fallback to OpenAI
            return await self._embed_texts_openai(texts)
    
    async def _embed_images_jina(self, images: List[bytes]) -> List[List[float]]:
        """Embed images using Jina"""
        try:
            # Convert images to base64
            image_b64_list = []
            for img_bytes in images:
                img_b64 = base64.b64encode(img_bytes).decode('utf-8')
                image_b64_list.append(f"data:image/jpeg;base64,{img_b64}")
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    self.settings.jina_embedding_url,
                    json={
                        "input": image_b64_list,
                        "model": self.settings.jina_vision_model
                    }
                )
                response.raise_for_status()
                data = response.json()
                return [item["embedding"] for item in data["data"]]
        except Exception as e:
            logger.error(f"Jina image embedding failed: {e}")
            raise

class ImageStorageService:
    """Service for storing and serving images"""
    
    def __init__(self):
        self.storage_path = settings.image_storage_path
        self.base_url = settings.image_base_url
        os.makedirs(self.storage_path, exist_ok=True)
    
    def save_image(self, image_bytes: bytes, source: str, page: int, image_index: int) -> Dict[str, str]:
        """Save image and return metadata with URL"""
        # Generate unique filename
        filename = f"{source}_page_{page}_img_{image_index}_{uuid.uuid4().hex[:8]}.jpg"
        filepath = os.path.join(self.storage_path, filename)
        
        try:
            # Process and save image
            image = Image.open(io.BytesIO(image_bytes))
            if image.mode not in ('RGB', 'L'):
                image = image.convert('RGB')
            
            # Resize if too large
            max_size = settings.max_image_size
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            # Save with compression
            image.save(filepath, format='JPEG', quality=settings.image_quality, optimize=True)
            
            # Return metadata
            return {
                "filename": filename,
                "filepath": filepath,
                "url": f"{self.base_url}/{filename}",
                "width": image.size[0],
                "height": image.size[1]
            }
            
        except Exception as e:
            logger.error(f"Failed to save image: {e}")
            raise

# Global instances
embedding_service = MultimodalEmbeddingService()
image_storage = ImageStorageService()
