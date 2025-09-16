import fitz  # PyMuPDF
import os
from typing import List, Tuple, Dict, Any, Optional
from PIL import Image
import io
import base64
import logging
from dataclasses import dataclass
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.core.multimodal_embeddings import embedding_service


logger = logging.getLogger(__name__)


@dataclass
class ExtractedImage:
    """Container for extracted image data"""
    image_bytes: bytes
    page_number: int
    image_index: int
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    width: int
    height: int
    description: Optional[str] = None


@dataclass
class ProcessedPage:
    """Container for processed page data"""
    page_number: int
    text_content: str
    images: List[ExtractedImage]
    metadata: Dict[str, Any]


class MultimodalPDFProcessor:
    """
    Advanced PDF processor for extracting text and images with multimodal capabilities
    """
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    async def process_pdf(self, pdf_path: str) -> Tuple[List[Document], List[ExtractedImage]]:
        """
        Process PDF and extract both text and images
        
        Returns:
            Tuple of (text_documents, extracted_images)
        """
        try:
            doc = fitz.open(pdf_path)
            pages = []
            all_images = []
            
            # Process each page
            for page_num in range(len(doc)):
                page = doc[page_num]
                processed_page = await self._process_page(page, page_num, pdf_path)
                pages.append(processed_page)
                all_images.extend(processed_page.images)
            
            doc.close()
            
            # Create text documents
            text_documents = self._create_text_documents(pages, pdf_path)
            
            # Generate image descriptions if vision is enabled
            if settings.enable_vision and all_images:
                await self._generate_image_descriptions(all_images)
            
            logger.info(f"Processed PDF: {len(text_documents)} text chunks, {len(all_images)} images")
            return text_documents, all_images
            
        except Exception as e:
            logger.error(f"Failed to process PDF {pdf_path}: {e}")
            raise
    
    async def _process_page(self, page: fitz.Page, page_num: int, source: str) -> ProcessedPage:
        """Process a single PDF page"""
        try:
            # Extract text
            text_content = page.get_text("text")
            
            # Extract images
            images = []
            if settings.extract_images_from_pdf:
                images = await self._extract_images_from_page(page, page_num)
            
            # Page metadata
            metadata = {
                "page_number": page_num + 1,
                "source": source,
                "has_images": len(images) > 0,
                "num_images": len(images),
                "text_length": len(text_content)
            }
            
            return ProcessedPage(
                page_number=page_num + 1,
                text_content=text_content,
                images=images,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Failed to process page {page_num}: {e}")
            return ProcessedPage(
                page_number=page_num + 1,
                text_content="",
                images=[],
                metadata={"page_number": page_num + 1, "source": source, "error": str(e)}
            )
    
    async def _extract_images_from_page(self, page: fitz.Page, page_num: int) -> List[ExtractedImage]:
        """Extract images from a PDF page"""
        images = []
        
        try:
            image_list = page.get_images(full=True)
            
            for img_index, img in enumerate(image_list):
                if len(images) >= settings.max_images_per_page:
                    break
                
                try:
                    # Extract image data
                    xref = img[0]
                    base_image = page.parent.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Get image properties
                    image_ext = base_image["ext"]
                    width = base_image["width"]
                    height = base_image["height"]
                    
                    # Process image (resize, compress)
                    processed_bytes = self._process_image_bytes(image_bytes)
                    
                    # Get image bounding box (approximate)
                    bbox = (0, 0, width, height)  # Default bbox
                    
                    extracted_image = ExtractedImage(
                        image_bytes=processed_bytes,
                        page_number=page_num + 1,
                        image_index=img_index,
                        bbox=bbox,
                        width=width,
                        height=height
                    )
                    
                    images.append(extracted_image)
                    
                except Exception as e:
                    logger.warning(f"Failed to extract image {img_index} from page {page_num}: {e}")
                    continue
            
            logger.debug(f"Extracted {len(images)} images from page {page_num}")
            return images
            
        except Exception as e:
            logger.error(f"Failed to extract images from page {page_num}: {e}")
            return []
    
    def _process_image_bytes(self, image_bytes: bytes) -> bytes:
        """Process image bytes: resize and compress"""
        try:
            # Open image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if needed
            if image.mode not in ('RGB', 'L'):
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
            logger.warning(f"Image processing failed: {e}")
            return image_bytes  # Return original if processing fails
    
    def _create_text_documents(self, pages: List[ProcessedPage], source: str) -> List[Document]:
        """Create LangChain documents from processed pages"""
        documents = []
        
        for page in pages:
            if not page.text_content.strip():
                continue
            
            # Enhanced text chunking for better table and structured data capture
            chunks = self._smart_text_chunking(page.text_content, page.page_number)
            
            for chunk_idx, chunk in enumerate(chunks):
                chunk_id = f"{source}_page_{page.page_number}_chunk_{chunk_idx}"
                metadata = {
                    "source": source,
                    "page": page.page_number,
                    "chunk_id": chunk_id,
                    "content_type": "text"
                }
                
                doc = Document(
                    page_content=chunk,
                    metadata=metadata
                )
                documents.append(doc)
        
        return documents
    
    async def _generate_image_descriptions(self, images: List[ExtractedImage]) -> None:
        """Generate descriptions for images using vision model"""
        if not settings.enable_vision:
            return
        
        try:
            # For now, we'll generate simple descriptions
            # In a full implementation, you'd use a vision-language model
            for image in images:
                image.description = f"Image from page {image.page_number}, size {image.width}x{image.height}"
            
            logger.debug(f"Generated descriptions for {len(images)} images")
            
        except Exception as e:
            logger.error(f"Failed to generate image descriptions: {e}")
    
    async def create_image_documents(self, images: List[ExtractedImage], source: str) -> List[Document]:
        """Create LangChain documents from extracted images"""
        documents = []
        
        for image in images:
            try:
                # Convert image to base64 for metadata storage
                img_b64 = base64.b64encode(image.image_bytes).decode('utf-8')
                
                # Store image and get URL
                from app.core.multimodal_embeddings import image_storage
                image_info = image_storage.save_image(
                    image.image_bytes, 
                    os.path.basename(source), 
                    image.page_number, 
                    image.image_index
                )
                
                image_id = f"{source}_page_{image.page_number}_image_{image.image_index}"
                metadata = {
                    "source": source,
                    "page": image.page_number,
                    "chunk_id": image_id,
                    "content_type": "image",
                    "width": image_info["width"],
                    "height": image_info["height"],
                    "image_url": image_info["url"],
                    "image_path": image_info["filepath"]
                }
                
                # Enhanced description based on AI vision analysis
                page_content = await self._generate_enhanced_image_description(image, source)
                
                doc = Document(
                    page_content=page_content,
                    metadata=metadata
                )
                documents.append(doc)
                
            except Exception as e:
                logger.error(f"Failed to create document for image {image.image_index}: {e}")
                continue
        
        return documents
    
    async def _generate_enhanced_image_description(self, image: ExtractedImage, source: str) -> str:
        """Generate enhanced image descriptions using AI vision models"""
        from app.core.multimodal_embeddings import embedding_service
        
        try:
            # Skip AI description if no valid OpenAI key
            if not settings.openai_api_key or settings.openai_api_key.startswith('your_'):
                logger.info("Skipping AI description - no valid OpenAI API key")
                return f"Image extracted from page {image.page_number}: Technical diagram or figure"
            
            # Use vision model to generate description if available
            if settings.use_vision_llm and settings.openai_api_key:
                import openai
                base64_image = base64.b64encode(image.image_bytes).decode('utf-8')
                
                client = openai.AsyncOpenAI(
                    api_key=settings.openai_api_key,
                    base_url=settings.openai_base_url
                )
                
                response = await client.chat.completions.create(
                    model=settings.openai_vision_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Describe this image in detail, focusing on any technical diagrams, charts, tables, or visual elements that would be useful for document retrieval."
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=300
                )
                
                description = response.choices[0].message.content
                logger.info(f"Generated AI description for image: {description[:100]}...")
                return description
            
            return f"Image extracted from page {image.page_number}: Technical diagram or figure"
            
        except Exception as e:
            logger.warning(f"AI description generation failed: {e}")
            return f"Image extracted from page {image.page_number}: Technical diagram or figure"
    
    def _smart_text_chunking(self, text: str, page_num: int) -> List[str]:
        """Completely generic text chunking without hardcoded content detection"""
        chunks = []
        lines = text.split('\n')
        current_chunk = []
        current_length = 0
        
        # Dynamic chunk sizing based on content density
        base_chunk_size = settings.chunk_size
        max_chunk_size = base_chunk_size * 2
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            line_length = len(line)
            
            # Check if adding this line would exceed chunk size
            if current_length + line_length > base_chunk_size and current_chunk:
                # Check if this looks like it should stay together (high symbol density)
                symbol_density = sum(1 for c in line if c in '|=+-*()[]{}') / max(len(line), 1)
                
                if symbol_density > 0.1 and current_length + line_length <= max_chunk_size:
                    # Keep structured content together
                    current_chunk.append(line)
                    current_length += line_length
                else:
                    # Finalize current chunk
                    if current_chunk:
                        chunks.append('\n'.join(current_chunk))
                    current_chunk = [line]
                    current_length = line_length
            else:
                current_chunk.append(line)
                current_length += line_length
        
        # Add remaining chunk
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return [chunk for chunk in chunks if chunk.strip()]
    
    async def process_and_embed_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Complete pipeline: process PDF, create documents, and prepare for embedding
        """
        try:
            # Process PDF
            text_documents, extracted_images = await self.process_pdf(pdf_path)
            
            # Create image documents
            image_documents = await self.create_image_documents(extracted_images, pdf_path)
            
            # Combine all documents
            all_documents = text_documents + image_documents
            
            # Prepare embeddings data
            text_contents = [doc.page_content for doc in text_documents]
            image_bytes = [img.image_bytes for img in extracted_images]
            
            result = {
                "text_documents": text_documents,
                "image_documents": image_documents,
                "all_documents": all_documents,
                "text_contents": text_contents,
                "image_bytes": image_bytes,
                "extracted_images": extracted_images,
                "stats": {
                    "total_documents": len(all_documents),
                    "text_documents": len(text_documents),
                    "image_documents": len(image_documents),
                    "total_images": len(extracted_images)
                }
            }
            
            logger.info(f"PDF processing complete: {result['stats']}")
            return result
            
        except Exception as e:
            logger.error(f"PDF processing pipeline failed: {e}")
            raise


# Global PDF processor instance
pdf_processor = MultimodalPDFProcessor()
