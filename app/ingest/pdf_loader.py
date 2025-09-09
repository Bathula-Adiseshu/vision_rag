from typing import List

import fitz  # pymupdf
from pypdf import PdfReader
from PIL import Image
import io


class PDFPage:
    def __init__(self, page_number: int, text: str, images: List[bytes]):
        self.page_number = page_number
        self.text = text
        self.images = images


def _extract_text_pymupdf(path: str) -> List[str]:
    texts: List[str] = []
    doc = fitz.open(path)
    try:
        for page in doc:
            texts.append(page.get_text("text") or "")
    finally:
        doc.close()
    return texts


def _extract_text_pypdf(path: str) -> List[str]:
    texts: List[str] = []
    reader = PdfReader(path)
    for p in reader.pages:
        try:
            t = p.extract_text() or ""
        except Exception:
            t = ""
        texts.append(t)
    return texts


def extract_pdf(path: str) -> List[PDFPage]:
    pages: List[PDFPage] = []

    # Gather text from both engines and merge per page
    mu_texts = _extract_text_pymupdf(path)
    py_texts = _extract_text_pypdf(path)
    max_pages = max(len(mu_texts), len(py_texts))

    # Open once for images
    doc = fitz.open(path)
    try:
        for i in range(max_pages):
            text_mu = mu_texts[i] if i < len(mu_texts) else ""
            text_py = py_texts[i] if i < len(py_texts) else ""
            merged_text = (text_mu.strip() + "\n" + text_py.strip()).strip()

            image_bytes: List[bytes] = []
            if i < len(doc):
                page = doc[i]
                for img in page.get_images(full=True):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    img_bytes = base_image.get("image")
                    if img_bytes:
                        image_bytes.append(img_bytes)

            pages.append(PDFPage(page_number=i + 1, text=merged_text, images=image_bytes))
    finally:
        doc.close()

    return pages


