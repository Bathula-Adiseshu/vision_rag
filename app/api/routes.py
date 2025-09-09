from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Body
from pydantic import BaseModel
import os
import tempfile
import httpx
from pymilvus.exceptions import MilvusException

from app.core.config import get_settings
from app.core.embeddings import JinaEmbeddingsClient
from app.core.llms import get_llm
from app.ingest.pdf_loader import extract_pdf
from app.ingest.chunking import sliding_window_chunks
from app.vectorstores.milvus_store import MilvusVectorStore
from app.utils.logging import get_logger
from app.retrievers.qa import answer_from_context
from app.retrievers.hybrid import HybridRanker


router = APIRouter()
logger = get_logger(__name__)


@router.post("/ingest")
async def ingest_pdf_endpoint(path: Optional[str] = None, file: Optional[UploadFile] = File(None), clear: bool = True) -> Dict[str, Any]:
    s = get_settings()
    tmp_path: Optional[str] = None
    if file is not None:
        # Persist upload to a temp file
        try:
            suffix = os.path.splitext(file.filename or "uploaded.pdf")[1] or ".pdf"
            fd, tmp_path = tempfile.mkstemp(prefix="ingest_", suffix=suffix)
            with os.fdopen(fd, "wb") as out:
                out.write(await file.read())
            pdf_path = tmp_path
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to save uploaded file: {e}")
    else:
        pdf_path = path or "biology_textbook.pdf"

    try:
        pages = extract_pdf(pdf_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read PDF: {e}")

    chunks: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    total_images = 0
    for p in pages:
        has_img = len(p.images) > 0
        total_images += len(p.images)
        for ch in sliding_window_chunks(p.text, s.chunk_size, s.chunk_overlap):
            chunks.append(ch)
            metadatas.append({
                "page": p.page_number,
                "source": pdf_path,
                "has_image": has_img,
                "num_images_on_page": len(p.images),
            })

    embedder = JinaEmbeddingsClient()
    # Filter out empty/short chunks before embedding
    filtered: List[tuple[str, Dict[str, Any]]] = [
        (ch.strip(), md) for ch, md in zip(chunks, metadatas) if ch and ch.strip()
    ]
    if not filtered:
        raise HTTPException(status_code=400, detail="No textual content extracted from PDF.")

    filt_chunks = [t[0] for t in filtered]
    filt_metas = [t[1] for t in filtered]

    try:
        vectors = await embedder.embed_texts(filt_chunks)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=503, detail=f"Embeddings service unavailable: {e}")

    # Ensure alignment in case the embeddings service returns fewer vectors
    n = min(len(filt_chunks), len(vectors), len(filt_metas))
    if n == 0:
        raise HTTPException(
            status_code=503,
            detail=(
                "No embeddings produced for the extracted content. Ensure embeddings server is running at "
                f"{get_settings().embeddings_base_url} and model {get_settings().embeddings_model} is available."
            ),
        )

    try:
        ids = [f"{i}" for i in range(n)]
        store = MilvusVectorStore()
        if clear:
            # Drop previous data as requested
            store.drop()
            # Re-create collection
            store = MilvusVectorStore()
        inserted = store.upsert(ids=ids, embeddings=vectors[:n], texts=filt_chunks[:n], metadatas=filt_metas[:n])
    except MilvusException as e:
        raise HTTPException(status_code=503, detail=f"Milvus unavailable: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
    logger.info(f"Ingested {len(chunks)} chunks; pages={len(pages)}; total_images={total_images}")
    return {"inserted": inserted, "chunks": len(chunks), "pages": len(pages), "total_images": total_images}


class QueryRequest(BaseModel):
    query: str
    k: Optional[int] = None


@router.post("/query")
async def query_endpoint(query: Optional[str] = None, k: Optional[int] = None, body: Optional[QueryRequest] = Body(None)) -> Dict[str, Any]:
    s = get_settings()
    user_query = query or (body.query if body else None)
    if not user_query or not user_query.strip():
        raise HTTPException(status_code=422, detail="Query text is required")
    top_k = k if k is not None else (body.k if body and body.k is not None else 8)
    # bounds
    top_k = max(1, min(50, top_k))

    embedder = JinaEmbeddingsClient()
    qvec = await embedder.embed_query(user_query)
    if not qvec:
        raise HTTPException(status_code=503, detail="Embeddings service returned empty vector for query")

    store = MilvusVectorStore()
    try:
        hits = store.search([qvec], k=max(20, top_k))
    except MilvusException as e:
        raise HTTPException(status_code=503, detail=f"Milvus unavailable: {e}")

    if not hits:
        logger.info("Query returned 0 hits")
        return {"answer": "I don't know based on the current index.", "hits": [], "images_in_hits": 0}

    texts = [h.get("text", "") for h in hits]
    ranker = HybridRanker(texts)
    dense_scores = [1.0 - min(1.0, h.get("distance", 1.0)) for h in hits]
    blended = ranker.blend(dense=dense_scores, query=user_query, alpha=s.hybrid_alpha)
    reranked = sorted(zip(hits, blended), key=lambda x: x[1], reverse=True)[:top_k]
    context = "\n\n".join([h[0].get("text", "") for h in reranked])
    answer = answer_from_context(context=context, question=user_query)
    # Image metrics in top-k
    top_hits = [h for h, _ in reranked]
    imgs_in_hits = sum(1 for h in top_hits if (h.get("metadata") or {}).get("has_image"))
    img_pages = sorted({(h.get("metadata") or {}).get("page") for h in top_hits if (h.get("metadata") or {}).get("has_image")})
    logger.info(f"Query images_in_hits={imgs_in_hits} pages_with_images={img_pages}")
    return {"answer": answer, "hits": top_hits, "images_in_hits": imgs_in_hits, "image_pages": img_pages}


@router.post("/reindex")
async def reindex_endpoint() -> Dict[str, Any]:
    store = MilvusVectorStore()
    store.drop()
    return {"status": "dropped"}


@router.get("/stats")
async def stats_endpoint() -> Dict[str, Any]:
    return {"status": "ok"}


