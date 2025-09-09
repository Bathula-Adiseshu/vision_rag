from typing import Any, Dict, List, Optional
import httpx

from app.core.config import get_settings
from app.utils.logging import get_logger
from app.utils.http import AsyncHttpClient


class JinaEmbeddingsClient:
    def __init__(self):
        s = get_settings()
        self._client = AsyncHttpClient(base_url=s.embeddings_base_url, timeout_seconds=s.embeddings_timeout_seconds)
        self._model = s.embeddings_model
        self._batch_size = 64
        self._max_chars = 8000  # safety limit to avoid oversized payloads
        self._log = get_logger(__name__)

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        # Truncate to avoid server rejecting overly long inputs
        safe_texts = [t[: self._max_chars] if isinstance(t, str) else "" for t in texts]
        vectors: List[List[float]] = []
        for i in range(0, len(safe_texts), self._batch_size):
            batch = safe_texts[i:i + self._batch_size]
            payload = {
                "input": batch,
                "model": self._model,
                # some servers expect OpenAI-compatible fields; harmless if ignored
                "encoding_format": "float",
            }
            try:
                resp = await self._client.post_json("/v1/embeddings", json=payload)
            except httpx.HTTPStatusError as e:
                # Fallback to pooling API if embeddings are not supported by the served model
                body = {}
                try:
                    body = e.response.json()
                except Exception:
                    pass
                message = str(body)
                if e.response.status_code == 400 and "does not support Embeddings API" in message:
                    pool_payload = {
                        "input": batch,
                        "model": self._model,
                        "pooling_type": "mean",
                    }
                    resp = await self._client.post_json("/pooling", json=pool_payload)
                else:
                    raise
            # Try multiple common response schemas
            batch_vectors: List[List[float]] = []
            if isinstance(resp, dict):
                if isinstance(resp.get("data"), list):
                    for item in resp["data"]:
                        if isinstance(item, dict):
                            emb = item.get("embedding")
                            if isinstance(emb, list):
                                batch_vectors.append(emb)
                            elif isinstance(emb, dict):
                                if isinstance(emb.get("values"), list):
                                    batch_vectors.append(emb.get("values"))
                            elif isinstance(item.get("values"), list):
                                batch_vectors.append(item["values"]) 
                        elif isinstance(item, list) and all(isinstance(x, (int, float)) for x in item):
                            batch_vectors.append(item)
                elif isinstance(resp.get("embeddings"), list):
                    batch_vectors = resp.get("embeddings")
                elif isinstance(resp.get("data"), dict) and isinstance(resp["data"].get("embeddings"), list):
                    batch_vectors = resp["data"].get("embeddings")
            elif isinstance(resp, list) and resp and all(isinstance(x, (int, float)) for x in resp):
                batch_vectors = [resp]

            self._log.debug(f"Embeddings batch {i//self._batch_size}: req={len(batch)} got={len(batch_vectors)}")
            vectors.extend(batch_vectors)
        return vectors

    async def embed_query(self, text: str) -> List[float]:
        res = await self.embed_texts([text])
        return res[0] if res else []


