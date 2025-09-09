from typing import Any, Dict, Iterable, List, Optional, Tuple

from pymilvus import (
    Collection, CollectionSchema, DataType, FieldSchema, connections, utility, MilvusException
)

from app.core.config import get_settings
from app.utils.logging import get_logger


logger = get_logger(__name__)


class MilvusVectorStore:
    def __init__(self, collection_name: Optional[str] = None, dim: Optional[int] = None):
        settings = get_settings()
        self.collection_name = collection_name or settings.milvus_collection
        self.dim = dim or settings.milvus_dim
        self._connect()
        self._ensure_collection()

    def _connect(self) -> None:
        settings = get_settings()
        uri = settings.milvus_uri.strip()
        # Normalize URI for pymilvus: prefer host/port if possible
        if uri.startswith("http://"):
            uri = uri[len("http://"):]
        if uri.startswith("https://"):
            uri = uri[len("https://"):]
        try:
            if ":" in uri:
                host, port_str = uri.split(":", 1)
                port = int(port_str)
                connections.connect(
                    alias="default",
                    host=host,
                    port=str(port),
                    user=settings.milvus_user,
                    password=settings.milvus_password,
                )
            else:
                connections.connect(
                    alias="default",
                    uri=uri,
                    user=settings.milvus_user,
                    password=settings.milvus_password,
                )
        except Exception as exc:
            logger.error(f"Failed to connect to Milvus at {settings.milvus_uri}: {exc}")
            raise

    def _ensure_collection(self) -> None:
        if not utility.has_collection(self.collection_name):
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64, auto_id=False),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192),
                FieldSchema(name="metadata", dtype=DataType.JSON),
            ]
            schema = CollectionSchema(fields=fields, description="Vision RAG documents")
            coll = Collection(name=self.collection_name, schema=schema)
            coll.create_index(
                field_name="vector",
                index_params={
                    "index_type": get_settings().milvus_index_type,
                    "metric_type": get_settings().milvus_metric_type,
                    "params": {"nlist": get_settings().milvus_nlist},
                },
            )
            coll.load()
        else:
            coll = Collection(self.collection_name)
            if not coll.has_index():
                coll.create_index(
                    field_name="vector",
                    index_params={
                        "index_type": get_settings().milvus_index_type,
                        "metric_type": get_settings().milvus_metric_type,
                        "params": {"nlist": get_settings().milvus_nlist},
                    },
                )
            coll.load()

    def upsert(self, ids: List[str], embeddings: List[List[float]], texts: List[str], metadatas: List[Dict[str, Any]]) -> int:
        coll = Collection(self.collection_name)
        entities = [ids, embeddings, texts, metadatas]
        mr = coll.upsert(entities)
        coll.flush()
        return mr.insert_count if hasattr(mr, "insert_count") else len(ids)

    def search(self, query_vectors: List[List[float]], k: int = 10, expr: Optional[str] = None, output_fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        coll = Collection(self.collection_name)
        search_params = {"metric_type": get_settings().milvus_metric_type, "params": {"nprobe": 32}}
        results = coll.search(
            data=query_vectors,
            anns_field="vector",
            param=search_params,
            limit=k,
            expr=expr,
            output_fields=output_fields or ["text", "metadata"],
        )
        out: List[Dict[str, Any]] = []
        for hits in results:
            for hit in hits:
                out.append({
                    "id": hit.id,
                    "distance": hit.distance,
                    "text": hit.entity.get("text"),
                    "metadata": hit.entity.get("metadata"),
                })
        return out

    def drop(self) -> None:
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)


