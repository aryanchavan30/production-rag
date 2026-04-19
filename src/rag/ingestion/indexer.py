import hashlib
import logging
import os
import pickle
import uuid

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    PointStruct,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    VectorParams,
)
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


def _make_point_id(source: str, chunk_index: int) -> str:
    raw = f"{source}::{chunk_index}"
    return str(uuid.UUID(hashlib.md5(raw.encode()).hexdigest()))


class QdrantIndexer:
    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection_name: str = "rag_documents",
        embed_dim: int = 1024,
    ):
        logger.info(f"Connecting to Qdrant at {url}")
        self._client = QdrantClient(":memory:") if url == ":memory:" else QdrantClient(url=url)
        self._collection = collection_name
        self._embed_dim = embed_dim
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self._client.get_collections().collections}
        if self._collection not in existing:
            logger.info(f"Creating collection '{self._collection}' (dim={self._embed_dim}, HNSW + INT8)")
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=self._embed_dim,
                    distance=Distance.COSINE,
                    hnsw_config=HnswConfigDiff(m=16, ef_construct=200),
                    quantization_config=ScalarQuantization(
                        scalar=ScalarQuantizationConfig(
                            type=ScalarType.INT8,
                            always_ram=True,
                        )
                    ),
                ),
            )
        else:
            logger.info(f"Collection '{self._collection}' already exists, reusing")

    def upsert(self, vectors: np.ndarray, payloads: list[dict]) -> None:
        logger.info(f"Upserting {len(payloads)} vector(s) to Qdrant collection '{self._collection}'")
        points = [
            PointStruct(
                id=_make_point_id(p["source"], p["chunk_index"]),
                vector=vectors[i].tolist(),
                payload=p,
            )
            for i, p in enumerate(payloads)
        ]
        for batch_start in range(0, len(points), 100):
            batch = points[batch_start: batch_start + 100]
            self._client.upsert(collection_name=self._collection, points=batch)
            logger.debug(f"Upserted batch [{batch_start}:{batch_start + len(batch)}]")

    def search(self, query_vector: np.ndarray, top_k: int = 20) -> list[dict]:
        logger.debug(f"Qdrant search (top_k={top_k})")
        result = self._client.query_points(
            collection_name=self._collection,
            query=query_vector.tolist(),
            limit=top_k,
            with_payload=True,
        )
        hits = [{**hit.payload, "score": hit.score} for hit in result.points]
        logger.debug(f"Qdrant returned {len(hits)} results")
        return hits

    def count(self) -> int:
        return self._client.count(collection_name=self._collection).count


class BM25Indexer:
    def __init__(self, index_path: str = "data/bm25_index.pkl"):
        self.index_path = index_path
        self._bm25: BM25Okapi | None = None
        self._chunks: list[dict] = []

    def build(self, chunks: list[dict]) -> None:
        logger.info(f"Building BM25 index over {len(chunks)} chunks")
        self._chunks = chunks
        tokenized = [c["text"].lower().split() for c in chunks]
        self._bm25 = BM25Okapi(tokenized)
        logger.info("BM25 index built")

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        if self._bm25 is None:
            raise RuntimeError("BM25 index not built. Call build() first.")
        logger.debug(f"BM25 search: '{query[:60]}' (top_k={top_k})")
        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        hits = [{**self._chunks[i], "score": float(scores[i])} for i in top_indices]
        logger.debug(f"BM25 returned {len(hits)} results")
        return hits

    def save(self) -> None:
        dir_name = os.path.dirname(self.index_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump({"bm25": self._bm25, "chunks": self._chunks}, f)
        logger.info(f"BM25 index saved → {self.index_path} ({len(self._chunks)} chunks)")

    @classmethod
    def load(cls, index_path: str) -> "BM25Indexer":
        logger.info(f"Loading BM25 index from {index_path}")
        with open(index_path, "rb") as f:
            data = pickle.load(f)
        indexer = cls(index_path=index_path)
        indexer._bm25 = data["bm25"]
        indexer._chunks = data["chunks"]
        logger.info(f"BM25 index loaded ({len(indexer._chunks)} chunks)")
        return indexer
