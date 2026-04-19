import hashlib
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
        self._client = QdrantClient(":memory:") if url == ":memory:" else QdrantClient(url=url)
        self._collection = collection_name
        self._embed_dim = embed_dim
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self._client.get_collections().collections}
        if self._collection not in existing:
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

    def upsert(self, vectors: np.ndarray, payloads: list[dict]) -> None:
        points = [
            PointStruct(
                id=_make_point_id(p["source"], p["chunk_index"]),
                vector=vectors[i].tolist(),
                payload=p,
            )
            for i, p in enumerate(payloads)
        ]
        for batch_start in range(0, len(points), 100):
            self._client.upsert(
                collection_name=self._collection,
                points=points[batch_start : batch_start + 100],
            )

    def search(self, query_vector: np.ndarray, top_k: int = 20) -> list[dict]:
        result = self._client.query_points(
            collection_name=self._collection,
            query=query_vector.tolist(),
            limit=top_k,
            with_payload=True,
        )
        return [{**hit.payload, "score": hit.score} for hit in result.points]

    def count(self) -> int:
        return self._client.count(collection_name=self._collection).count


class BM25Indexer:
    def __init__(self, index_path: str = "data/bm25_index.pkl"):
        self.index_path = index_path
        self._bm25: BM25Okapi | None = None
        self._chunks: list[dict] = []

    def build(self, chunks: list[dict]) -> None:
        self._chunks = chunks
        tokenized = [c["text"].lower().split() for c in chunks]
        self._bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        if self._bm25 is None:
            raise RuntimeError("BM25 index not built. Call build() first.")
        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [{**self._chunks[i], "score": float(scores[i])} for i in top_indices]

    def save(self) -> None:
        dir_name = os.path.dirname(self.index_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump({"bm25": self._bm25, "chunks": self._chunks}, f)

    @classmethod
    def load(cls, index_path: str) -> "BM25Indexer":
        with open(index_path, "rb") as f:
            data = pickle.load(f)
        indexer = cls(index_path=index_path)
        indexer._bm25 = data["bm25"]
        indexer._chunks = data["chunks"]
        return indexer
