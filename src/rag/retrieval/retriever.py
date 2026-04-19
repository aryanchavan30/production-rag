import numpy as np
from rag.ingestion.embedder import Embedder
from rag.ingestion.indexer import QdrantIndexer, BM25Indexer


def rrf_fuse(results_list: list[list[dict]], k: int = 60) -> list[dict]:
    scores: dict[str, float] = {}
    docs: dict[str, dict] = {}
    for results in results_list:
        for rank, doc in enumerate(results):
            doc_id = f"{doc['source']}::{doc['chunk_index']}"
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
            if doc_id not in docs:
                docs[doc_id] = doc
    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [{**docs[did], "rrf_score": scores[did]} for did in sorted_ids]


class HybridRetriever:
    def __init__(
        self,
        qdrant: QdrantIndexer,
        bm25: BM25Indexer,
        embedder: Embedder,
        top_k: int = 20,
    ):
        self.qdrant = qdrant
        self.bm25 = bm25
        self.embedder = embedder
        self.top_k = top_k

    def retrieve(self, query: str) -> list[dict]:
        query_vec = self.embedder.embed([query])[0]
        dense = self.qdrant.search(query_vec, top_k=self.top_k)
        sparse = self.bm25.search(query, top_k=self.top_k)
        return rrf_fuse([dense, sparse], k=60)[: self.top_k]
