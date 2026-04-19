import transformers
transformers.logging.set_verbosity_error()
from FlagEmbedding import FlagReranker


class BGEReranker:
    def __init__(self, model_path: str = "models/bge-reranker-v2-m3"):
        self._reranker = FlagReranker(model_path, use_fp16=False)

    def rerank(self, query: str, docs: list[dict], top_k: int = 5) -> list[dict]:
        if not docs:
            return []
        pairs = [[query, d["text"]] for d in docs]
        scores = self._reranker.compute_score(pairs, normalize=True)
        if isinstance(scores, float):
            scores = [scores]
        scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [{**doc, "rerank_score": float(score)} for doc, score in scored[:top_k]]
