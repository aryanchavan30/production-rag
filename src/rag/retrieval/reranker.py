import logging

import transformers
transformers.logging.set_verbosity_error()
from FlagEmbedding import FlagReranker

logger = logging.getLogger(__name__)


class BGEReranker:
    def __init__(self, model_path: str = "models/bge-reranker-v2-m3"):
        logger.info(f"Loading BGE-Reranker from {model_path} (fp16=False)")
        self._reranker = FlagReranker(model_path, use_fp16=False)
        logger.info("BGE-Reranker ready")

    def rerank(self, query: str, docs: list[dict], top_k: int = 5) -> list[dict]:
        if not docs:
            logger.info("[RERANK] No docs to rerank, returning empty list")
            return []

        logger.info(f"[RERANK] Reranking {len(docs)} docs for query: '{query[:60]}'")
        pairs = [[query, d["text"]] for d in docs]
        scores = self._reranker.compute_score(pairs, normalize=True)
        if isinstance(scores, float):
            scores = [scores]

        scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        results = [{**doc, "rerank_score": float(score)} for doc, score in scored[:top_k]]
        logger.info(f"[RERANK] Top {len(results)} scores: {[round(r['rerank_score'], 4) for r in results]}")
        return results
