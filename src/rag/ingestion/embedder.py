import logging

import numpy as np
import torch
import transformers
transformers.logging.set_verbosity_error()

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(self, model_path: str = "models/bge-m3"):
        from FlagEmbedding import BGEM3FlagModel

        logger.info(f"Loading BGE-M3 from {model_path} (fp16={torch.cuda.is_available()})")
        use_fp16 = torch.cuda.is_available()
        self._model = BGEM3FlagModel(model_path, use_fp16=use_fp16)
        logger.info("BGE-M3 ready")

    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            logger.debug("embed() called with empty list, returning empty array")
            return np.empty((0, 1024), dtype=np.float32)

        logger.info(f"Embedding {len(texts)} text(s) (batch_size={batch_size})")
        output = self._model.encode(
            texts,
            batch_size=batch_size,
            max_length=8192,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        vectors = np.array(output["dense_vecs"], dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        vectors = vectors / norms
        logger.info(f"Embedded → shape {vectors.shape}")
        return vectors
