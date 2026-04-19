import numpy as np
import torch
import transformers
transformers.logging.set_verbosity_error()


class Embedder:
    def __init__(self, model_path: str = "models/bge-m3"):
        from FlagEmbedding import BGEM3FlagModel

        use_fp16 = torch.cuda.is_available()
        self._model = BGEM3FlagModel(model_path, use_fp16=use_fp16)

    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.empty((0, 1024), dtype=np.float32)

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
        return vectors / norms
