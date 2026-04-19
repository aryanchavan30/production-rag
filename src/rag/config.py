from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "rag_documents"
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "gemma4:e2b"
    embed_model_path: str = "models/bge-m3"
    reranker_model_path: str = "models/bge-reranker-v2-m3"
    chunker_embed_model_path: str = "models/potion-base-32M"
    chunk_size: int = 512
    semantic_threshold: float = 0.8
    bm25_index_path: str = "data/bm25_index.pkl"
    embed_dim: int = 1024

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
