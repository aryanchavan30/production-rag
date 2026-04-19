from dataclasses import dataclass, field
from pathlib import Path
import hashlib

from rag.ingestion.parser import parse_document, parse_directory, ParsedDocument
from rag.ingestion.chunker import DocumentChunker
from rag.ingestion.embedder import Embedder
from rag.ingestion.indexer import QdrantIndexer, BM25Indexer


@dataclass
class IngestionResult:
    files_processed: int = 0
    chunks_indexed: int = 0
    files_skipped: int = 0
    errors: list[str] = field(default_factory=list)


def _doc_hash(doc: ParsedDocument) -> str:
    return hashlib.sha256(doc.text.encode()).hexdigest()


class IngestionPipeline:
    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "rag_documents",
        embed_dim: int = 1024,
        bm25_index_path: str = "data/bm25_index.pkl",
        chunk_size: int = 512,
        semantic_threshold: float = 0.8,
        add_context: bool = False,
        llm_model: str = "gemma4:e2b",
        llm_base_url: str = "http://localhost:11434/v1",
        embed_model_path: str = "models/bge-m3",
        chunker_embed_model_path: str = "models/potion-base-32M",
        embed_batch_size: int = 32,
    ):
        self.bm25_index_path = bm25_index_path
        self._embed_batch_size = embed_batch_size

        self.chunker = DocumentChunker(
            chunk_size=chunk_size,
            threshold=semantic_threshold,
            add_context=add_context,
            llm_model=llm_model,
            llm_base_url=llm_base_url,
            chunker_embed_model_path=chunker_embed_model_path,
        )
        self.embedder = Embedder(model_path=embed_model_path)
        self.qdrant = QdrantIndexer(
            url=qdrant_url,
            collection_name=collection_name,
            embed_dim=embed_dim,
        )
        self.bm25 = BM25Indexer(index_path=bm25_index_path)
        self._all_chunks: list[dict] = []

    def ingest_file(self, file_path: Path) -> IngestionResult:
        result = IngestionResult()
        try:
            doc = parse_document(file_path)
            self._process_document(doc, result)
            self._flush_bm25()
        except Exception as e:
            result.errors.append(f"{file_path}: {e}")
        return result

    def ingest_directory(self, directory: Path) -> IngestionResult:
        result = IngestionResult()
        documents = parse_directory(directory)
        for doc in documents:
            try:
                self._process_document(doc, result)
            except Exception as e:
                result.errors.append(f"{doc.source}: {e}")
        self._flush_bm25()
        return result

    def _process_document(self, doc: ParsedDocument, result: IngestionResult) -> None:
        chunks = self.chunker.chunk(doc.text, source=doc.source)
        if not chunks:
            return

        texts = [c.text for c in chunks]
        vectors = self.embedder.embed(texts, batch_size=self._embed_batch_size)

        payloads = [
            {
                "text": c.text,
                "source": c.source,
                "chunk_index": c.chunk_index,
                "doc_hash": _doc_hash(doc),
                "filename": Path(c.source).name,
            }
            for c in chunks
        ]

        self.qdrant.upsert(vectors, payloads)
        self._all_chunks.extend(payloads)

        result.files_processed += 1
        result.chunks_indexed += len(chunks)

    def _flush_bm25(self) -> None:
        if self._all_chunks:
            self.bm25.build(self._all_chunks)
            self.bm25.save()
