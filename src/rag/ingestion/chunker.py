from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from chonkie import SemanticChunker


@dataclass
class ChunkedDocument:
    text: str
    source: str
    chunk_index: int


class DocumentChunker:
    def __init__(
        self,
        chunk_size: int = 512,
        threshold: float = 0.8,
        add_context: bool = True,
        llm_model: str = "gemma4:e2b",
        llm_base_url: str = "http://localhost:11434/v1",
        chunker_embed_model_path: str = "models/potion-base-32M",
    ):
        self._chunker = SemanticChunker(
            embedding_model=chunker_embed_model_path,
            threshold=threshold,
            chunk_size=chunk_size,
        )
        self.add_context = add_context
        self._llm = ChatOpenAI(
            model=llm_model,
            base_url=llm_base_url,
            api_key="ollama",
        )

    def chunk(self, text: str, source: str) -> list[ChunkedDocument]:
        raw_chunks = self._chunker.chunk(text)
        results = []
        for i, chunk in enumerate(raw_chunks):
            chunk_text = chunk.text
            if self.add_context:
                chunk_text = self._add_context_prefix(chunk_text, source)
            results.append(
                ChunkedDocument(text=chunk_text, source=source, chunk_index=i)
            )
        return results

    def _add_context_prefix(self, chunk_text: str, source: str) -> str:
        prompt = (
            f"Document source: {source}\n"
            f"Chunk text (first 300 chars): {chunk_text[:300]}\n\n"
            "Write exactly ONE sentence starting with 'This chunk discusses' "
            "that describes what this chunk is about. Be specific."
        )
        response = self._llm.invoke([HumanMessage(content=prompt)])
        return f"[{response.content.strip()}]\n\n{chunk_text}"
