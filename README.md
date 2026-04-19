# local-rag-pipeline

Production-ready local RAG system — hybrid retrieval (Qdrant + BM25), BGE-M3 embeddings, adaptive LangGraph pipeline, and streaming FastAPI — no external APIs, runs fully offline.

---

## What It Does

You point it at a folder of PDFs, DOCX, or text files. It ingests them into a local vector database. You then ask questions and get answers grounded in your documents — streamed word-by-word, with source citations.

Everything runs on your machine. No OpenAI. No cloud. No data leaving your computer.

---

## Architecture

```
                         INGEST
  Documents → Parse → Chunk → Embed (BGE-M3) → Qdrant (dense) + BM25 (sparse)

                         QUERY
  Question
     │
     ▼
  [EXPAND]  — LLM generates 2 alternative phrasings
     │
     ▼
  [RETRIEVE] — Qdrant dense + BM25 keyword → RRF fusion → BGE Reranker top-5
     │
     ▼
  [GRADE]   — LLM checks each doc: relevant? YES/NO
     │
  ┌──┴──┐
  │     │
DONE  [REWRITE] — LLM rewrites question → back to RETRIEVE (max 2×)
  │
  ▼
  [GENERATE] — LLM answers from grounded context → streamed via SSE
```

---

## Tech Stack

| Component | Library |
|-----------|---------|
| Document parsing | Docling 2.90+ |
| Semantic chunking | Chonkie 1.6+ (potion-base-32M) |
| Embeddings | BGE-M3 via FlagEmbedding 1.3+ (1024-dim) |
| Vector DB | Qdrant 1.17+ (HNSW + INT8 quantization) |
| Sparse index | BM25Okapi (rank-bm25) |
| Reranker | BGE-Reranker-v2-m3 (cross-encoder) |
| LLM | Any OpenAI-compatible API (Ollama, vLLM) |
| Pipeline | LangGraph 1.1.8 + LangChain 1.2.15 |
| API | FastAPI + SSE streaming |
| CLI | Typer + Rich |

---

## Hardware Requirements

Tested on:
- GPU: GTX 1650 4GB VRAM
- RAM: 24GB
- All models run on CPU (BGE-M3 is 570M params, fits comfortably in RAM)

Minimum: 16GB RAM, no GPU required.

---

## Setup

### 1. Install dependencies

```bash
# Requires Python 3.12+ and uv
uv sync --extra dev
```

### 2. Download models

```bash
uv run python scripts/download_models.py
```

This downloads ~3GB of models into `models/`:
- `BAAI/bge-m3` — embedder
- `BAAI/bge-reranker-v2-m3` — reranker
- `minishlab/potion-base-32M` — chunker

### 3. Start Qdrant

```bash
docker run -d -p 6333:6333 -v ./qdrant_storage:/qdrant/storage qdrant/qdrant
```

### 4. Start Ollama (or vLLM)

```bash
ollama serve
ollama pull ministral:3b   # or any model you prefer
```

### 5. Configure `.env`

```env
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=rag_documents
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=ministral:3b
EMBED_MODEL_PATH=models/bge-m3
RERANKER_MODEL_PATH=models/bge-reranker-v2-m3
CHUNKER_EMBED_MODEL_PATH=models/potion-base-32M
CHUNK_SIZE=512
SEMANTIC_THRESHOLD=0.8
BM25_INDEX_PATH=data/bm25_index.pkl
EMBED_DIM=1024
```

To switch from Ollama to vLLM, only change `LLM_BASE_URL`. Everything else stays the same.

---

## Usage

### Ingest documents

```bash
# Ingest a folder
uv run rag ingest ./my_documents/

# Ingest a single file
uv run rag ingest report.pdf

# With LLM context prefixes (improves retrieval quality, slower)
uv run rag ingest ./my_documents/ --context
```

### Query from the CLI

```bash
uv run rag query "What are the key findings?"

# Show source chunks used to answer
uv run rag query "What are the key findings?" --sources

# Control how many documents to retrieve
uv run rag query "Summarize the methodology" --top-k 10
```

### Run the API server

```bash
uv run uvicorn rag.api.main:app --reload --port 8000
```

#### Query via HTTP (streaming SSE)

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the key findings?"}' \
  --no-buffer
```

Response stream:
```
data: {"type": "sources", "data": [...]}
data: {"type": "token", "text": "The "}
data: {"type": "token", "text": "key "}
data: {"type": "token", "text": "findings "}
...
data: [DONE]
```

#### API docs

Visit `http://localhost:8000/docs` after starting the server.

---

## Project Structure

```
src/rag/
├── config.py              # Pydantic settings — all paths and URLs from .env
├── cli.py                 # Typer CLI: rag ingest / rag query
├── ingestion/
│   ├── parser.py          # Docling (PDF/DOCX) + plain text (TXT/MD)
│   ├── chunker.py         # Chonkie SemanticChunker + optional LLM context prefix
│   ├── embedder.py        # BGE-M3 batch embedding
│   ├── indexer.py         # Qdrant upsert + BM25 build/save/load
│   └── pipeline.py        # Orchestrates parse → chunk → embed → index
├── retrieval/
│   ├── retriever.py       # HybridRetriever: dense + BM25 + RRF fusion
│   └── reranker.py        # BGEReranker: cross-encoder reranking
├── graph/
│   ├── state.py           # RAGState TypedDict
│   ├── nodes.py           # expand / retrieve / grade / rewrite node factories
│   └── graph.py           # AdaptiveRAGGraph: LangGraph StateGraph
├── generation/
│   └── generator.py       # generate() sync + generate_stream() async
└── api/
    ├── schemas.py          # QueryRequest / QueryResponse Pydantic models
    └── main.py             # FastAPI app: POST /query (SSE), GET /health
```

---

## Running Tests

```bash
# All tests (excludes model-dependent tests)
uv run pytest tests/ -v --ignore=tests/ingestion/test_embedder.py --ignore=tests/ingestion/test_pipeline.py

# Phase 2 only (no models needed — all mocked)
uv run pytest tests/retrieval/ tests/graph/ tests/api/ -v
```

---

## Roadmap

- [x] Phase 1 — Ingestion pipeline (parse, chunk, embed, index)
- [x] Phase 2 — Query pipeline (hybrid retrieval, adaptive LangGraph, streaming API)
- [ ] Phase 3 — Redis semantic cache, Mem0 memory, RAGAS evaluation, Langfuse tracing
