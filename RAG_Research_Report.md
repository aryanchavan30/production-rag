# Production-Ready Local RAG: Deep Research Report (2024–2026)

## Executive Summary

Naive RAG (embed → retrieve → generate) succeeds in demos but fails in production (10–40% success rate in enterprise). The 2025 production standard is multi-stage pipelines with hybrid retrieval, reranking, evaluation loops, and agentic self-correction — **no external APIs needed**.

---

## 1. RAG Architecture Generations

| Generation | Pattern | Status |
|---|---|---|
| Naive RAG (2022–2023) | embed → top-k retrieval → LLM | ❌ Demo only |
| Advanced RAG (2023–2024) | + HyDE, multi-query, hybrid search, reranking | ✅ Minimum baseline |
| Modular RAG (2024) | Query router → retriever(s) → reranker → generator → evaluator | ✅ Production standard |
| Agentic/Adaptive RAG (2024–2025) | System decides HOW to retrieve at runtime via LangGraph state machine | ✅ Best quality |
| Corrective RAG (CRAG) | If relevance score < threshold → rewrite query → retrieve again | ✅ For high-stakes domains |

### Real Company Deployments

- **Uber (Genie On-Call Copilot)** — Spark ingestion → vector DB → Slack bot. Saved 13,000 engineering hours, answered 70k+ questions. LLM-as-judge + user feedback (Resolved/Helpful/Not Relevant).
- **Cloudflare AutoRAG** — Edge-deployed semantic search on Workers AI. $5–10/month vs typical $100–200+.
- **Microsoft GraphRAG** — Knowledge graph extraction → community summaries. 72–83% better comprehensiveness on global questions. LazyGraphRAG (2024) reduces indexing cost to 0.1%.
- **LinkedIn** — LangGraph in production for stateful agent workflows with human-in-the-loop.

---

## 2. Tech Stack Deep Dive

### Framework Layer

**LangGraph** (Orchestration — production standard)
```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

class RAGState(TypedDict):
    question: str
    documents: List[str]
    generation: str
    rewrite_count: int

# Nodes: retrieve → grade → (rewrite if irrelevant) → generate
# Conditional edges enable Corrective RAG / Adaptive RAG
```
Used in production at LinkedIn, Uber, 400+ companies.

**LlamaIndex** (Best RAG-specific framework)
```python
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from llama_index.core.node_parser import HierarchicalNodeParser

# Parent-child chunking: 2048 → 512 → 128 tokens
parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[2048, 512, 128])
# Retrieve small chunks (precision) → return large parent (full context)
```
Fully async (`arun`, `astream`), built-in `IngestionCache` skips re-embedding unchanged docs.

**DSPy** (Prompt optimization — use INSIDE LangGraph/LlamaIndex)
```python
class RAGSignature(dspy.Signature):
    """Answer question using retrieved context."""
    question: str = dspy.InputField()
    context: list[str] = dspy.InputField()
    answer: str = dspy.OutputField()

# MIPROv2 optimizer: optimizes instructions + examples jointly
# Demonstrated: 53% → 61% accuracy on retrieval benchmarks
teleprompter = MIPROv2(metric=dspy.SemanticF1(), auto="medium")
optimized_rag = teleprompter.compile(RAGModule(), trainset=train_examples)
```
Not an orchestration framework — use it to auto-optimize the prompts inside your pipeline.

---

### Chunking — Chonkie

| Chunker | Speed | Use Case |
|---|---|---|
| FastChunker | 100+ GB/s | Max throughput |
| TokenChunker | 33x faster | Fixed token chunks |
| SentenceChunker | ~2x faster | Natural boundaries |
| RecursiveChunker | Fast | Markdown/code |
| SemanticChunker | 2.5x faster | Topic-aware splits |
| LateChunker | Moderate | Cross-reference docs (legal, manuals) |
| SlumberChunker | Slowest | LLM-driven high quality |

**SemanticChunker:** Savitzky-Golay filter on cosine similarity curve → detects local minima as boundaries.

**LateChunker:** Embeds full document first → pools token embeddings per chunk → 12–18% retrieval accuracy improvement on cross-reference-heavy docs (legal contracts, technical manuals).

**Hierarchical (most impactful for RAG quality):**
```
Document → Section (1024 tokens) → Paragraph (256 tokens) → Sentence (64 tokens)
Index only leaf nodes → retrieve parent chunk at query time (full context)
```

---

### Vector Databases

| DB | Best For | Key Strength |
|---|---|---|
| Qdrant ⭐ | Production local, complex filtering | Rust-based, Filterable HNSW, INT8 quantization, 41 QPS at 99% recall on 50M vectors |
| pgvector/pgvectorscale | Already on Postgres, <10M vectors | 471 QPS at 99% recall (TimescaleDB) |
| Weaviate | Multimodal, knowledge graphs | Native hybrid search, local HuggingFace modules |
| Milvus | Billions of vectors, GPU cluster | 100k+ QPS, GPU-accelerated CAGRA index |
| Chroma | Dev/prototyping only | Easiest setup — not production-grade |

**Qdrant production config:**
```python
client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(
        size=768, distance=Distance.COSINE,
        hnsw_config=HnswConfigDiff(m=16, ef_construct=200),
        quantization_config=ScalarQuantizationConfig(
            type=QuantizationType.INT8, quantile=0.99, always_ram=True
        ),
    ),
)
```

---

### Local LLM Inference

| Tool | Best For | Throughput |
|---|---|---|
| vLLM ⭐ | Production server (GPU) | 2–35x faster than naive; 44x vs llama.cpp at concurrency |
| Ollama | Dev + single-user apps | Good; degrades under concurrent load |
| llama.cpp | CPU/edge/embedded | ~15–30 tok/sec 7B; not for concurrent users |

**vLLM key features:** PagedAttention (60–80% less KV cache waste), continuous batching, automatic prefix caching, streaming.

**Recommended local models for production:**

| Model | VRAM | Best For |
|---|---|---|
| Mistral Small 3 (24B) | 24GB | General production (sweet spot) |
| Qwen3-14B | 12–16GB | Strong reasoning, multilingual |
| Llama-3.3-70B (Q4) | 40GB | Highest capability |
| Phi-4 (14B) | 12GB | Code + reasoning, efficient |
| Qwen3-4B | 4–6GB | Resource-constrained |

---

### Local Embedding Models

| Model | Latency | Context | Key Feature |
|---|---|---|---|
| BGE-M3 ⭐ | ~50ms | 8192 tokens | Dense + sparse + multi-vector from ONE model; 100+ languages; MIT |
| Nomic Embed v1/v2 | ~100ms | 8192 tokens | Matryoshka (truncate to 64 dims); fully open-source |
| E5-base-instruct | ~30ms | 512 tokens | Balanced; instruction-tuned |
| all-MiniLM-L6-v2 | <30ms | 256 tokens | Prototype/CPU only |

**BGE-M3 recommendation:** Single model handles dense vector search AND sparse (BM25-like) retrieval — eliminates need for separate embedding models.

---

## 3. Chunking Strategy (Production)

**Hierarchy of strategies (worst to best):**

1. Fixed-size → fast but poor coherence
2. Sentence → natural but ignores topics
3. Recursive → great for Markdown/code
4. Semantic → production standard for most cases
5. **Hierarchical + Semantic** → best quality (index small, return large)
6. **Contextual Chunking (Anthropic-style)** → prepend 50–100 token context per chunk before embedding:

```
"[From: Q3 2024 Earnings Report, Section: Revenue]
The North American division grew 23% YoY..."
```

Cost: one local LLM call per chunk at ingestion. Pre-compute and cache.
**Result: 49% fewer retrieval failures** (67% with reranking on top).

---

## 4. Memory Systems

**Mem0** (arXiv April 2025) — production memory layer:
- Dynamically extracts salient facts using LLM
- Deduplicates + consolidates memories
- Vector DB (dense) + optional graph store (relational) for multi-hop memory
- **Results: 26% better than OpenAI's memory, 91% lower p95 latency, 90%+ token cost reduction**

```python
config = {
    "llm": {"provider": "ollama", "config": {"model": "llama3.2"}},
    "embedder": {"provider": "huggingface", "config": {"model": "BAAI/bge-m3"}},
    "vector_store": {"provider": "qdrant", "config": {"host": "localhost", "port": 6333}}
}
m = Memory.from_config(config)
m.add("User prefers concise technical answers", user_id="user_123")
```

**Memory taxonomy:**
- **Short-term:** last N turns, rolling window of 5–10
- **Summary:** LLM compresses older history to ~200 words
- **Entity:** extract named entities, track their state
- **Long-term semantic:** store full history in vector DB, retrieve relevant past exchanges

---

## 5. Retrieval Pipeline (Production Standard)

```
Query
  → Multi-Query Expansion (3 rephrased versions)
  → Parallel: Dense (Qdrant) + Sparse (BM25 or SPLADE)
  → Reciprocal Rank Fusion (k=60)
  → MMR Deduplication (λ=0.7, top-20 → top-20 unique)
  → Reranker (BGE Reranker v2-m3 OR FlashRank for CPU)
  → Top-5 to LLM
```

**Hybrid search improvement: 15–30% better accuracy** vs pure vector search alone.

**HyDE (Hypothetical Document Embeddings):**
```python
hyde_answer = llm.generate(f"Write a passage that answers: {query}")
embedding = embed(hyde_answer)  # embed the answer, not the question
results = vector_store.search(embedding, top_k=20)
```
20–35% improvement on ambiguous/specialized domain queries.

**Rerankers:**

| Reranker | Speed | Quality | Use Case |
|---|---|---|---|
| BGE Reranker v2-m3 (278M) | GPU fast, CPU ok | = Cohere Rerank | Production (GPU) |
| FlashRank | <20ms CPU | Good | CPU-only / latency critical |
| ColBERT via RAGatouille | Fast at scale | Excellent | High-volume systems |
| cross-encoder/ms-marco-MiniLM-L6 | Fast CPU | Good (EN only) | Lightweight English |

---

## 6. Latency Optimization

**Multi-Level Cache Architecture:**

| Cache Layer | Latency Reduction | How |
|---|---|---|
| Exact query cache | Sub-ms | Hash → Redis |
| Semantic cache | 60–85% (65x vs full pipeline) | Embed query → search cache DB (threshold 0.95) |
| KV prefix cache | Free for repeated prefixes | vLLM automatic prefix caching |
| Retrieval result cache | Fast | Redis TTL 60–300s |

**Critical async pattern:**
```python
# Embed query AND BM25 search in PARALLEL
embed_task = asyncio.create_task(embed_model.aembed(query))
sparse_task = asyncio.create_task(bm25_index.asearch(query, k=20))
query_embedding, sparse_results = await asyncio.gather(embed_task, sparse_task)
```

**Streaming:** First token in <1s feels fast even with 10s total latency.

**Pre-computation at ingestion:** Embed all docs, build BM25 index, generate contextual prefixes, generate LLM summaries — all cached. Never compute at query time.

---

## 7. Full Ingestion Pipeline (PTI)

```
Parse     → Docling (IBM) or Unstructured.io — preserves tables, OCR
          → LlamaParse for complex PDF layouts

Transform → Chonkie SemanticChunker or HierarchicalChunker
          → Metadata extraction (title, section, date, entities)
          → Contextual prefix generation (local LLM — cached)
          → Question generation ("What questions does this chunk answer?")
          → LLM summaries per document/section

Index     → BGE-M3 embeddings → Qdrant (dense)
          → BM25Okapi index (sparse, Elasticsearch or rank-bm25)
          → Parent chunks stored separately for parent-child expansion
```

**Incremental updates — LlamaIndex IngestionCache with Redis:**
```python
cache = IngestionCache(cache=RedisKVStore(redis_uri="redis://localhost:6379"))
pipeline = IngestionPipeline(transformations=[splitter, embedder], cache=cache)
# Skips re-embedding unchanged documents via content hashing
```

---

## 8. Evaluation & Observability

**RAGAS metrics:**

| Metric | Target |
|---|---|
| Faithfulness | > 0.8 |
| Answer Relevancy | > 0.8 |
| Context Precision | > 0.8 |
| Context Recall | > 0.7 |

**Observability tools (self-hosted):**
- **Arize Phoenix** — `pip install arize-phoenix`, runs locally at port 6006, auto-instruments LangChain
- **Langfuse** — Docker Compose self-hosted, RAGAS integration, prompt management, MIT license

**What to monitor in production:**
- Faithfulness drift (alert if drops >10% week-over-week)
- p50/p95/p99 latency per pipeline stage
- Semantic cache hit rate
- User feedback (thumbs up/down, regenerate requests)
- Dead chunks (documents never retrieved = useless index)

---

## 9. Recommended Production Stack (Local-Only)

### Full Architecture Diagram

```
[Data Sources]
      ↓
[Ingestion Pipeline]
  ├── Docling / Unstructured.io (parse)
  ├── Chonkie SemanticChunker + Hierarchical (chunk)
  ├── Local LLM via Ollama (contextual prefix generation, cached)
  ├── BGE-M3 (embed)
  └── BM25Okapi (sparse index)
      ↓
[Storage]
  ├── Qdrant (vector + payload, INT8 quantization)
  ├── Redis (semantic cache + retrieval cache)
  └── Mem0 + Qdrant (long-term memory)
      ↓
[Query Pipeline — LangGraph State Machine]
  ├── Query Router (simple / complex / multi-hop)
  ├── Multi-Query Expander (3 rephrased queries)
  ├── Parallel: Dense (Qdrant) + Sparse (BM25)
  ├── RRF Fusion (k=60)
  ├── MMR Deduplication (λ=0.7)
  ├── BGE Reranker v2-m3 (top-20 → top-5)
  └── Parent Chunk Expansion
      ↓
[Generation]
  ├── vLLM (Mistral Small 3 24B / Llama 3.3 70B)
  ├── Streaming response
  └── Source citation
      ↓
[Evaluation + Observability]
  ├── RAGAS (offline sampling)
  └── Langfuse or Arize Phoenix (self-hosted traces)
```

### By Constraint

| Hardware | LLM | Embedding | Reranker | Orchestration |
|---|---|---|---|---|
| 24GB+ VRAM | Mistral Small 3 24B (vLLM) | BGE-M3 | BGE Reranker v2-m3 | LangGraph + DSPy |
| 12–16GB VRAM | Qwen3-14B (vLLM AWQ) | Nomic Embed v1 | FlashRank | LlamaIndex Workflows |
| CPU-only / 8GB | Qwen3-4B or Phi-4 Q4 (Ollama) | E5-small-v2 | FlashRank | LlamaIndex basic |

---

## 10. Key Numbers to Know

| Parameter | Production Value |
|---|---|
| Chunk size | 512 tokens (leaf), 1024–2048 (parent) |
| Chunk overlap | 10–15% (50–64 tokens) |
| Retrieve top-k | 20 candidates |
| After reranking | Top 5 to LLM |
| Embedding dim | 768 (BGE-M3), or 256 with Matryoshka |
| Qdrant HNSW m | 16, ef_construct = 200 |
| Semantic cache threshold | 0.95 cosine similarity |
| RRF k constant | 60 |
| MMR lambda | 0.7 (relevance/diversity balance) |
| Faithfulness target | > 0.8 |
