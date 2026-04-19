# Complete RAG Tutorial — From Zero to Production

---

## PART 1: What is RAG and Why Does It Exist?

### The Problem with LLMs Alone

Imagine you hired a genius consultant. They went to school for 20 years, read millions of books, and know almost everything. But here's the catch:

- They graduated in **2023** — they know nothing that happened after that
- They can't remember anything you told them in your **last meeting** (no memory)
- They sometimes **confidently make up facts** (hallucination)
- They don't know anything about **your company's private documents**

That genius consultant is your LLM (like Llama, Mistral, Qwen, etc.).

### RAG is the Solution

RAG = **R**etrieval **A**ugmented **G**eneration

The idea: Before the LLM answers your question, **go find the relevant documents first**, then give those documents to the LLM along with the question.

```
User asks: "What is our refund policy?"

Without RAG:
  → LLM guesses based on training data → might hallucinate

With RAG:
  → System searches your documents → finds "Refund Policy.pdf"
  → Gives that document to the LLM → LLM reads it → gives accurate answer
```

**Simple analogy:** RAG is like giving an open-book exam to a student instead of asking them to recall everything from memory. The student (LLM) is still smart — they still need to understand and reason. But now they have the right book (retrieved document) in front of them.

---

## PART 2: The 5 Generations of RAG (Simple to Best)

### Generation 1: Naive RAG (2022–2023) — "The Basic Library"

**How it works:**
```
Step 1: Take your documents, cut them into pieces (chunks)
Step 2: Convert each chunk into a number (embedding/vector)
Step 3: Store all these numbers in a database (vector store)

At query time:
Step 4: Convert the user's question into a number
Step 5: Find the chunks whose numbers are closest (most similar)
Step 6: Give those chunks + the question to the LLM
Step 7: LLM generates an answer
```

**Why it fails in production:**

Imagine you're looking for "Python error when connecting to database" in a giant library. The librarian brings you books about "Python snakes near water bodies" because the numbers were similar — but semantically wrong.

Real failure examples:
- Can't handle exact codes like "Product ID: XK-229-B" (keywords matter, not meaning)
- Misses answers spread across 3 different documents
- Returns the same chunk 5 times (duplicates)
- Only 10–40% accuracy on real enterprise questions

---

### Generation 2: Advanced RAG (2023–2024) — "The Smart Library"

Adds several improvements BEFORE and AFTER retrieval:

**Before retrieval (query improvement):**
- Rewrite the query in multiple ways
- Generate a hypothetical answer first (HyDE — explained later)

**Better storage:**
- Better chunking (semantic, hierarchical)
- Hybrid search (keyword + meaning)

**After retrieval (result improvement):**
- Rerank the results to pick the best ones
- Filter by metadata (date, department, file type)

**This is the minimum acceptable for production.**

---

### Generation 3: Modular RAG (2024) — "The Assembly Line"

Think of a car factory. Each station does ONE job:
- Station 1: Paint
- Station 2: Wheels
- Station 3: Engine

Each station can be upgraded independently without touching the others.

Modular RAG does the same:
```
[Query Router] → [Retriever] → [Reranker] → [Context Builder] → [Generator] → [Evaluator]
```

Each module is independent. You can swap out just the reranker, or just the retriever, without rebuilding everything. This is what LangGraph, LlamaIndex, and Haystack implement.

---

### Generation 4: Agentic/Adaptive RAG (2024–2025) — "The Smart Employee"

The system **thinks before it acts**.

**Router classifies the question first:**
```
"What's 2 + 2?"
  → Type: Simple math → Answer directly, no retrieval needed

"What did our CEO say in the Q3 report?"
  → Type: Internal document → Search internal vector store

"Who won the 2025 World Cup?"
  → Type: Recent news → Trigger web search tool

"Compare our product with competitor X's product, considering market trends"
  → Type: Multi-hop → Break into sub-questions → retrieve for each
```

This is LangGraph running as a state machine (explained fully in Part 3).

---

### Generation 5: Corrective RAG (CRAG) — "The Self-Checker"

After retrieval, the system grades its own results:

```
Retrieve documents
  ↓
Grade each document: "Is this actually relevant to the question?"
  ↓
IF documents score < 0.7:
  → Rewrite the query
  → Retrieve again (max 2 retries to avoid infinite loop)
  ↓
ELSE:
  → Proceed to generate answer
```

**Real example:**
```
Query: "What is our sick leave policy for remote workers?"

First retrieval → gets "General HR Policy 2019" (outdated, irrelevant)
Grader scores: 0.3 (low relevance)
Rewrite: "Remote employee sick leave rules 2024"
Second retrieval → gets "Remote Work Policy 2024.pdf"
Grader scores: 0.9 (high relevance)
→ Generate answer
```

---

## PART 3: The Tech Stack — What Does Each Tool Do?

### 3.1 LangGraph — "The Traffic Director"

**What problem does it solve?**

Imagine you're building a pipeline with many steps, and some steps need to loop back, some need to go in different directions based on conditions. Normal code (if/else chains) becomes a nightmare to manage.

LangGraph lets you draw this as a **graph** (nodes = actions, edges = paths between them).

**Key concepts:**

**State:** A shared notepad everyone can read and write to
```python
from typing import TypedDict, List

class RAGState(TypedDict):
    question: str          # The user's question
    documents: List[str]   # Retrieved documents
    generation: str        # The LLM's answer
    rewrite_count: int     # How many times we rewrote the query
```

**Nodes:** Functions that do work and update the state
```python
def retrieve(state):
    # Gets documents from the vector store
    docs = vector_store.search(state["question"])
    return {"documents": docs}

def grade_documents(state):
    # Scores each document for relevance
    relevant_docs = []
    for doc in state["documents"]:
        score = grader.score(state["question"], doc)
        if score > 0.7:
            relevant_docs.append(doc)
    return {"documents": relevant_docs}

def generate(state):
    # Asks the LLM to answer using the documents
    answer = llm.answer(state["question"], state["documents"])
    return {"generation": answer}

def rewrite_query(state):
    # Makes a better version of the question
    better_question = llm.rewrite(state["question"])
    count = state["rewrite_count"] + 1
    return {"question": better_question, "rewrite_count": count}
```

**Routing function:** Decides which node to go to next
```python
def should_rewrite(state):
    # If we have relevant docs, generate
    if state["documents"]:
        return "generate"
    # If we've tried rewriting too many times, just generate with what we have
    if state["rewrite_count"] >= 2:
        return "generate"
    # Otherwise, try rewriting the query
    return "rewrite"
```

**Building the graph:**
```python
from langgraph.graph import StateGraph, END

graph = StateGraph(RAGState)

# Add nodes (the workers)
graph.add_node("retrieve", retrieve)
graph.add_node("grade", grade_documents)
graph.add_node("rewrite", rewrite_query)
graph.add_node("generate", generate)

# Add edges (the paths)
graph.set_entry_point("retrieve")            # Start here
graph.add_edge("retrieve", "grade")          # After retrieve → grade
graph.add_conditional_edges(                 # After grade → decide
    "grade",
    should_rewrite,                          # This function decides
    {"generate": "generate", "rewrite": "rewrite"}
)
graph.add_edge("rewrite", "retrieve")        # After rewrite → retrieve again
graph.add_edge("generate", END)              # After generate → done

app = graph.compile()

# Run it
result = app.invoke({"question": "What is our refund policy?", "rewrite_count": 0})
print(result["generation"])
```

**Visual flow:**
```
[START]
   ↓
[Retrieve] → [Grade] → [should_rewrite?]
                             ↓ "generate"    ↓ "rewrite"
                         [Generate]      [Rewrite Query]
                             ↓                 ↓
                           [END]          [Retrieve] (loop back)
```

---

### 3.2 LlamaIndex — "The Document Expert"

LlamaIndex specializes in: loading documents, processing them, and making them searchable. It's better than LangChain for the "store and retrieve" part.

**Loading documents:**
```python
from llama_index.core import SimpleDirectoryReader

# Automatically handles PDF, DOCX, TXT, HTML, Markdown
documents = SimpleDirectoryReader("./my_documents/").load_data()
# Returns a list of Document objects with content + metadata
```

**Ingestion pipeline (the assembly line for documents):**
```python
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore

pipeline = IngestionPipeline(
    transformations=[
        # Step 1: Split documents into chunks
        SentenceSplitter(chunk_size=512, chunk_overlap=64),
        # Step 2: Convert chunks into numbers (embeddings)
        HuggingFaceEmbedding(model_name="BAAI/bge-m3"),
    ],
    # Where to store the final vectors
    vector_store=QdrantVectorStore(client=qdrant_client, collection_name="docs"),
    # Cache: skip re-processing documents that haven't changed
    cache=IngestionCache(),
)

nodes = await pipeline.arun(documents=documents)
print(f"Indexed {len(nodes)} chunks")
```

**The cache is magic:** If you run this pipeline again tomorrow after adding 10 new documents, it only processes those 10 new documents. Everything else is skipped because the content hash hasn't changed.

**Hierarchical chunking (parent-child pattern):**
```python
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes

parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[2048, 512, 128]
    # Level 1: 2048 tokens (a whole section)
    # Level 2: 512 tokens (a paragraph)
    # Level 3: 128 tokens (a few sentences) ← only these go into the vector store
)

nodes = parser.get_nodes_from_documents(documents)
leaf_nodes = get_leaf_nodes(nodes)
# Only index leaf_nodes (128-token chunks) for precise retrieval
# But at query time, return the 512-token parent for full context
```

**Why parent-child?** Think of a book:
- You search the INDEX for a specific term (small = precise)
- But you READ the full CHAPTER to understand the context (large = informative)

---

### 3.3 DSPy — "The Prompt Scientist"

**The problem with regular prompts:**

You write: `"Answer the following question using the provided context. Be concise and accurate."`

Is that the best prompt? Maybe. Maybe not. There are thousands of possible instructions. You'd have to test them all manually.

**DSPy's idea:** Treat prompts as learnable parameters. Let an optimizer find the best instructions automatically.

**Step 1: Define what you want (Signature)**
```python
import dspy

class RAGSignature(dspy.Signature):
    """Answer the question based on the retrieved passages."""
    # Inputs (what goes IN)
    question: str = dspy.InputField(desc="The user's question")
    context: list[str] = dspy.InputField(desc="Retrieved document passages")
    # Output (what comes OUT)
    answer: str = dspy.OutputField(desc="A factual answer supported by the context")
```

**Step 2: Build a module**
```python
class RAGModule(dspy.Module):
    def __init__(self):
        # ChainOfThought = think step by step before answering
        self.generate = dspy.ChainOfThought(RAGSignature)
    
    def forward(self, question, context):
        return self.generate(question=question, context=context)
```

**Step 3: Optimize with MIPROv2**
```python
from dspy.teleprompt import MIPROv2

# Your training examples (question + expected answer pairs)
train_examples = [
    dspy.Example(
        question="What is our return window?",
        context=["Returns accepted within 30 days of purchase..."],
        answer="30 days"
    ),
    # ... more examples
]

# Run the optimizer (takes 10–30 min but runs once)
teleprompter = MIPROv2(metric=dspy.SemanticF1(), auto="medium")
optimized_module = teleprompter.compile(RAGModule(), trainset=train_examples)

# Save for production
optimized_module.save("optimized_rag_prompt.json")
```

**What the optimizer actually does:** It tries hundreds of different instruction phrasings and few-shot examples. It picks the combination that scores highest on your metric (SemanticF1 = "does the answer mean the same as the expected answer?").

**Result:** Accuracy went from 53% → 61% on real benchmarks. That's 10% relative improvement just from better prompts — no model change needed.

**Key insight:** Use DSPy INSIDE LangGraph. LangGraph handles the flow (what step runs when), DSPy handles the prompt quality (how the LLM is instructed at each step).

---

## PART 4: Chunking — Cutting Documents the Right Way

**Why chunking matters:**

Your LLM can only read so much text at once (context window limit). If a document is 100 pages, you can't feed all of it. You need to cut it into pieces. But HOW you cut matters enormously.

**Bad chunking:** Cut every 500 characters
```
"The refund policy states that customers can return products within 30 days. 
Refunds are processed within 5-7 bu"
↑ CUT HERE
"siness days after we receive the returned item."
```
You just split a sentence in half. Now neither chunk makes sense alone.

**Good chunking:** Cut at natural boundaries (sentences, paragraphs, topics).

---

### Chonkie — The Chunking Library

Install: `pip install chonkie`

#### TokenChunker — Fastest, Fixed Size
```python
from chonkie import TokenChunker

chunker = TokenChunker(
    chunk_size=512,      # 512 tokens per chunk
    chunk_overlap=64     # 64 token overlap between consecutive chunks
)

chunks = chunker("Your very long document text here...")
for chunk in chunks:
    print(f"Tokens: {chunk.token_count}, Text: {chunk.text[:50]}...")
```

**Why overlap?** If a key sentence falls at the boundary between two chunks, overlap ensures it appears in at least one chunk fully.

```
Chunk 1: [token 1 ... token 512]
Chunk 2: [token 449 ... token 960]  ← overlaps by 64 tokens
```

#### SentenceChunker — Respects Natural Language
```python
from chonkie import SentenceChunker

chunker = SentenceChunker(
    chunk_size=256,       # Target size in tokens
    chunk_overlap=32,     # Overlap in tokens
    min_sentences_per_chunk=1
)

chunks = chunker("The cat sat on the mat. The dog ran away. It was a sunny day.")
# Never cuts mid-sentence
```

#### RecursiveChunker — Best for Structured Documents
```python
from chonkie import RecursiveChunker

# For Markdown/code: tries to split at headers first,
# then paragraphs, then sentences, then words
chunker = RecursiveChunker(
    chunk_size=512,
    rules=RecursiveRules()  # defaults: \n\n, \n, ". ", " "
)

markdown_text = """
# Section 1
This is the first section content.

# Section 2  
This is the second section content.
"""

chunks = chunker(markdown_text)
# Splits at ## headers first — keeps sections coherent
```

#### SemanticChunker — Splits Where Topics Change

This is the smart one. It detects when the *topic* changes (not just where a paragraph ends).

```python
from chonkie import SemanticChunker

chunker = SemanticChunker(
    embedding_model="minishlab/potion-base-8M",  # tiny model for speed
    threshold=0.5,    # similarity below this = new chunk
    chunk_size=512    # max size limit
)

text = """
The solar system consists of the Sun and everything bound to it by gravity.
The eight planets orbit the Sun. Mars is the fourth planet.
Jupiter is the largest planet in our solar system.

Python is a high-level programming language.
It was created by Guido van Rossum in 1991.
Python is known for its simple, readable syntax.
"""

chunks = chunker(text)
# Chunk 1: All sentences about planets (same topic)
# Chunk 2: All sentences about Python (topic changed)
```

**How it works internally:**
1. Split into sentences
2. Embed each sentence (convert to number)
3. Calculate similarity between consecutive sentences
4. Apply Savitzky-Golay filter (smooths the similarity curve, removes noise)
5. Where similarity drops sharply = topic boundary = split here

#### LateChunker — For Cross-Reference Heavy Documents

```python
from chonkie import LateChunker

chunker = LateChunker(
    embedding_model="BAAI/bge-m3",
    chunk_size=512
)

legal_doc = """
The party hereinafter referred to as "the Contractor" (see Section 2.1 for definition)
shall complete the work by December 31, 2024. The Contractor must adhere to 
all specifications listed in Appendix B.
"""

chunks = chunker(legal_doc)
```

**Why it's different:** Normal chunkers embed each chunk in isolation. "The Contractor" in chunk 5 doesn't know what it refers to from Section 2.1.

LateChunker:
1. Embeds the ENTIRE document first (full context)
2. Every token's embedding contains information from the whole document
3. THEN cuts into chunks, but the embeddings already have full context

Result: 12–18% better retrieval accuracy on legal documents, technical manuals, academic papers.

---

### Hierarchical Chunking — The Best Pattern

```python
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes

# Create 3 levels: big → medium → small
parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[2048, 512, 128]
)

# This creates a tree:
# Document
#   └─ 2048-token section (Level 1)
#        └─ 512-token paragraph (Level 2)
#             └─ 128-token sentences (Level 3) ← INDEX THESE

nodes = parser.get_nodes_from_documents(documents)
leaf_nodes = get_leaf_nodes(nodes)  # only 128-token chunks

# Store ONLY leaf nodes in vector store
# But each leaf knows its parent (512 tokens) and grandparent (2048 tokens)
```

**At query time (Small-to-Big Retrieval):**
```
User query → Search 128-token chunks (high precision, find exact match)
Found: "30-day return window" (128 tokens) → too small for context
Fetch parent: 512-token paragraph (contains the full refund policy)
Send this 512-token chunk to LLM → LLM has full context
```

**Think of it like a book:**
- Index entry: "refund, page 47" (small, precise)
- You open to page 47 and read the whole paragraph (big, contextual)

---

### Contextual Chunking (Anthropic's Technique)

**The problem:** A chunk often loses context when separated from its document.

Example chunk: "It grew by 23% year over year."

What is "it"? We don't know without the document context.

**The fix:** Before embedding, prepend a short context description:
```
"[Document: Q3 2024 Earnings Report | Section: North America Revenue]
It grew by 23% year over year."
```

Now when someone searches for "North America revenue growth", this chunk matches perfectly.

**Implementation:**
```python
def add_context_to_chunk(chunk: str, document_title: str, section: str, 
                          local_llm) -> str:
    prompt = f"""
    Document: {document_title}
    Section: {section}
    Chunk: {chunk}
    
    Write a brief 1-2 sentence context that explains what this chunk is about
    within the document. Be specific.
    """
    context = local_llm.generate(prompt)  # e.g., Ollama
    return f"[Context: {context}]\n\n{chunk}"

# Do this at ingestion time, store the result
# Never do this at query time (too slow)
```

**Results:**
- Contextual embeddings alone: 35% fewer retrieval failures
- + BM25 hybrid: 49% fewer failures
- + Reranking on top: 67% fewer failures

---

## PART 5: Vector Databases — Where You Store the Chunks

**What is a vector database?**

A regular database stores text and numbers. A vector database stores **embeddings** (arrays of numbers that represent meaning) and can find the most similar ones very fast.

**Analogy:** Imagine every document chunk is a point in 3D space. Similar chunks are close together. When you search, you find the nearest points to your query point.

In reality it's 768 dimensions, not 3. But same idea.

---

### Qdrant — The Production Choice

**Why Qdrant:**
- Written in Rust (fast + memory safe)
- Filterable HNSW (filter AND search at the same time, not after)
- INT8 quantization (4x smaller, <2% accuracy loss)
- Hybrid search (dense + sparse vectors)
- Docker deployment in one command

**Start Qdrant:**
```bash
docker run -p 6333:6333 -v ./qdrant_storage:/qdrant/storage qdrant/qdrant
# Now available at http://localhost:6333
```

**Create a collection:**
```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, HnswConfigDiff,
    ScalarQuantizationConfig, QuantizationType
)

client = QdrantClient(url="http://localhost:6333")

client.create_collection(
    collection_name="my_documents",
    vectors_config=VectorParams(
        size=768,                    # BGE-M3 produces 768-dim vectors
        distance=Distance.COSINE,    # Cosine similarity (best for text)
        hnsw_config=HnswConfigDiff(
            m=16,            # Connections per node. Higher = better recall, more memory
            ef_construct=200 # How hard it works during indexing. Higher = more accurate
        ),
        quantization_config=ScalarQuantizationConfig(
            type=QuantizationType.INT8,  # Compress 32-bit floats to 8-bit integers
            quantile=0.99,               # Keep 99th percentile as reference
            always_ram=True              # Keep quantized vectors in RAM (faster)
        ),
    ),
)
```

**Add vectors:**
```python
from qdrant_client.models import PointStruct

# Embed your chunks first
embeddings = embed_model.encode(["chunk text 1", "chunk text 2"])

# Insert into Qdrant
client.upsert(
    collection_name="my_documents",
    points=[
        PointStruct(
            id=1,
            vector=embeddings[0].tolist(),
            payload={                       # Metadata — can filter on these
                "text": "chunk text 1",
                "source": "document.pdf",
                "page": 5,
                "section": "Refund Policy",
                "date": "2024-01-15"
            }
        ),
        PointStruct(
            id=2,
            vector=embeddings[1].tolist(),
            payload={"text": "chunk text 2", "source": "manual.pdf"}
        ),
    ]
)
```

**Search with filtering:**
```python
from qdrant_client.models import Filter, FieldCondition, MatchValue

query_vector = embed_model.encode(["What is the refund policy?"])[0]

results = client.search(
    collection_name="my_documents",
    query_vector=query_vector.tolist(),
    query_filter=Filter(                    # Only search within these constraints
        must=[
            FieldCondition(
                key="section",
                match=MatchValue(value="Refund Policy")
            )
        ]
    ),
    limit=10,                               # Return top 10
    with_payload=True                       # Include the text
)

for result in results:
    print(f"Score: {result.score:.3f} | Text: {result.payload['text'][:100]}")
```

**What is HNSW?** (You don't need to understand the math, just the concept)

Imagine you're looking for the nearest coffee shop. Instead of checking every coffee shop in the city (slow), you:
1. First check a few "hub" locations
2. From the closest hub, check their neighbors
3. Keep narrowing down

HNSW does this for vectors. `m=16` means each point connects to 16 neighbors. More connections = more accurate but more memory.

---

## PART 6: Local LLM Inference — Running AI Without Internet

### vLLM — Production Server

**What problem does it solve?**

Regular inference: serve one request, wait for it to finish, then serve the next. GPU sits idle half the time.

vLLM: serve MANY requests simultaneously, GPU never idles. 2–35x more throughput.

**Key innovations:**

**PagedAttention:**
```
Normal: "I need memory for 1000 tokens. Reserve a block."
         → Block is only 60% used. 40% wasted. (Memory fragmentation)

PagedAttention: Memory is split into small pages.
                Fill pages as needed. No waste.
```

**Continuous Batching:**
```
Normal:
  Batch [Request A, Request B] → Wait for both to finish → Start new batch
  (If A finishes early, GPU waits for B)

Continuous Batching:
  [A, B] in flight → A finishes → [C, B] immediately → B finishes → [C, D]...
  GPU always working at full capacity
```

**Start vLLM server:**
```bash
pip install vllm

# With GPU (recommended)
python -m vllm.entrypoints.openai.api_server \
  --model "Qwen/Qwen3-14B" \
  --max-model-len 32768 \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.90 \
  --quantization awq

# It exposes an OpenAI-compatible API at http://localhost:8000
```

**Use it from Python:**
```python
import openai

# Point to your local vLLM server — same code as using OpenAI API
client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # vLLM doesn't need a key locally
)

response = client.chat.completions.create(
    model="Qwen/Qwen3-14B",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is our refund policy?"}
    ],
    max_tokens=1024,
    stream=True  # Get tokens as they're generated (feels faster)
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

---

### Ollama — For Development and Single Users

```bash
# Install: https://ollama.ai
# Pull a model
ollama pull llama3.2:latest
ollama pull qwen2.5:14b

# It runs automatically on port 11434
```

```python
import ollama

response = ollama.chat(
    model='qwen2.5:14b',
    messages=[{'role': 'user', 'content': 'What is our refund policy?'}],
    stream=True
)

for chunk in response:
    print(chunk['message']['content'], end='', flush=True)
```

**When to use which:**

| Situation | Use |
|---|---|
| Building and testing locally | Ollama |
| 1 person using it | Ollama |
| 10+ simultaneous users | vLLM |
| Production server with GPU | vLLM |
| CPU only, single user | Ollama |
| Raspberry Pi / embedded | llama.cpp directly |

---

## PART 7: Embedding Models — Converting Text to Numbers

**What is an embedding?**

It's a list of numbers that represents the MEANING of text.

```
"I love dogs"              → [0.21, -0.53, 0.88, 0.04, ...]  (768 numbers)
"I adore puppies"          → [0.23, -0.51, 0.85, 0.06, ...]  (similar numbers!)
"The stock market crashed" → [0.91,  0.12, -0.44, ...]       (very different numbers)
```

Similar meanings → similar numbers → close together in vector space → retrieved together.

---

### BGE-M3 — The Best All-Around Choice

**Why BGE-M3 is special:** One model, three types of retrieval:

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

texts = ["What is RAG?", "Retrieval Augmented Generation is a technique..."]

# 1. Dense embeddings (semantic similarity)
dense = model.encode(texts, return_dense=True)['dense_vecs']

# 2. Sparse embeddings (keyword matching, like BM25 but learned)
sparse = model.encode(texts, return_sparse=True)['lexical_weights']

# 3. ColBERT vectors (token-level for precise matching)
colbert = model.encode(texts, return_colbert_vecs=True)['colbert_vecs']

# For production, use dense + sparse together (hybrid search)
```

**Practical installation and usage:**
```python
# Simple way with sentence-transformers
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('BAAI/bge-m3')

# Encoding queries (use query prefix)
query_embedding = model.encode(
    "What is the refund policy?",
    prompt="Represent this sentence for searching relevant passages: "
)

# Encoding documents (no prefix needed)
doc_embeddings = model.encode([
    "Refunds are accepted within 30 days",
    "Our return policy allows exchanges",
    "Shipping takes 3-5 business days"
])

# Find most similar document
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
best_match_idx = np.argmax(similarities)
print(f"Best match: {doc_embeddings[best_match_idx]}")
print(f"Similarity: {similarities[best_match_idx]:.3f}")
```

---

## PART 8: The Full Retrieval Pipeline

This is the core of production RAG. Let's build it step by step.

### Step 1: Hybrid Search (Dense + Sparse)

**Why hybrid?**

Dense (semantic): Great for synonym matching, conceptual questions
Sparse (BM25): Great for exact keywords, product codes, names

**Example where each fails alone:**
```
Query: "product XK-229 refund"

Dense only: Finds docs about "product returns" but misses XK-229 (doesn't know that code)
Sparse only: Finds docs containing "XK-229" but misses related refund docs

Hybrid: Finds docs that both contain "XK-229" AND talk about refunds ✓
```

**BM25 in Python:**
```python
from rank_bm25 import BM25Okapi

# Your document chunks
corpus = [
    "Refunds are accepted within 30 days of purchase",
    "Product XK-229 has a special 90-day warranty",
    "Shipping costs are non-refundable",
    "Contact customer service for returns"
]

# Tokenize (split into words)
tokenized_corpus = [doc.lower().split() for doc in corpus]

# Build BM25 index
bm25 = BM25Okapi(tokenized_corpus)

# Query
query = "XK-229 refund policy"
tokenized_query = query.lower().split()

# Get BM25 scores
scores = bm25.get_scores(tokenized_query)
# Returns: [0.3, 0.8, 0.2, 0.1] — XK-229 doc scores highest!

# Get top results
top_n = bm25.get_top_n(tokenized_query, corpus, n=10)
```

### Step 2: Reciprocal Rank Fusion (RRF) — Combining Results

**Problem:** Dense retrieval returns its own ranking. BM25 returns its own ranking. How do you combine them?

**RRF solution:** Don't care about scores, only care about RANK POSITION.

```python
def reciprocal_rank_fusion(
    dense_results: list,    # [(doc_id, content), ...]  ordered by dense score
    sparse_results: list,   # [(doc_id, content), ...]  ordered by BM25 score
    k: int = 60             # constant (60 works well empirically)
) -> list:
    
    scores = {}
    
    # Score based on RANK (position 1 is best)
    # Formula: 1 / (k + rank)  — rank starts at 1
    for rank, doc in enumerate(dense_results, start=1):
        doc_id = doc[0]
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    
    for rank, doc in enumerate(sparse_results, start=1):
        doc_id = doc[0]
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    
    # Sort by combined score
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

# Example:
# Dense rank:  [Doc_A(1st), Doc_C(2nd), Doc_B(3rd)]
# Sparse rank: [Doc_B(1st), Doc_A(2nd), Doc_D(3rd)]
# 
# Doc_A score: 1/61 + 1/62 = 0.0164 + 0.0161 = 0.0325
# Doc_B score: 1/63 + 1/61 = 0.0159 + 0.0164 = 0.0323
# Doc_C score: 1/62 = 0.0161
# Doc_D score: 1/63 = 0.0159
#
# Final order: Doc_A, Doc_B, Doc_C, Doc_D
# Doc_A wins because it was high in BOTH rankings
```

**Why k=60?** Research shows this value minimizes sensitivity to outliers. Don't overthink it.

### Step 3: HyDE — Searching With a Hypothetical Answer

**The problem:** Your query is a question. Your documents are answers. They use different vocabulary.

```
Query:   "How do I get my money back?"
Document: "Customers may initiate a refund request via the customer portal."

These might not have high similarity even though they match!
```

**HyDE trick:** Generate a hypothetical document that ANSWERS the question. Then search with THAT.

```python
def hyde_retrieval(query: str, llm, vector_store, embed_model) -> list:
    
    # Step 1: Generate a hypothetical answer
    hyde_prompt = f"""
    Write a detailed passage that would answer this question:
    "{query}"
    
    Write it as if it came from an official company document.
    """
    
    hypothetical_doc = llm.generate(hyde_prompt)
    # → "Customers can request refunds within 30 days by visiting the 
    #    customer portal and selecting 'Return Item'. Refunds are processed
    #    within 5-7 business days..."
    
    # Step 2: Embed the HYPOTHETICAL ANSWER (not the original question)
    hyde_embedding = embed_model.encode(hypothetical_doc)
    
    # Step 3: Retrieve using that embedding
    results = vector_store.search(hyde_embedding, top_k=20)
    
    return results

# Why this works: the hypothetical answer uses DOCUMENT vocabulary,
# so it finds similar documents much more reliably.
```

### Step 4: Multi-Query Expansion

Different phrasings catch different documents:

```python
def multi_query_expansion(query: str, llm, vector_store, embed_model) -> list:
    
    # Generate rephrased versions
    prompt = f"""
    Generate 3 different ways to ask this question. 
    Focus on different aspects and vocabulary.
    Question: {query}
    
    Return as a numbered list.
    """
    
    rephrased = llm.generate(prompt)
    # Output:
    # 1. "What is the return window for purchases?"
    # 2. "How long do I have to send back a product?"
    # 3. "Can I get a refund after 30 days?"
    
    all_queries = [query] + parse_numbered_list(rephrased)
    
    # Retrieve for EACH query
    all_results = {}
    for q in all_queries:
        embedding = embed_model.encode(q)
        results = vector_store.search(embedding, top_k=10)
        for r in results:
            all_results[r.id] = r  # deduplicate by ID
    
    return list(all_results.values())
```

### Step 5: MMR — Removing Duplicates While Keeping Diversity

**Problem:** If your document has "The refund period is 30 days" in 5 different sections, you might retrieve all 5. That wastes your context window.

**MMR** (Maximal Marginal Relevance) selects results that are:
1. **Relevant** to the query AND
2. **Different** from already-selected results

```python
# LangChain makes this easy:
results = vector_store.max_marginal_relevance_search(
    query="What is the refund policy?",
    k=5,          # Return 5 final results
    fetch_k=20,   # Start with 20 candidates
    lambda_mult=0.7  # 0 = max diversity, 1 = max relevance, 0.7 = balanced
)
```

**Manual MMR logic:**
```python
import numpy as np

def mmr(query_embedding, doc_embeddings, k=5, lambda_mult=0.7):
    selected = []
    remaining = list(range(len(doc_embeddings)))
    
    for _ in range(k):
        best_score = -float('inf')
        best_idx = None
        
        for idx in remaining:
            # Relevance: similarity to query
            relevance = cosine_sim(query_embedding, doc_embeddings[idx])
            
            # Diversity: how different from already selected?
            if selected:
                max_similarity = max(
                    cosine_sim(doc_embeddings[idx], doc_embeddings[s])
                    for s in selected
                )
            else:
                max_similarity = 0
            
            # MMR score: balance relevance and diversity
            score = lambda_mult * relevance - (1 - lambda_mult) * max_similarity
            
            if score > best_score:
                best_score = score
                best_idx = idx
        
        selected.append(best_idx)
        remaining.remove(best_idx)
    
    return selected
```

### Step 6: Reranking — The Final Filter

After all retrieval steps, you have 20 candidate chunks. But the LLM can only handle 5 well. A reranker picks the best 5.

**Why not just use the retrieval scores?** Embedding similarity is fast but approximate. A cross-encoder reranker does full attention between the query AND each document — much more accurate.

**CrossEncoder reranker:**
```python
from sentence_transformers import CrossEncoder

# Load the reranker model (downloaded once, runs locally)
reranker = CrossEncoder('BAAI/bge-reranker-v2-m3')

query = "What is the refund policy?"
candidates = [
    "Refunds are accepted within 30 days",
    "Our products come with a warranty",
    "Returns must include original packaging",
    "Shipping costs are non-refundable",
    "Contact us at support@company.com",
]

# Score each (query, document) pair together
scores = reranker.predict([(query, doc) for doc in candidates])

# Sort by score
ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

# Take top 3
top_3 = [doc for doc, score in ranked[:3]]
print(top_3)
# → ["Refunds are accepted within 30 days",
#    "Returns must include original packaging",
#    "Shipping costs are non-refundable"]
```

**FlashRank for CPU (faster, slightly less accurate):**
```python
from flashrank import Ranker, RerankRequest

ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")

request = RerankRequest(
    query="What is the refund policy?",
    passages=[{"text": doc} for doc in candidates]
)

results = ranker.rerank(request)
# <20ms on CPU!
```

---

## PART 9: Memory Systems

**The problem:** Every conversation starts fresh. The AI doesn't remember:
- Your name
- What you discussed yesterday
- Your preferences
- Previous decisions made

**Memory taxonomy:**

```
Memory Types:
├── Short-term (conversation buffer)
│     "What was said in the last 5 messages?"
│     → Just keep last N messages in the prompt
│
├── Summary memory
│     "Summarize the conversation so far"
│     → LLM compresses old messages into summary
│     → Save context window space
│
├── Entity memory
│     "Extract and track: [Alice, Project Phoenix, Q3 deadline]"
│     → When user says "it", know what "it" refers to
│
└── Long-term semantic memory
      "What relevant things did this user tell us in past sessions?"
      → Store all conversations in vector DB
      → Retrieve relevant past exchanges at query time
```

### Mem0 — Production Memory Library

```bash
pip install mem0ai
```

```python
from mem0 import Memory

# Configure with local models
config = {
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "llama3.2",
            "ollama_base_url": "http://localhost:11434"
        }
    },
    "embedder": {
        "provider": "huggingface",
        "config": {"model": "BAAI/bge-m3"}
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {"host": "localhost", "port": 6333}
    }
}

memory = Memory.from_config(config)

# ============ ADDING MEMORIES ============

# Explicit memory
memory.add(
    "User prefers technical explanations with code examples",
    user_id="user_123"
)

# From conversation (Mem0 extracts important facts automatically)
messages = [
    {"role": "user", "content": "I'm building a RAG system for a hospital"},
    {"role": "assistant", "content": "Great! For healthcare, HIPAA compliance is key..."},
    {"role": "user", "content": "I have about 500GB of medical records to index"},
]
memory.add(messages, user_id="user_123")
# Mem0 automatically extracts: "Building RAG for hospital, 500GB medical records"

# ============ RETRIEVING MEMORIES ============

# Get relevant memories for current query
relevant_memories = memory.search(
    "How should I handle patient data?",
    user_id="user_123"
)

for mem in relevant_memories["results"]:
    print(f"Memory: {mem['memory']}")
    print(f"Relevance: {mem['score']:.3f}")
    print()
# Output:
# Memory: Building RAG for hospital, 500GB medical records
# Relevance: 0.92
# Memory: User prefers technical explanations with code examples
# Relevance: 0.71

# ============ USE IN RAG ============

def rag_with_memory(question: str, user_id: str):
    # 1. Get relevant memories
    memories = memory.search(question, user_id=user_id)
    memory_context = "\n".join([m['memory'] for m in memories["results"]])
    
    # 2. Retrieve relevant documents
    docs = retriever.retrieve(question)
    
    # 3. Generate answer with memory context
    prompt = f"""
    User context (from memory):
    {memory_context}
    
    Relevant documents:
    {docs}
    
    Question: {question}
    
    Answer:
    """
    return llm.generate(prompt)
```

**What Mem0 does under the hood:**
1. Takes conversation → asks LLM "what facts are worth remembering from this?"
2. Checks if similar memories already exist → deduplicates
3. Stores in vector DB
4. At query time: embed query → find relevant memories → inject into prompt

**Results vs naive "store everything" approach:**
- 91% lower latency (doesn't search 100k past messages, only relevant ones)
- 90% fewer tokens used (only inject what's relevant)
- 26% better accuracy (relevant context, not noise)

---

## PART 10: Latency Optimization

**What is latency?** The time from user asking a question to seeing the answer.

**Production targets:**
- TTFT (Time to First Token): < 1 second (feels responsive)
- Full answer: < 10 seconds (acceptable for complex queries)

### Multi-Level Caching

**Level 1: Exact Cache**
```python
import redis
import hashlib
import json

r = redis.Redis(host='localhost', port=6379)

def query_with_exact_cache(question: str) -> str:
    # Create a unique key for this exact question
    cache_key = f"exact:{hashlib.md5(question.encode()).hexdigest()}"
    
    # Check if we've seen this exact question before
    cached = r.get(cache_key)
    if cached:
        return json.loads(cached)  # Return instantly!
    
    # Not cached → run the full pipeline
    answer = full_rag_pipeline(question)
    
    # Store for next time (expires in 1 hour)
    r.setex(cache_key, 3600, json.dumps(answer))
    return answer
```

**Level 2: Semantic Cache (Most Impactful)**

Different questions, same answer:
- "What's your return policy?"
- "How do I return a product?"
- "Can I get a refund?"

These are different strings but same MEANING. Exact cache won't help. Semantic cache will.

```python
from sentence_transformers import SentenceTransformer
import numpy as np

class SemanticCache:
    def __init__(self, similarity_threshold=0.95):
        self.embed_model = SentenceTransformer('BAAI/bge-m3')
        self.threshold = similarity_threshold
        self.cache = []  # In production: use Qdrant as cache store
    
    def get(self, question: str):
        question_embedding = self.embed_model.encode(question)
        
        # Find the most similar cached question
        best_similarity = 0
        best_answer = None
        
        for cached_q, cached_a, cached_emb in self.cache:
            similarity = np.dot(question_embedding, cached_emb)  # cosine sim
            if similarity > best_similarity:
                best_similarity = similarity
                best_answer = cached_a
        
        if best_similarity >= self.threshold:
            print(f"Cache hit! Similarity: {best_similarity:.3f}")
            return best_answer
        return None
    
    def set(self, question: str, answer: str):
        embedding = self.embed_model.encode(question)
        self.cache.append((question, answer, embedding))

# Usage
cache = SemanticCache(similarity_threshold=0.95)

def cached_rag(question):
    # Check semantic cache first
    cached_answer = cache.get(question)
    if cached_answer:
        return cached_answer  # Milliseconds!
    
    # Full pipeline (seconds)
    answer = full_rag_pipeline(question)
    cache.set(question, answer)
    return answer

# First call: "What is the refund policy?" → runs full pipeline (5 seconds)
# Second call: "How do I get a refund?" → cache hit! (50ms) ← 65x faster!
```

### Async Parallel Execution

Instead of running steps one after another, run them simultaneously:

```python
import asyncio

async def fast_retrieval(question: str):
    # These two operations don't depend on each other
    # Run them at the SAME TIME
    
    embed_task = asyncio.create_task(
        embed_model.aencode(question)  # async embedding
    )
    bm25_task = asyncio.create_task(
        bm25_search(question, k=20)    # async BM25 search
    )
    
    # Wait for BOTH to finish (total time = max(embed_time, bm25_time))
    query_embedding, sparse_results = await asyncio.gather(embed_task, bm25_task)
    
    # Now do dense retrieval (needs the embedding)
    dense_results = await qdrant_client.asearch(
        collection_name="docs",
        query_vector=query_embedding.tolist(),
        limit=20
    )
    
    # Merge results
    merged = reciprocal_rank_fusion(dense_results, sparse_results)
    return merged

# Without async: embed(2s) + BM25(1s) + dense search(1s) = 4s total
# With async:    max(embed, BM25)(2s) + dense(1s) = 3s total  ← 25% faster
```

### Streaming Responses

```python
async def stream_rag_response(question: str):
    # Retrieve documents (not streamable, must complete first)
    docs = await retrieve_documents(question)
    
    # Stream the generation token by token
    async for token in llm.astream_chat(
        context=docs,
        question=question
    ):
        yield token  # Send each token immediately as it's generated

# The user sees the first word in <1 second
# Even if the total response takes 10 seconds, it FEELS fast
```

---

## PART 11: Full Ingestion Pipeline (PTI)

This runs ONCE (or when documents update). Never at query time.

```python
import asyncio
from pathlib import Path
from chonkie import SemanticChunker
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from rank_bm25 import BM25Okapi
import pickle

async def ingest_documents(documents_folder: str):
    
    # ============ STEP 1: PARSE ============
    print("Parsing documents...")
    
    parsed_docs = []
    for file_path in Path(documents_folder).glob("**/*"):
        if file_path.suffix.lower() == ".pdf":
            # Docling: best for PDFs with tables
            from docling.document_converter import DocumentConverter
            converter = DocumentConverter()
            result = converter.convert(str(file_path))
            text = result.document.export_to_markdown()  # Preserves structure
            
        elif file_path.suffix.lower() in [".txt", ".md"]:
            text = file_path.read_text(encoding="utf-8")
        
        parsed_docs.append({
            "text": text,
            "source": str(file_path),
            "filename": file_path.name
        })
    
    print(f"Parsed {len(parsed_docs)} documents")
    
    # ============ STEP 2: CHUNK ============
    print("Chunking documents...")
    
    chunker = SemanticChunker(
        embedding_model="minishlab/potion-base-8M",  # fast model for chunking
        chunk_size=512,
        threshold=0.5
    )
    
    all_chunks = []
    for doc in parsed_docs:
        chunks = chunker(doc["text"])
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "id": f"{doc['filename']}_chunk_{i}",
                "text": chunk.text,
                "source": doc["source"],
                "chunk_index": i,
                "total_chunks": len(chunks)
            })
    
    print(f"Created {len(all_chunks)} chunks")
    
    # ============ STEP 3: ADD CONTEXT (Anthropic-style) ============
    print("Adding contextual prefixes...")
    
    async def add_context(chunk: dict, doc_title: str) -> dict:
        prompt = f"""
        Document: {doc_title}
        Chunk: {chunk['text'][:200]}...
        
        Write ONE sentence describing what this chunk is about.
        Be specific. Start with "This chunk discusses..."
        """
        context = await ollama_client.agenerate(prompt)
        chunk["text"] = f"[{context}]\n\n{chunk['text']}"
        return chunk
    
    # Process in parallel batches of 10
    for i in range(0, len(all_chunks), 10):
        batch = all_chunks[i:i+10]
        tasks = [add_context(c, c["source"]) for c in batch]
        await asyncio.gather(*tasks)
    
    # ============ STEP 4: EMBED ============
    print("Generating embeddings...")
    
    embed_model = SentenceTransformer("BAAI/bge-m3")
    texts = [chunk["text"] for chunk in all_chunks]
    
    # Batch embed (much faster than one-by-one)
    embeddings = embed_model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True  # Normalize for cosine similarity
    )
    
    # ============ STEP 5: INDEX INTO QDRANT ============
    print("Indexing into Qdrant...")
    
    client = QdrantClient(url="http://localhost:6333")
    
    points = [
        PointStruct(
            id=i,
            vector=embeddings[i].tolist(),
            payload=all_chunks[i]  # Store full metadata
        )
        for i in range(len(all_chunks))
    ]
    
    # Upsert in batches of 100
    for i in range(0, len(points), 100):
        client.upsert(
            collection_name="documents",
            points=points[i:i+100]
        )
    
    # ============ STEP 6: BUILD BM25 INDEX ============
    print("Building BM25 index...")
    
    tokenized = [chunk["text"].lower().split() for chunk in all_chunks]
    bm25 = BM25Okapi(tokenized)
    
    # Save BM25 index to disk
    with open("bm25_index.pkl", "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": all_chunks}, f)
    
    print(f"Done! Indexed {len(all_chunks)} chunks")
    return len(all_chunks)


# Run ingestion
asyncio.run(ingest_documents("./my_documents/"))
```

---

## PART 12: Evaluation — How Do You Know Your RAG is Good?

**RAGAS** gives you 4 metrics. You need NO ground truth labels for most of them.

```bash
pip install ragas datasets
```

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,      # Does the answer only use info from the retrieved docs?
    answer_relevancy,  # Does the answer actually address the question?
    context_precision, # Are the BEST docs ranked first in retrieval?
    context_recall     # Did we retrieve ALL the docs needed to answer?
)
from datasets import Dataset

# Collect test data from your running system
test_data = {
    "question": [
        "What is the refund policy?",
        "How long does shipping take?",
        "Can I return used products?"
    ],
    "answer": [
        "Refunds are accepted within 30 days",     # ← what your RAG generated
        "Shipping takes 3-5 business days",
        "Used products cannot be returned"
    ],
    "contexts": [
        ["Refunds are accepted within 30 days of purchase..."],  # ← what was retrieved
        ["Standard shipping 3-5 days, express 1-2 days..."],
        ["Only unopened items in original packaging accepted..."]
    ],
    "ground_truth": [
        "30 day return window",     # ← optional; needed only for context_recall
        "3-5 business days",
        "No, used items cannot be returned"
    ]
}

result = evaluate(
    Dataset.from_dict(test_data),
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    # Use local LLM as judge (no OpenAI needed)
    llm=your_local_llm,
    embeddings=your_local_embedder
)

print(result)
# {
#   'faithfulness': 0.87,       # Good! Above 0.8 threshold
#   'answer_relevancy': 0.91,
#   'context_precision': 0.79,  # ← Border case. Improve retrieval ranking.
#   'context_recall': 0.85
# }
```

**What each metric means in plain English:**

**Faithfulness (target > 0.8)**
"Did the AI only use information from the retrieved documents, or did it hallucinate?"

```
Retrieved doc: "Refunds accepted within 30 days"
Answer: "Refunds accepted within 60 days" ← WRONG! Not in the doc → Low faithfulness
Answer: "Refunds accepted within 30 days" ← Good → High faithfulness
```

**Answer Relevancy (target > 0.8)**
"Does the answer actually address the question?"

```
Question: "How do I return a product?"
Answer: "Our company was founded in 2005..." ← Irrelevant → Low relevancy
Answer: "Visit our returns portal at..."     ← Relevant   → High relevancy
```

**Context Precision (target > 0.8)**
"Are the best documents ranked first, or buried at position 15?"

```
Top 5 retrieved:
1. Random HR policy     ← not relevant
2. Shipping policy      ← not relevant
3. REFUND POLICY        ← this is what we needed!
4. Vacation policy
5. Benefits guide

Context Precision is LOW because the relevant doc is at position 3, not 1.
```

**Context Recall (target > 0.7)**
"Did we retrieve ALL the information needed to answer the question?"

```
Question: "What are the refund rules for international orders?"
Answer needs: [refund policy] + [international shipping policy]

Retrieved: [refund policy only] ← missed half of what's needed → Low recall
```

---

## PART 13: Putting It All Together — Complete Production RAG System

```python
# production_rag.py

import asyncio
import pickle
from typing import AsyncGenerator
import numpy as np
from qdrant_client import AsyncQdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from mem0 import Memory
import openai  # pointing to local vLLM

class ProductionRAG:
    def __init__(self):
        # Embedding model
        self.embed_model = SentenceTransformer("BAAI/bge-m3")
        
        # Reranker
        self.reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
        
        # Vector store
        self.qdrant = AsyncQdrantClient(url="http://localhost:6333")
        
        # Load BM25 index
        with open("bm25_index.pkl", "rb") as f:
            data = pickle.load(f)
            self.bm25 = data["bm25"]
            self.chunks = data["chunks"]
        
        # LLM (local vLLM server)
        self.llm = openai.AsyncOpenAI(
            base_url="http://localhost:8000/v1",
            api_key="not-needed"
        )
        
        # Memory
        self.memory = Memory.from_config({
            "llm": {"provider": "ollama", "config": {"model": "llama3.2"}},
            "embedder": {"provider": "huggingface", "config": {"model": "BAAI/bge-m3"}},
            "vector_store": {"provider": "qdrant", "config": {"host": "localhost", "port": 6333}}
        })
    
    async def retrieve(self, query: str, top_k: int = 20) -> list:
        """Hybrid retrieval: dense + sparse, fused with RRF"""
        
        # Run embedding and BM25 in parallel
        embed_task = asyncio.create_task(
            asyncio.to_thread(self.embed_model.encode, query, normalize_embeddings=True)
        )
        bm25_task = asyncio.create_task(
            asyncio.to_thread(self._bm25_search, query, top_k)
        )
        
        query_embedding, sparse_results = await asyncio.gather(embed_task, bm25_task)
        
        # Dense retrieval
        dense_hits = await self.qdrant.search(
            collection_name="documents",
            query_vector=query_embedding.tolist(),
            limit=top_k,
            with_payload=True
        )
        dense_results = [(hit.id, hit.payload["text"]) for hit in dense_hits]
        
        # Fuse with RRF
        fused = self._rrf(dense_results, sparse_results, k=60)
        
        # Return top candidates
        all_chunks_dict = {c["id"]: c for c in self.chunks}
        return [all_chunks_dict.get(doc_id, {"text": ""}) for doc_id, _ in fused[:top_k]]
    
    def _bm25_search(self, query: str, top_k: int) -> list:
        tokenized = query.lower().split()
        scores = self.bm25.get_scores(tokenized)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i]["id"], self.chunks[i]["text"]) for i in top_indices]
    
    def _rrf(self, dense: list, sparse: list, k: int = 60) -> list:
        scores = {}
        for rank, (doc_id, _) in enumerate(dense, 1):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
        for rank, (doc_id, _) in enumerate(sparse, 1):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    def rerank(self, query: str, candidates: list, top_n: int = 5) -> list:
        """Rerank candidates with cross-encoder"""
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [c for c, _ in ranked[:top_n]]
    
    async def generate_stream(
        self,
        question: str,
        context_chunks: list,
        user_id: str
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response"""
        
        # Get relevant memories
        memories = self.memory.search(question, user_id=user_id)
        memory_text = "\n".join([m["memory"] for m in memories["results"][:3]])
        
        # Build context
        context = "\n\n---\n\n".join([c["text"] for c in context_chunks])
        sources = list(set([c["source"] for c in context_chunks]))
        
        prompt = f"""You are a helpful assistant. Answer the question using ONLY the provided context.
If the answer is not in the context, say "I don't have information about that."

User context (from memory):
{memory_text}

Retrieved context:
{context}

Question: {question}

Answer:"""
        
        response = await self.llm.chat.completions.create(
            model="Qwen/Qwen3-14B",
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            max_tokens=1024,
            temperature=0.1  # Low temperature for factual answers
        )
        
        async for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
        
        # Add source citations
        yield f"\n\n**Sources:** {', '.join(sources)}"
    
    async def query(self, question: str, user_id: str = "default") -> AsyncGenerator[str, None]:
        """Full RAG pipeline"""
        
        # Step 1: Retrieve (hybrid)
        candidates = await self.retrieve(question, top_k=20)
        
        # Step 2: Rerank (top 20 → top 5)
        top_chunks = self.rerank(question, candidates, top_n=5)
        
        # Step 3: Generate (streaming)
        async for token in self.generate_stream(question, top_chunks, user_id):
            yield token
        
        # Step 4: Store interaction in memory
        asyncio.create_task(asyncio.to_thread(
            self.memory.add,
            [{"role": "user", "content": question}],
            user_id=user_id
        ))


# ============ USAGE ============

async def main():
    rag = ProductionRAG()
    
    print("Question: What is the refund policy?")
    print("Answer: ", end="")
    
    async for token in rag.query("What is the refund policy?", user_id="user_123"):
        print(token, end="", flush=True)
    
    print("\n")

asyncio.run(main())
```

---

## PART 14: Key Numbers Cheat Sheet

```
CHUNKING
├── Chunk size:          512 tokens (leaf), 1024-2048 (parent)
├── Overlap:             10-15% = 50-64 tokens
└── Contextual prefix:   50-100 tokens prepended per chunk

RETRIEVAL
├── Retrieve:            20 candidates (top-20)
├── After rerank:        5 chunks → LLM
└── RRF constant:        k = 60

QDRANT
├── HNSW m:              16 (connections per node)
├── ef_construct:        200 (build-time search depth)
└── Quantization:        INT8 (4x smaller, <2% accuracy loss)

CACHING
├── Semantic threshold:  0.95 cosine similarity
├── Retrieval cache TTL: 60-300 seconds (Redis)
└── Cache hit latency:   <50ms vs 5-10s full pipeline

MMR
└── lambda = 0.7         (0=max diversity, 1=max relevance)

QUALITY TARGETS (RAGAS)
├── Faithfulness:        > 0.8
├── Answer Relevancy:    > 0.8
├── Context Precision:   > 0.8
└── Context Recall:      > 0.7

LOCAL MODELS (by GPU memory)
├── 4-6GB:   Qwen3-4B + Ollama
├── 8GB:     Phi-4 Q4 + Ollama
├── 12-16GB: Qwen3-14B + vLLM
└── 24GB+:   Mistral Small 3 24B + vLLM

EMBEDDING
└── BGE-M3: 768 dims, 8192 token context, ~50ms latency
```

---

## Learning Path — What to Build First

```
Week 1: Basic working RAG
  → Ollama + LlamaIndex + Chroma
  → TokenChunker → BGE-M3 → simple retrieval → answer
  → Goal: understand the flow

Week 2: Add hybrid retrieval
  → Qdrant (Docker) + BM25
  → RRF fusion
  → Goal: understand why hybrid beats pure vector

Week 3: Add reranking + quality measurement
  → BGE Reranker or FlashRank
  → RAGAS evaluation
  → Goal: see the numbers improve

Week 4: Add LangGraph
  → Build Corrective RAG
  → Add query routing
  → Goal: understand agentic patterns

Week 5: Add memory + caching
  → Mem0 integration
  → Redis semantic cache
  → Goal: production-level performance

Week 6+: Optimize
  → vLLM for multi-user
  → DSPy for prompt optimization
  → Langfuse for monitoring
```
