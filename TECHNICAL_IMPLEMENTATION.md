# Technical Implementation: Advanced RAG Pipeline

This document details the engineering specifications of the production-ready RAG pipeline implemented in [advance_rag.py](file:///d:/Fine%20Tunning/ADVANCE%20RAG/RAG/Advance-Rag/advance_rag.py).

---

## 🔍 Core Pipeline Specifications

### 1. Document Loading
- **Engine**: PyMuPDF (`pymupdf` parser).
- **Rationale**: Highly optimized C-binding text extractor that resolves spacing errors (e.g. avoiding concatenated tokens like `Thedocumentdescribes`) common in pure-python alternatives.

### 2. Semantic Chunking
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` with local cache-backed storage (`LocalFileStore` pointing to `./.embeddings_cache`).
- **Chunking Method**: `SemanticChunker` with a **90th percentile** breakpoint threshold.

### 3. Vector Database (Chroma HNSW)
- **HNSW Parameters**:
  - `hnsw:M = 20` (Max number of outgoing links per node in graph).
  - `hnsw:construction_ef = 100` (Size of the dynamic candidate list for index building).
  - `hnsw:search_ef = 100` (Size of the dynamic candidate list for query search).

### 4. Custom Hybrid Search
- **Retrievers**: Combines Chroma Vector Search ($k=5$) and a dynamic `BM25Retriever` ($k=5$).
- **Fusion Algorithm**: Reciprocal Rank Fusion (RRF) with equal weight ($0.5$ Vector, $0.5$ Lexical):
  $$\text{RRF Score}(d \in D) = \sum_{m \in M} w_m \cdot \frac{1}{\text{rrf\_k} + r_m(d)}$$
  Where $M = \{\text{Vector}, \text{BM25}\}$, $w_m = 0.5$, $\text{rrf\_k} = 60$, and $r_m(d)$ is the rank of document $d$ in retriever $m$.

---

## 🛡️ Resilience & Operational Reliability

### 1. Exponential Backoff Retry
- **Decorator**: `with_retry` / `with_retry_async`
- **Config**: `max_retries = 3`, `base_delay = 1.0s`, `max_delay = 10.0s` with randomized jitter to prevent thundering herd issues on the Nvidia endpoint.

### 2. Circuit Breaker
- **Config**: `failure_threshold = 3`, `recovery_timeout = 30.0s`
- **State transitions**: If primary Nvidia model API triggers 3 consecutive errors, the circuit transitions to `OPEN` state, blocking immediate calls and executing fallback execution instantly.

### 3. Model Fallback Chain
- **Primary LLM**: `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` (hosted on Nvidia Nim endpoints).
- **Secondary LLM**: `gpt-4o-mini` (OpenAI endpoint). Used dynamically when primary model times out or throws exceptions.

---

## 📊 Logging & Instrumentation

- **Log Aggregator Format**: Custom `JSONFormatter` writes structured logs containing metadata timestamps, levels, execution steps, latency, and token estimates.
- **Metrics Dashboard tracking**: Exposes request rates, total errors, latency averages, and cache hits.

---

## 🌐 3D Embedding Space Projection
- **Dimensionality Reduction**: Principal Component Analysis (PCA) computed using Numpy linear algebra (`np.linalg.eigh` on centered covariance matrix) to project 384-dimensional sentence vectors down to 3D coordinate space `(x, y, z)` for visual rendering.
