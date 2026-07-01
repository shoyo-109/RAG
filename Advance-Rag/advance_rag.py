"""
Advanced RAG Pipeline

Features:
1. Semantic Chunking (Hugging Face embeddings "all-MiniLM-L6-v2", 90 percentile threshold)
2. Embeddings (sentence-transformers/all-MiniLM-L6-v2)
3. LLM (Nvidia Nemotron reasoning model implementation)
4. Hybrid Search (Chroma Vector retriever + BM25, customized ensembler with 50-50 weights)
5. HNSW index configuration in Chroma (M=20, ef=100)
6. Dynamic BM25 rebuilding when new documents are added
7. Hallucination filter in the RAG pipeline using LLM reflection
8. Batch query support
9. Dynamic Semantic Cache (caching frequent queries in real-time)
"""

import os
import uuid
import logging
import asyncio
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv

# LangChain and Community imports
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_experimental.text_splitter import SemanticChunker

from langchain_classic.embeddings.cache import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore

load_dotenv()

# Enable logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("AdvancedRAG")


class RAGCache:
    """
    Dynamic semantic caching system to cache frequent queries in real-time.
    Uses cosine similarity of query embeddings to detect similar queries.
    """
    def __init__(self, embeddings, similarity_threshold: float = 0.95):
        self.embeddings = embeddings
        self.threshold = similarity_threshold
        # Stores cache entry: query_text -> {"embedding": np.ndarray, "response": str}
        self.cache: Dict[str, Dict] = {}

    def get(self, query: str) -> Optional[str]:
        if not self.cache:
            return None
        
        # Check exact match first
        if query in self.cache:
            logger.info("Cache hit: Exact match found.")
            return self.cache[query]["response"]

        # Compute query embedding
        query_emb = np.array(self.embeddings.embed_query(query))
        
        # Check semantic similarity
        best_score = -1.0
        best_response = None
        
        for cached_query, data in self.cache.items():
            cached_emb = data["embedding"]
            # Cosine similarity
            dot_product = np.dot(query_emb, cached_emb)
            norm_q = np.linalg.norm(query_emb)
            norm_c = np.linalg.norm(cached_emb)
            if norm_q > 0 and norm_c > 0:
                similarity = dot_product / (norm_q * norm_c)
                if similarity > best_score:
                    best_score = similarity
                    best_response = data["response"]

        if best_score >= self.threshold:
            logger.info(f"Cache hit: Semantic match found (similarity: {best_score:.4f}).")
            return best_response

        return None

    def set(self, query: str, response: str):
        query_emb = np.array(self.embeddings.embed_query(query))
        self.cache[query] = {
            "embedding": query_emb,
            "response": response
        }
        logger.info(f"Cached response for query: '{query}'")

    def clear(self):
        self.cache.clear()
        logger.info("Cache cleared.")


class AdvancedRAGPipeline:
    def __init__(self, cache_threshold: float = 0.95):
        # 1. Initialize SentenceTransformers Embeddings with Cache-Backed Storage
        self.underlying_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Ensure embedding cache directory exists
        os.makedirs("./.embeddings_cache", exist_ok=True)
        self.store = LocalFileStore(root_path="./.embeddings_cache")
        
        self.embeddings = CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings=self.underlying_embeddings,
            document_embedding_cache=self.store,
            namespace="advance_rag"
        )
        
        # 2. Semantic Chunker with 90 percentile threshold
        self.chunker = SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=90
        )

        # 3. Vector Database (Chroma) initialized with HNSW parameters (m=20, ef=100)
        # We specify collection_metadata to configure the underlying HNSW index in Chroma
        self.vector_store = Chroma(
            collection_name="advanced_rag_collection",
            embedding_function=self.embeddings,
            collection_metadata={
                "hnsw:M": 20,
                "hnsw:construction_ef": 100,
                "hnsw:search_ef": 100
            }
        )

        # 4. In-memory document storage for rebuilding BM25
        self.all_chunks: List[Document] = []
        self.bm25_retriever: Optional[BM25Retriever] = None

        # 5. Initialize Nvidia Nemotron LLM
        self.llm = ChatOpenAI(
            model_name="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.getenv("NVIDIA_API_KEY"),
            temperature=0.5,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": True},
                "reasoning_budget": 16384
            }
        )

        # Prompts for RAG and Hallucination Filter
        self.rag_prompt = ChatPromptTemplate.from_template(
            """Answer the question based on the context provided:
            
            Context:
            {context}
            
            Question:
            {question}

            If you don't have enough context, respond with "I don't have enough context"
            """
        )

        self.hallucination_prompt = ChatPromptTemplate.from_template(
            """You are a hallucination filter. Analyze the context and the answer.
Determine if the answer is completely supported by and grounded in the context without any extra assumptions, external facts, or fabrications.

Context:
{context}

Answer:
{answer}

Respond ONLY with "YES" if the answer is fully supported by the context, or "NO" if it contains hallucinations or unsupported information. Do not write anything else.
"""
        )

        # 6. Dynamic Cache system
        self.cache = RAGCache(embeddings=self.embeddings, similarity_threshold=cache_threshold)

    def add_documents(self, raw_documents: List[Document]):
        """
        Adds documents to the pipeline by performing semantic chunking, 
        indexing into Chroma, and rebuilding/updating the BM25 index.
        """
        if not raw_documents:
            return

        # Perform semantic chunking
        chunks = self.chunker.split_documents(raw_documents)
        logger.info(f"Semantically chunked {len(raw_documents)} documents into {len(chunks)} chunks.")

        # Index in Chroma vector store
        self.vector_store.add_documents(chunks)
        
        # Append to our document pool and rebuild BM25 retriever
        self.all_chunks.extend(chunks)
        self.bm25_retriever = BM25Retriever.from_documents(self.all_chunks, k=5)
        logger.info("BM25 index successfully rebuilt with updated knowledge base.")

    def add_text_document(self, text: str, metadata: Optional[dict] = None):
        doc = Document(page_content=text, metadata=metadata or {})
        self.add_documents([doc])

    def custom_hybrid_search(self, query: str, top_k: int = 5, rrf_k: int = 60) -> List[Document]:
        """
        Custom hybrid search ensembler combining Chroma Vector Search and BM25 search
        using Reciprocal Rank Fusion (RRF) with equal (50-50) weights.
        """
        if not self.all_chunks:
            return []

        # 1. Fetch candidates from Vector search
        vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": top_k})
        vector_docs = vector_retriever.invoke(query)

        # 2. Fetch candidates from BM25 search
        bm25_docs = []
        if self.bm25_retriever:
            bm25_docs = self.bm25_retriever.invoke(query)

        # RRF scoring dict
        rrf_scores = {}
        doc_map = {}

        # Combine results with 50-50 weight (weight = 0.5 for each retriever)
        retrievers_results = [vector_docs, bm25_docs]
        weight = [0.5, 0.5]

        for r_idx, docs in enumerate(retrievers_results):
            w = weight[r_idx]
            for rank_idx, doc in enumerate(docs):
                doc_key = doc.page_content
                doc_map[doc_key] = doc
                # RRF Score formula: weight * (1.0 / (rrf_k + rank))
                score = w * (1.0 / (rrf_k + (rank_idx + 1)))
                rrf_scores[doc_key] = rrf_scores.get(doc_key, 0.0) + score

        # Sort documents based on combined RRF scores
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[doc_key] for doc_key, _ in sorted_docs[:top_k]]

    def hallucination_filter(self, context: str, answer: str) -> bool:
        """
        Evaluates whether the generated response is grounded in the retrieved context.
        Returns True if the response is safe (grounded), and False if it contains hallucinations.
        """
        # If model outputs fallback, it's not a hallucination
        if "I don't have enough context" in answer:
            return True

        filter_chain = self.hallucination_prompt | self.llm | StrOutputParser()
        try:
            verdict = filter_chain.invoke({"context": context, "answer": answer}).strip().upper()
            logger.info(f"Hallucination Filter Verdict: {verdict}")
            return "YES" in verdict
        except Exception as e:
            logger.error(f"Error executing hallucination check: {e}")
            # Fallback to true to avoid blocking in case of API failure
            return True

    def query(self, question: str) -> str:
        """
        Executes a single RAG query with dynamic caching, hybrid search, 
        and hallucination filtering.
        """
        # 1. Dynamic cache lookup
        cached_res = self.cache.get(question)
        if cached_res is not None:
            return cached_res

        # 2. Retrieve relevant context using custom hybrid search
        relevant_docs = self.custom_hybrid_search(question, top_k=5)
        context_str = "\n\n".join([doc.page_content for doc in relevant_docs])

        # 3. Generate response using Nvidia Nemotron
        rag_chain = self.rag_prompt | self.llm | StrOutputParser()
        response = rag_chain.invoke({"context": context_str, "question": question})

        # 4. Apply hallucination filter
        is_grounded = self.hallucination_filter(context_str, response)
        if not is_grounded:
            logger.warning("Hallucination detected in generated response!")
            final_response = f"[WARNING: Response failed hallucination filter] {response}"
        else:
            final_response = response

        # 5. Populate cache
        self.cache.set(question, final_response)
        return final_response

    def get_chunk_projections(self) -> List[Dict]:
        """
        Projects all chunk embeddings to a 3D coordinate space using PCA.
        Returns a list of dicts with chunk texts, IDs, and 3D coordinates.
        """
        if not self.all_chunks:
            return []
            
        texts = [doc.page_content for doc in self.all_chunks]
        # Retrieve embeddings
        raw_embs = self.embeddings.embed_documents(texts)
        X = np.array(raw_embs)
        
        # Simple PCA to 3D
        n_samples = X.shape[0]
        if n_samples < 3:
            # Generate deterministic positions if there are less than 3 chunks
            projections = []
            for i in range(n_samples):
                projections.append({
                    "id": i,
                    "text": texts[i],
                    "x": float((i - (n_samples - 1)/2.0) * 4.0),
                    "y": float(0.0),
                    "z": float(0.0)
                })
            return projections

        # Center data
        X_centered = X - np.mean(X, axis=0)
        # Covariance
        cov = np.cov(X_centered, rowvar=False)
        # Eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        vectors = eigenvectors[:, idx[:3]]
        # Project
        projected = np.dot(X_centered, vectors)
        
        # Normalize to nice range
        max_val = np.max(np.abs(projected))
        if max_val > 0:
            projected = (projected / max_val) * 7.0
            
        results = []
        for i in range(n_samples):
            results.append({
                "id": i,
                "text": texts[i],
                "x": float(projected[i, 0]),
                "y": float(projected[i, 1]),
                "z": float(projected[i, 2])
            })
        return results

    async def astream_query(self, question: str):
        """
        Async generator to stream search stages, answer tokens,
        and hallucination detection warning if applicable.
        """
        # 1. Cache hit check
        cached_res = self.cache.get(question)
        if cached_res is not None:
            # If cached_res has warning, split it or stream it
            yield f"data: {cached_res}\n\n"
            return

        # Stage 1: Consulting KB
        yield "data: stage:🔍 Consulting advanced hybrid knowledge base...\n\n"
        await asyncio.sleep(0.4)
        
        # Retrieve context
        relevant_docs = self.custom_hybrid_search(question, top_k=5)
        context_str = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Send retrieved chunk texts to highlight in the UI
        retrieved_texts = [doc.page_content for doc in relevant_docs]
        yield f"data: retrieved_chunks:{json.dumps(retrieved_texts)}\n\n"

        # Stage 2: Analyzing
        yield "data: stage:🧠 Blending vector and BM25 search indices...\n\n"
        await asyncio.sleep(0.4)

        # Stage 3: Formulating
        yield "data: stage:📝 Formulating response with Nvidia Nemotron...\n\n"
        await asyncio.sleep(0.2)

        # Stream LLM output
        rag_chain = self.rag_prompt | self.llm | StrOutputParser()
        full_response = ""
        
        async for chunk in rag_chain.astream({"context": context_str, "question": question}):
            full_response += chunk
            yield f"data: {chunk}\n\n"

        # Stage 4: Run Hallucination Filter
        is_grounded = self.hallucination_filter(context_str, full_response)
        if not is_grounded:
            logger.warning("Hallucination detected in generated response!")
            yield "data: \n\n⚠️ [WARNING: Response failed hallucination filter check]\n\n"
            
        # Store in cache
        final_cached = full_response if is_grounded else f"[WARNING: Response failed hallucination filter] {full_response}"
        self.cache.set(question, final_cached)

    def batch_query(self, questions: List[str]) -> List[str]:
        """
        Batch queries using LangChain's native execution or mapping.
        """
        # We process queries individually to handle cache lookups, hybrid searches, and hallucination filters correctly.
        return [self.query(q) for q in questions]


# --- Verification & Demo Function ---
def test_advanced_pipeline():
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except:
            pass
    logger.info("Initializing Advanced RAG Pipeline...")
    pipeline = AdvancedRAGPipeline(cache_threshold=0.92)

    # 1. Load a custom test document
    test_document_text = """
    Antigravity AI is an advanced agentic software coding assistant developed by Google DeepMind.
    It specializes in complex codebase maintenance, software engineering, and pair programming.
    The principal design objective of Antigravity is zero-friction code creation and high reliability.
    It was launched in mid-2026. Antigravity does not eat food; it operates entirely on electrical power.
    """
    
    logger.info("Adding test document to database...")
    pipeline.add_text_document(test_document_text, metadata={"source": "test_spec"})

    # Test dynamic addition of a second document to verify BM25 rebuilds
    second_doc = """
    Antigravity AI users should interact with it through approved IDE integrations.
    It supports multiple programming languages including Python, TypeScript, and Go.
    """
    pipeline.add_text_document(second_doc, metadata={"source": "user_guide"})

    # 2. Test Single Query
    q1 = "Who developed Antigravity AI?"
    logger.info(f"Querying Q1: '{q1}'")
    ans1 = pipeline.query(q1)
    print(f"\nQ: {q1}\nA: {ans1}\n")

    # 3. Test Dynamic caching (Semantic match)
    q1_similar = "Who is the developer of Antigravity AI?"
    logger.info(f"Querying Q1 (similar): '{q1_similar}'")
    ans1_cached = pipeline.query(q1_similar)
    print(f"\nQ: {q1_similar}\nA (From Cache): {ans1_cached}\n")

    # 4. Test Batch Queries
    batch_qs = [
        "What programming languages are supported by Antigravity?",
        "When was Antigravity AI launched?",
        "Does Antigravity AI eat food?"
    ]
    logger.info(f"Running Batch queries for: {batch_qs}")
    batch_responses = pipeline.batch_query(batch_qs)
    for q, ans in zip(batch_qs, batch_responses):
        print(f"\nBatch Q: {q}\nBatch A: {ans}\n")

    # 5. Test Hallucination Filter with an out-of-context question that forces hallucination
    # The LLM should respond with "I don't have enough context" or if it hallucinates, the filter will warning-tag it.
    q_unrelated = "What is the favorite food of the CEO of Google?"
    logger.info(f"Querying Q (unrelated): '{q_unrelated}'")
    ans_unrelated = pipeline.query(q_unrelated)
    print(f"\nQ: {q_unrelated}\nA: {ans_unrelated}\n")


if __name__ == "__main__":
    test_advanced_pipeline()