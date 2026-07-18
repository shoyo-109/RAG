import os
import time
import logging
import asyncio
import json
import numpy as np
from typing import List, Dict, Tuple, Optional, Any

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_experimental.text_splitter import SemanticChunker
from langchain_classic.embeddings.cache import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore

# Local imports
try:
    from .metrics import MetricsCollector
    from .retry import with_retry
    from .circuit_breaker import CircuitBreaker
    from .sanitizer import InputSanitizer
    from .pii_detector import PIIDetector
    from .output_validator import OutputValidator
    from .cache import RAGCache
except (ImportError, ValueError):
    from metrics import MetricsCollector
    from retry import with_retry
    from circuit_breaker import CircuitBreaker
    from sanitizer import InputSanitizer
    from pii_detector import PIIDetector
    from output_validator import OutputValidator
    from cache import RAGCache


logger = logging.getLogger("AdvancedRAG")

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

        # 5. Initialize Primary Nvidia LLM
        self.primary_llm = ChatOpenAI(
            model_name="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.getenv("NVIDIA_API_KEY"),
            temperature=0.5,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": True},
                "reasoning_budget": 16384
            }
        )

        # 6. Initialize Fallback LLM (GPT-4o-Mini)
        self.fallback_llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.5,
            timeout=15.0
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

        # 7. Dynamic Cache system
        self.cache = RAGCache(embeddings=self.embeddings, similarity_threshold=cache_threshold)

        # 8. Operational Reliability & Security Layers
        self.input_sanitizer = InputSanitizer()
        self.pii_detector = PIIDetector()
        self.output_validator = OutputValidator()
        self.metrics = MetricsCollector()
        self.circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30.0)
        self.top_k = 10
        
        # In-memory cache for PCA 3D projections (Optimization: 0ms projection calculations)
        self._cached_projections: Optional[List[Dict]] = None

    # Invoke LLM chain with Fallback & Circuit Breaker & Retry
    @with_retry(max_retries=3, base_delay=1.0, exceptions=(Exception,))
    def _execute_with_primary(self, chain_input: dict) -> str:
        self.circuit_breaker.check_call_allowed()
        try:
            chain = self.rag_prompt | self.primary_llm | StrOutputParser()
            res = chain.invoke(chain_input)
            self.circuit_breaker.record_success()
            return res
        except Exception as e:
            self.circuit_breaker.record_failure()
            raise e

    def run_llm_chain(self, chain_input: dict) -> Tuple[str, str]:
        """Runs the LLM chain, falling back to ChatOpenAI if the primary Nvidia LLM fails."""
        try:
            logger.info("Attempting primary LLM execution...")
            response = self._execute_with_primary(chain_input)
            return response, "primary_nvidia"
        except Exception as e:
            logger.error(f"Primary LLM failed: {e}. Attempting fallback model...")
            try:
                chain = self.rag_prompt | self.fallback_llm | StrOutputParser()
                response = chain.invoke(chain_input)
                return response, "fallback_openai"
            except Exception as fe:
                logger.critical(f"All LLMs failed in chain run: {fe}")
                raise fe

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
        
        # Invalidate PCA projections cache (Optimization)
        self._cached_projections = None
        
        logger.info("BM25 index successfully rebuilt with updated knowledge base.")

    def add_text_document(self, text: str, metadata: Optional[dict] = None):
        doc = Document(page_content=text, metadata=metadata or {})
        self.add_documents([doc])

    def custom_hybrid_search(self, query: str, query_embedding: np.ndarray, top_k: int = 10, rrf_k: int = 60) -> List[Document]:
        """
        Custom hybrid search ensembler combining Chroma Vector Search and BM25 search
        using Reciprocal Rank Fusion (RRF) with equal (50-50) weights.
        
        Optimization: uses precomputed query embedding to avoid double embedding calculations.
        """
        if not self.all_chunks:
            return []

        # 1. Fetch candidates from Vector search by precomputed embedding (saves 50-150ms)
        vector_docs = self.vector_store.similarity_search_by_vector(query_embedding.tolist(), k=top_k)

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

    async def custom_hybrid_search_async(self, query: str, query_embedding: np.ndarray, top_k: int = 10, rrf_k: int = 60) -> List[Document]:
        """
        Asynchronous custom hybrid search. Runs Vector and BM25 searches in parallel threads.
        
        Optimization: concurrent retrieval + precomputed query embedding.
        """
        if not self.all_chunks:
            return []

        # Run vector search and BM25 search concurrently in threadpool to prevent blocking the event loop
        tasks = [
            asyncio.to_thread(self.vector_store.similarity_search_by_vector, query_embedding.tolist(), k=top_k)
        ]
        if self.bm25_retriever:
            tasks.append(asyncio.to_thread(self.bm25_retriever.invoke, query))
        else:
            tasks.append(asyncio.to_thread(lambda: []))
            
        vector_docs, bm25_docs = await asyncio.gather(*tasks)

        # RRF scoring dict
        rrf_scores = {}
        doc_map = {}

        # Combine results with 50-50 weight
        retrievers_results = [vector_docs, bm25_docs]
        weight = [0.5, 0.5]

        for r_idx, docs in enumerate(retrievers_results):
            w = weight[r_idx]
            for rank_idx, doc in enumerate(docs):
                doc_key = doc.page_content
                doc_map[doc_key] = doc
                score = w * (1.0 / (rrf_k + (rank_idx + 1)))
                rrf_scores[doc_key] = rrf_scores.get(doc_key, 0.0) + score

        # Sort documents based on combined RRF scores
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[doc_key] for doc_key, _ in sorted_docs[:top_k]]

    def hallucination_filter(self, context: str, answer: str) -> bool:
        """
        Evaluates whether the generated response is grounded in the retrieved context.
        """
        if "I don't have enough context" in answer or "[CONTENT BLOCKED]" in answer:
            return True

        # Fallback evaluation on Hallucination Filter as well
        try:
            filter_chain = self.hallucination_prompt | self.fallback_llm | StrOutputParser()
            verdict = filter_chain.invoke({"context": context, "answer": answer}).strip().upper()
            logger.info(f"Hallucination Filter Verdict: {verdict}")
            return "YES" in verdict
        except Exception as e:
            logger.error(f"Error executing hallucination check: {e}")
            return True

    def query(self, question: str) -> str:
        """
        Executes a single RAG query with validation, cache lookup, hybrid search,
        llm fallback chain, metrics tracking, and output validation.
        """
        start_time = time.time()
        input_tokens = len(question.split()) * 4 // 3
        output_tokens = 0
        error_occurred = False
        cache_hit = False

        try:
            # 1. Security Check & Sanitization
            is_suspicious, reason = self.input_sanitizer.is_suspicious(question)
            if is_suspicious:
                self.metrics.record_request(0.0, input_tokens, 0, error=True, cache_hit=False, security_block=True)
                logger.warning("Query blocked by security sanitizer", extra={"extra_data": {"reason": reason}})
                return f"[BLOCKED: {reason}]"

            sanitized_question = self.input_sanitizer.sanitize(question)
            
            # Mask PII in input
            sanitized_question = self.pii_detector.mask(sanitized_question)

            # Optimization: Compute query embedding exactly ONCE to avoid duplicate embedding calculation
            query_emb = np.array(self.embeddings.embed_query(sanitized_question))

            # 2. Dynamic Cache lookup (using vectorized similarity and precomputed query embedding)
            cached_res = self.cache.get(sanitized_question, query_emb=query_emb)
            if cached_res is not None:
                cache_hit = True
                output_tokens = len(cached_res.split()) * 4 // 3
                self.metrics.record_request(
                    latency_ms=(time.time() - start_time) * 1000,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    error=False,
                    cache_hit=True
                )
                return cached_res

            # 3. Hybrid search (using precomputed embedding to avoid redundant calls)
            relevant_docs = self.custom_hybrid_search(sanitized_question, query_emb, top_k=self.top_k)
            context_str = "\n\n".join([doc.page_content for doc in relevant_docs])

            # 4. Generate response with Fallback LLM Chain
            response, model_used = self.run_llm_chain({"context": context_str, "question": sanitized_question})
            output_tokens = len(response.split()) * 4 // 3

            # 5. Hallucination filter
            is_grounded = self.hallucination_filter(context_str, response)
            if not is_grounded:
                logger.warning("Hallucination detected in generated response!")
                final_response = f"[WARNING: Response failed hallucination filter] {response}"
            else:
                final_response = response

            # 6. Output Validation
            is_valid, final_response, val_reason = self.output_validator.validate(final_response)
            if not is_valid:
                logger.warning(f"Output altered by validator: {val_reason}")

            # 7. Populate cache (saving query embedding for reuse)
            self.cache.set(sanitized_question, final_response, query_emb=query_emb)
            
            self.metrics.record_request(
                latency_ms=(time.time() - start_time) * 1000,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                error=False,
                cache_hit=False
            )
            
            logger.info("RAG query complete", extra={"extra_data": {"model_used": model_used, "cache_hit": False}})
            return final_response

        except Exception as e:
            logger.error(f"Error handling query: {e}")
            self.metrics.record_request(
                latency_ms=(time.time() - start_time) * 1000,
                input_tokens=input_tokens,
                output_tokens=0,
                error=True,
                cache_hit=False
            )
            return "An internal system error occurred. Please try again."

    async def astream_query(self, question: str):
        """
        Async generator to stream search stages, answer tokens,
        and security/hallucination checks.
        """
        start_time = time.time()
        input_tokens = len(question.split()) * 4 // 3
        full_response = ""
        
        # 1. Security check
        is_suspicious, reason = self.input_sanitizer.is_suspicious(question)
        if is_suspicious:
            self.metrics.record_request(0.0, input_tokens, 0, error=True, cache_hit=False, security_block=True)
            yield f"data: [BLOCKED: {reason}]\n\n"
            return

        sanitized_question = self.input_sanitizer.sanitize(question)
        sanitized_question = self.pii_detector.mask(sanitized_question)

        # Optimization: Compute query embedding exactly ONCE to avoid duplicate embedding calculation
        query_emb = np.array(self.embeddings.embed_query(sanitized_question))

        # 2. Dynamic cache lookup (using vectorized similarity and precomputed query embedding)
        cached_res = self.cache.get(sanitized_question, query_emb=query_emb)
        if cached_res is not None:
            self.metrics.record_request(0.0, input_tokens, len(cached_res.split()) * 4 // 3, error=False, cache_hit=True)
            yield f"data: {cached_res}\n\n"
            return

        yield "data: stage:🔍 Consulting advanced hybrid knowledge base...\n\n"
        await asyncio.sleep(0.2)
        
        # Retrieve context (parallel optimized)
        relevant_docs = await self.custom_hybrid_search_async(sanitized_question, query_emb, top_k=self.top_k)
        context_str = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        retrieved_texts = [doc.page_content for doc in relevant_docs]
        yield f"data: retrieved_chunks:{json.dumps(retrieved_texts)}\n\n"

        yield "data: stage:🧠 Blending vector and BM25 search indices...\n\n"
        await asyncio.sleep(0.2)

        yield "data: stage:📝 Formulating response...\n\n"
        
        # Execute streaming with fallback
        stream_successful = False
        try:
            self.circuit_breaker.check_call_allowed()
            # Stream from primary Nvidia Nemotron
            async for chunk in self.primary_llm.astream(
                self.rag_prompt.format(context=context_str, question=sanitized_question)
            ):
                token_text = chunk.content
                full_response += token_text
                yield f"data: {token_text}\n\n"
            self.circuit_breaker.record_success()
            stream_successful = True
        except Exception as e:
            self.circuit_breaker.record_failure()
            logger.error(f"Primary streaming failed: {e}. Falling back to GPT-4o-Mini...")
            
        if not stream_successful:
            try:
                # Stream from fallback
                async for chunk in self.fallback_llm.astream(
                    self.rag_prompt.format(context=context_str, question=sanitized_question)
                ):
                    token_text = chunk.content
                    full_response += token_text
                    yield f"data: {token_text}\n\n"
            except Exception as fe:
                logger.critical(f"Fallback streaming failed: {fe}")
                yield "data: Error: An internal error occurred while generating response.\n\n"
                return

        # Stage 4: Run Hallucination Filter
        is_grounded = self.hallucination_filter(context_str, full_response)
        if not is_grounded:
            logger.warning("Hallucination detected in generated response!")
            yield "data: \n\n⚠️ [WARNING: Response failed hallucination filter check]\n\n"
            full_response += "\n\n⚠️ [WARNING: Response failed hallucination filter check]"

        # Validate final output
        is_valid, clean_response, val_reason = self.output_validator.validate(full_response)
        if not is_valid:
            yield f"data: \n\n[Security modification: {val_reason}]\n\n"
            full_response = clean_response

        # Store in cache
        self.cache.set(sanitized_question, full_response, query_emb=query_emb)
        
        self.metrics.record_request(
            latency_ms=(time.time() - start_time) * 1000,
            input_tokens=input_tokens,
            output_tokens=len(full_response.split()) * 4 // 3,
            error=False,
            cache_hit=False
        )

    def batch_query(self, questions: List[str]) -> List[str]:
        return [self.query(q) for q in questions]

    def get_chunk_projections(self) -> List[Dict]:
        if not self.all_chunks:
            return []
            
        # Optimization: Return cached projections immediately if they exist (0ms overhead)
        if self._cached_projections is not None:
            logger.info("Returning cached PCA projections.")
            return self._cached_projections
            
        texts = [doc.page_content for doc in self.all_chunks]
        raw_embs = self.embeddings.embed_documents(texts)
        X = np.array(raw_embs)
        
        n_samples = X.shape[0]
        if n_samples < 3:
            projections = []
            for i in range(n_samples):
                projections.append({
                    "id": i,
                    "text": texts[i],
                    "x": float((i - (n_samples - 1)/2.0) * 4.0),
                    "y": float(0.0),
                    "z": float(0.0)
                })
            self._cached_projections = projections
            return projections

        X_centered = X - np.mean(X, axis=0)
        cov = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        vectors = eigenvectors[:, idx[:3]]
        projected = np.dot(X_centered, vectors)
        
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
            
        self._cached_projections = results
        return results
