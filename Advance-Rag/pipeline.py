import os
import time
import logging
import asyncio
import json
import threading
import re
import numpy as np
from typing import List, Dict, Tuple, Optional, Any

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
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
    from .response_tuner import get_tuned_prompt, post_process_response
    from .memory import MultiLayerMemoryManager
    from .presentation import PresentationBuilder, PresentationTuner
except (ImportError, ValueError):
    from metrics import MetricsCollector
    from retry import with_retry
    from circuit_breaker import CircuitBreaker
    from sanitizer import InputSanitizer
    from pii_detector import PIIDetector
    from output_validator import OutputValidator
    from cache import RAGCache
    from response_tuner import get_tuned_prompt, post_process_response
    from memory import MultiLayerMemoryManager
    from presentation import PresentationBuilder, PresentationTuner


logger = logging.getLogger("AdvancedRAG")

_shared_embeddings = None
_embeddings_lock = threading.Lock()

class LowContextError(Exception):
    """Raised when retrieved documents have low similarity/relevance to the query."""
    pass

class AdvancedRAGPipeline:
    def __init__(self, cache_threshold: float = 0.95):
        # 1. Initialize SentenceTransformers Embeddings with Cache-Backed Storage
        global _shared_embeddings
        with _embeddings_lock:
            if _shared_embeddings is None:
                logger.info("Initializing shared HuggingFaceEmbeddings model...")
                _shared_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.underlying_embeddings = _shared_embeddings
        
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

        # 3. Vector Database (Qdrant Cloud) initialized with Cosine distance metric
        qdrant_url = os.getenv("QDRANT_ENDPOINT") or os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API") or os.getenv("QDRANT_API_KEY")

        if not qdrant_url or not qdrant_api_key:
            logger.warning("QDRANT_ENDPOINT/QDRANT_URL or QDRANT_API/QDRANT_API_KEY missing in environment variables!")

        self.qdrant_client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
        )

        collection_name = "advanced_rag_collection"
        if qdrant_url and not self.qdrant_client.collection_exists(collection_name):
            logger.info(f"Creating Qdrant collection '{collection_name}' with 384 dimensions...")
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )

        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=collection_name,
            embedding=self.embeddings,
        )

        # 4. In-memory document storage for rebuilding BM25
        self.all_chunks: List[Document] = []
        self.bm25_retriever: Optional[BM25Retriever] = None

        # 5. Initialize Primary Nvidia Instruct LLM (Sub-1.5s Fast Latency)
        self.primary_llm = ChatOpenAI(
            model_name="meta/llama-3.1-70b-instruct",
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.getenv("NVIDIA_API_KEY"),
            temperature=0.3,
            timeout=15.0
        )

        # 6. Initialize Cognitive Nvidia Nemotron Reasoning LLM (Complex Cognitive Reasoning)
        self.cognitive_llm = ChatOpenAI(
            model_name="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.getenv("NVIDIA_API_KEY"),
            temperature=0.3,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": True},
                "reasoning_budget": 2048
            }
        )


        # Fallback LLM uses primary Nvidia model (completely removing OpenAI dependency)
        self.fallback_llm = self.primary_llm

        # Prompts for RAG and Hallucination Filter
        self.rag_prompt = get_tuned_prompt()

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

        # 9. Multi-Layer Memory Engine & Presentation Engine
        self.memory_manager = MultiLayerMemoryManager(capacity=5, max_context_tokens=16000, vector_store=self.vector_store)


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

        # Index in Qdrant vector store
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

    def custom_hybrid_search(self, query: str, query_embedding: np.ndarray, top_k: int = 10, rrf_k: int = 60) -> Tuple[List[Document], float, float]:
        """
        Custom hybrid search ensembler combining Qdrant Vector Search and BM25 search
        using Reciprocal Rank Fusion (RRF) with equal (50-50) weights.
        
        Optimization: uses precomputed query embedding to avoid double embedding calculations.
        Returns: (List[Document], max_rrf_score, top_vector_score)
        """
        if not self.all_chunks:
            return [], 0.0, 0.0

        # 1. Fetch candidates from Vector search by precomputed embedding (saves 50-150ms)
        vector_results = self.vector_store.similarity_search_with_score_by_vector(query_embedding.tolist(), k=top_k)
        vector_docs = [doc for doc, _ in vector_results]
        top_vector_score = vector_results[0][1] if vector_results else 0.0

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
        max_score = sorted_docs[0][1] if sorted_docs else 0.0
        return [doc_map[doc_key] for doc_key, _ in sorted_docs[:top_k]], max_score, top_vector_score

    async def custom_hybrid_search_async(self, query: str, query_embedding: np.ndarray, top_k: int = 10, rrf_k: int = 60) -> Tuple[List[Document], float, float]:
        """
        Asynchronous custom hybrid search. Runs Vector and BM25 searches in parallel threads.
        
        Optimization: concurrent retrieval + precomputed query embedding.
        Returns: (List[Document], max_rrf_score, top_vector_score)
        """
        if not self.all_chunks:
            return [], 0.0, 0.0

        # Run vector search and BM25 search concurrently in threadpool to prevent blocking the event loop
        tasks = [
            asyncio.to_thread(self.vector_store.similarity_search_with_score_by_vector, query_embedding.tolist(), k=top_k)
        ]
        if self.bm25_retriever:
            tasks.append(asyncio.to_thread(self.bm25_retriever.invoke, query))
        else:
            tasks.append(asyncio.to_thread(lambda: []))
            
        vector_results, bm25_docs = await asyncio.gather(*tasks)
        vector_docs = [doc for doc, _ in vector_results]
        top_vector_score = vector_results[0][1] if vector_results else 0.0

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
        max_score = sorted_docs[0][1] if sorted_docs else 0.0
        return [doc_map[doc_key] for doc_key, _ in sorted_docs[:top_k]], max_score, top_vector_score

    def rewrite_query(self, question: str) -> str:
        """Self-healing: Rewrites a poor query to improve retrieval scores."""
        try:
            rewrite_prompt = ChatPromptTemplate.from_template(
                "You are an AI assistant tasked with reformulating user queries for better information retrieval. "
                "Rewrite the following query to be more descriptive and search-engine friendly. "
                "Do not answer the query, just output the rewritten query.\n\nQuery: {question}"
            )
            chain = rewrite_prompt | self.fallback_llm | StrOutputParser()
            rewritten = chain.invoke({"question": question}).strip()
            logger.info(f"Rewrote query from '{question}' to '{rewritten}'")
            return rewritten
        except Exception as e:
            logger.error(f"Error rewriting query: {e}")
            return question

    def hallucination_filter(self, context: str, answer: str) -> bool:
        """
        Evaluates whether the generated response is grounded in the retrieved context.
        """
        if "[CONTENT BLOCKED]" in answer:
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

    def _route_query(self, question: str) -> Tuple[ChatOpenAI, str]:
        """
        Regex Intent Classifier to route between fast instruct model (sub-2s latency)
        and cognitive reasoning model (Nemotron).
        """
        cognitive_patterns = [
            r'\bcompare\b', r'\bcontrast\b', r'\brelationship\b', r'\bsynthesize\b',
            r'\banalyze\b', r'\bexplain why\b', r'\bhow does .* relate\b',
            r'\bsummarize all\b', r'\bconnect\b', r'\bevaluate\b'
        ]
        
        q_lower = question.lower()
        for pattern in cognitive_patterns:
            if re.search(pattern, q_lower):
                logger.info(f"Intent Classifier: Routing to Cognitive Reasoning Model (Nemotron) based on pattern '{pattern}'")
                return self.cognitive_llm, "cognitive_nemotron"
                
        logger.info("Intent Classifier: Routing to Fast Primary Model (Llama-3.1-70b)")
        return self.primary_llm, "primary_llama70b"

    def _build_hierarchical_context(self, docs: List[Document]) -> str:
        """
        Extracts parent_text and breadcrumb_path metadata from retrieved child documents,
        deduplicates section blocks, and constructs rich hierarchical context for the LLM.
        """
        seen_parents = set()
        formatted_blocks = []

        for doc in docs:
            parent_text = doc.metadata.get("parent_text", doc.page_content) if hasattr(doc, "metadata") and isinstance(doc.metadata, dict) else doc.page_content
            breadcrumb = doc.metadata.get("breadcrumb_path", "") if hasattr(doc, "metadata") and isinstance(doc.metadata, dict) else ""

            if parent_text in seen_parents:
                continue
            seen_parents.add(parent_text)

            if breadcrumb:
                block = f"[Category: {breadcrumb}]\n{parent_text}"
            else:
                block = parent_text
            formatted_blocks.append(block)

        return "\n\n---\n\n".join(formatted_blocks)

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

            # 3. Hybrid search and Self-Healing Thresholding
            relevant_docs, max_score, top_vector_score = self.custom_hybrid_search(sanitized_question, query_emb, top_k=self.top_k)
            
            # Threshold Check: either vector similarity is too low or RRF score is too low
            MIN_SIMILARITY_THRESHOLD = 0.25
            MIN_SCORE_THRESHOLD = 0.008
            
            if top_vector_score < MIN_SIMILARITY_THRESHOLD or max_score < MIN_SCORE_THRESHOLD:
                logger.warning(f"Low retrieval score (sim={top_vector_score:.4f}, rrf={max_score:.4f}), attempting query rewrite.")
                sanitized_question = self.rewrite_query(sanitized_question)
                query_emb = np.array(self.embeddings.embed_query(sanitized_question))
                relevant_docs, max_score, top_vector_score = self.custom_hybrid_search(sanitized_question, query_emb, top_k=self.top_k)
                
                if top_vector_score < MIN_SIMILARITY_THRESHOLD or max_score < MIN_SCORE_THRESHOLD:
                    logger.warning(f"Score still too low after rewrite (sim={top_vector_score:.4f}, rrf={max_score:.4f}). Short-circuiting.")
                    raise LowContextError("Insufficient context found in the knowledge base.")

            rag_context_str = self._build_hierarchical_context(relevant_docs)
            allocated_context = self.memory_manager.prepare_allocated_context(rag_context_str, sanitized_question)
            
            context_with_memory = allocated_context["rag_context"]
            if allocated_context["memory_context"]:
                context_with_memory = f"Memory History:\n{allocated_context['memory_context']}\n\nDocument Context:\n{allocated_context['rag_context']}"

            # 4. Generate response with routed model
            selected_llm, model_used = self._route_query(sanitized_question)
            chain = self.rag_prompt | selected_llm | StrOutputParser()
            response = chain.invoke({"context": context_with_memory, "question": sanitized_question})
            output_tokens = len(response.split()) * 4 // 3

            # 5. Active Self-Healing Hallucination Filter
            is_grounded = self.hallucination_filter(allocated_context["rag_context"], response)
            if not is_grounded:
                logger.warning("Hallucination detected! Executing Self-Healing strict grounding correction...")
                try:
                    correction_prompt = ChatPromptTemplate.from_template(
                        "You are an active self-healing grounding verifier. Your task is to rewrite the draft response so it is 100% strictly grounded in the provided context.\n\n"
                        "UNIVERSAL GROUNDING RULES:\n"
                        "1. Strictly limit facts, metrics, entities, and details to the exact subject or condition requested in the user's question.\n"
                        "2. Remove any claim, entity, or detail that is not explicitly present in the retrieved context.\n"
                        "3. Do NOT extrapolate or infer facts outside the context.\n\n"
                        "Retrieved Context:\n{context}\n\n"
                        "User Question:\n{question}\n\n"
                        "Draft Response to Cleanse:\n{response}\n\n"
                        "Strictly Grounded Corrected Answer:"
                    )

                    correction_chain = correction_prompt | self.fallback_llm | StrOutputParser()
                    final_response = correction_chain.invoke({
                        "context": allocated_context["rag_context"],
                        "question": sanitized_question,
                        "response": response
                    })
                except Exception as ce:
                    logger.error(f"Failed to execute self-correction: {ce}")
                    final_response = response
            else:
                final_response = response


            # 6. Output Validation
            is_valid, final_response, val_reason = self.output_validator.validate(final_response)
            if not is_valid:
                logger.warning(f"Output altered by validator: {val_reason}")

            # Apply Presentation Engine transformation (Rule 1-9 CPU Native < 0.5ms)
            final_response = PresentationBuilder.transform_to_presentation(
                final_response,
                active_topic=allocated_context["active_topic"],
                should_render_header=allocated_context["should_render_top_header"]
            )

            # Record completed turn in Memory Manager
            self.memory_manager.record_completed_turn(sanitized_question, final_response)

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

        except LowContextError as lce:
            self.metrics.record_request(
                latency_ms=(time.time() - start_time) * 1000,
                input_tokens=input_tokens,
                output_tokens=0,
                error=True,
                cache_hit=False
            )
            fallback_response = {
                "error": "LowContextError",
                "message": "I don't have enough context to answer accurately.",
                "actions": [
                    {"label": "Search the web instead?", "action": "web_search"},
                    {"label": "Escalate to human support?", "action": "escalate"},
                    {"label": "Did you mean to ask a different question?", "action": "suggest"}
                ]
            }
            return json.dumps(fallback_response)
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
        relevant_docs, max_score, top_vector_score = await self.custom_hybrid_search_async(sanitized_question, query_emb, top_k=self.top_k)
        
        MIN_SIMILARITY_THRESHOLD = 0.25
        MIN_SCORE_THRESHOLD = 0.008
        if top_vector_score < MIN_SIMILARITY_THRESHOLD or max_score < MIN_SCORE_THRESHOLD:
            yield "data: stage:⚠️ Low confidence retrieval, attempting to rewrite query...\n\n"
            sanitized_question = self.rewrite_query(sanitized_question)
            query_emb = np.array(self.embeddings.embed_query(sanitized_question))
            relevant_docs, max_score, top_vector_score = await self.custom_hybrid_search_async(sanitized_question, query_emb, top_k=self.top_k)
            
            if top_vector_score < MIN_SIMILARITY_THRESHOLD or max_score < MIN_SCORE_THRESHOLD:
                fallback_response = {
                    "error": "LowContextError",
                    "message": "I don't have enough context to answer accurately.",
                    "actions": [
                        {"label": "Search the web instead?", "action": "web_search"},
                        {"label": "Escalate to human support?", "action": "escalate"},
                        {"label": "Did you mean to ask a different question?", "action": "suggest"}
                    ]
                }
                yield f"data: {json.dumps(fallback_response)}\n\n"
                return

        context_str = self._build_hierarchical_context(relevant_docs)
        allocated_context = self.memory_manager.prepare_allocated_context(context_str, sanitized_question)
        
        context_with_memory = allocated_context["rag_context"]
        if allocated_context["memory_context"]:
            context_with_memory = f"Memory History:\n{allocated_context['memory_context']}\n\nDocument Context:\n{allocated_context['rag_context']}"

        retrieved_texts = [doc.page_content for doc in relevant_docs]
        yield f"data: retrieved_chunks:{json.dumps(retrieved_texts)}\n\n"

        yield "data: stage:🧠 Blending vector and BM25 search indices...\n\n"
        await asyncio.sleep(0.2)

        selected_llm, model_tag = self._route_query(sanitized_question)

        if model_tag == "cognitive_nemotron":
            yield "data: stage:🤔 Nemotron Cognitive Reasoning Stage...\n\n"
        else:
            yield "data: stage:📝 Formulating response...\n\n"
        
        # Execute streaming with routed model via LCEL chain
        stream_successful = False
        try:
            self.circuit_breaker.check_call_allowed()
            chain = self.rag_prompt | selected_llm | StrOutputParser()
            async for token_text in chain.astream({"context": context_with_memory, "question": sanitized_question}):
                full_response += token_text
                yield f"data: {token_text}\n\n"
            self.circuit_breaker.record_success()
            stream_successful = True
        except Exception as e:
            self.circuit_breaker.record_failure()
            logger.error(f"Routed streaming ({model_tag}) failed: {e}. Falling back to primary LLM...")
            
        if not stream_successful:
            try:
                # Stream from fallback
                fallback_chain = self.rag_prompt | self.fallback_llm | StrOutputParser()
                async for token_text in fallback_chain.astream({"context": context_with_memory, "question": sanitized_question}):
                    full_response += token_text
                    yield f"data: {token_text}\n\n"
            except Exception as fe:
                logger.critical(f"Fallback streaming failed: {fe}")
                yield "data: Error: An internal error occurred while generating response.\n\n"
                return



        # Stage 4: Active Self-Healing Hallucination Filter
        is_grounded = self.hallucination_filter(allocated_context["rag_context"], full_response)
        if not is_grounded:
            logger.warning("Hallucination detected! Executing Self-Healing strict grounding correction...")
            try:
                correction_prompt = ChatPromptTemplate.from_template(
                    "You are an active self-healing grounding verifier. Your task is to rewrite the draft response so it is 100% strictly grounded in the provided context.\n\n"
                    "UNIVERSAL GROUNDING RULES:\n"
                    "1. Strictly limit facts, metrics, entities, and details to the exact subject or condition requested in the user's question.\n"
                    "2. Remove any claim, entity, or detail that is not explicitly present in the retrieved context.\n"
                    "3. Do NOT extrapolate or infer facts outside the context.\n\n"
                    "Retrieved Context:\n{context}\n\n"
                    "User Question:\n{question}\n\n"
                    "Draft Response to Cleanse:\n{response}\n\n"
                    "Strictly Grounded Corrected Answer:"
                )

                correction_chain = correction_prompt | self.fallback_llm | StrOutputParser()
                full_response = correction_chain.invoke({
                    "context": allocated_context["rag_context"],
                    "question": sanitized_question,
                    "response": full_response
                })
            except Exception as ce:
                logger.error(f"Failed to execute hallucination self-correction: {ce}")


        # Validate final output
        is_valid, clean_response, val_reason = self.output_validator.validate(full_response)
        if not is_valid:
            yield f"data: \n\n[Security modification: {val_reason}]\n\n"
            full_response = clean_response

        # Transform response via Presentation Engine (CPU Native < 0.5ms)
        clean_response_tuned = PresentationBuilder.transform_to_presentation(
            full_response,
            active_topic=allocated_context["active_topic"],
            should_render_header=allocated_context["should_render_top_header"]
        )

        # Yield presentation-transformed response safely JSON-encoded for SSE stream
        yield f"data: final_transformed:{json.dumps(clean_response_tuned)}\n\n"


        # Record completed turn in memory
        self.memory_manager.record_completed_turn(sanitized_question, clean_response_tuned)


        # Store in cache
        self.cache.set(sanitized_question, clean_response_tuned, query_emb=query_emb)
        
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
