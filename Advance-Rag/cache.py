import os
import json
import logging
import numpy as np
from typing import Dict, Optional

logger = logging.getLogger("AdvancedRAG")

try:
    import redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False


class RAGCache:
    """
    Dynamic semantic caching system to cache frequent queries in real-time.
    Uses vectorized cosine similarity of query embeddings and integrates with
    Redis (Upstash) for persistent cloud caching across server restarts.
    """
    def __init__(self, embeddings, similarity_threshold: float = 0.95):
        self.embeddings = embeddings
        self.threshold = similarity_threshold
        # Stores local cache entry: query_text -> {"embedding": np.ndarray, "response": str}
        self.cache: Dict[str, Dict] = {}

        # Redis connection setup
        self.redis_client = None
        redis_url = os.getenv("REDIS_URL") or os.getenv("UPSTASH_REDIS_URL")
        
        if HAS_REDIS and redis_url:
            try:
                self.redis_client = redis.from_url(redis_url, decode_responses=True, socket_timeout=3.0)
                self.redis_client.ping()
                logger.info("Successfully connected to Upstash Redis cache!")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis ({e}). Falling back to in-memory cache.")
                self.redis_client = None
        else:
            if not HAS_REDIS:
                logger.info("Redis package not installed. Running on in-memory RAG cache.")
            else:
                logger.info("REDIS_URL not set in environment. Running on in-memory RAG cache.")

    def get(self, query: str, query_emb: Optional[np.ndarray] = None) -> Optional[str]:
        # 1. Check Redis exact match first (0ms cloud lookup)
        if self.redis_client:
            try:
                redis_val = self.redis_client.get(f"rag:cache:{query}")
                if redis_val:
                    data = json.loads(redis_val)
                    logger.info("Cache hit: Exact match found in Upstash Redis.")
                    return data.get("response")
            except Exception as e:
                logger.error(f"Redis lookup error: {e}")

        # 2. Check local in-memory exact match
        if query in self.cache:
            logger.info("Cache hit: Exact match found in local memory.")
            return self.cache[query]["response"]

        # 3. Vectorized semantic matching across local cache entries
        if not self.cache:
            return None

        if query_emb is None:
            query_emb = np.array(self.embeddings.embed_query(query))
        
        cached_keys = list(self.cache.keys())
        
        # Vectorized similarity computation
        embeddings_matrix = np.stack([self.cache[k]["embedding"] for k in cached_keys])
        dot_products = np.dot(embeddings_matrix, query_emb)
        norms_cached = np.linalg.norm(embeddings_matrix, axis=1)
        norm_query = np.linalg.norm(query_emb)
        
        norms_cached[norms_cached == 0] = 1e-9
        if norm_query <= 0:
            norm_query = 1e-9
            
        similarities = dot_products / (norms_cached * norm_query)
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        
        if best_score >= self.threshold:
            best_query = cached_keys[best_idx]
            logger.info(f"Cache hit: Semantic match found (similarity: {best_score:.4f}).")
            return self.cache[best_query]["response"]

        return None

    def set(self, query: str, response: str, query_emb: Optional[np.ndarray] = None, ttl: int = 86400):
        if query_emb is None:
            query_emb = np.array(self.embeddings.embed_query(query))
        
        # Store in local memory
        self.cache[query] = {
            "embedding": query_emb,
            "response": response
        }

        # Store in Upstash Redis with TTL (default 24 hours)
        if self.redis_client:
            try:
                val = json.dumps({
                    "response": response,
                    "query": query
                })
                self.redis_client.setex(f"rag:cache:{query}", ttl, val)
                logger.info(f"Cached response in Upstash Redis for query: '{query}'")
            except Exception as e:
                logger.error(f"Failed to set Redis cache: {e}")

        logger.info(f"Cached response locally for query: '{query}'")

    def clear(self):
        self.cache.clear()
        if self.redis_client:
            try:
                keys = self.redis_client.keys("rag:cache:*")
                if keys:
                    self.redis_client.delete(*keys)
                logger.info("Upstash Redis cache cleared.")
            except Exception as e:
                logger.error(f"Error clearing Redis cache: {e}")
        logger.info("Local cache cleared.")

