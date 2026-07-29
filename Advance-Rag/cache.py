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
    Scoped by session_id to enforce multi-tenant session isolation.
    """
    def __init__(self, embeddings, similarity_threshold: float = 0.95, session_id: Optional[str] = None):
        self.embeddings = embeddings
        self.threshold = similarity_threshold
        self.session_id = session_id
        # Stores local cache entry: (session_id, query_text) -> {"embedding": np.ndarray, "response": str}
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

    def _get_key(self, query: str, session_id: Optional[str] = None) -> str:
        s_id = session_id or self.session_id or "global"
        return f"{s_id}:{query}"

    def get(self, query: str, query_emb: Optional[np.ndarray] = None, session_id: Optional[str] = None) -> Optional[str]:
        cache_key = self._get_key(query, session_id)
        
        # 1. Check Redis exact match first (0ms cloud lookup)
        if self.redis_client:
            try:
                redis_val = self.redis_client.get(f"rag:cache:{cache_key}")
                if redis_val:
                    data = json.loads(redis_val)
                    logger.info("Cache hit: Exact match found in Upstash Redis.")
                    return data.get("response")
            except Exception as e:
                logger.error(f"Redis lookup error: {e}")

        # 2. Check local in-memory exact match
        if cache_key in self.cache:
            logger.info("Cache hit: Exact match found in local memory.")
            return self.cache[cache_key]["response"]

        # 3. Vectorized semantic matching across local cache entries (for the same session)
        target_s_id = session_id or self.session_id or "global"
        session_cache_keys = [k for k in self.cache.keys() if k.startswith(f"{target_s_id}:")]
        
        if not session_cache_keys:
            return None

        if query_emb is None:
            query_emb = np.array(self.embeddings.embed_query(query))
        
        # Vectorized similarity computation over session-specific entries
        embeddings_matrix = np.stack([self.cache[k]["embedding"] for k in session_cache_keys])
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
            best_key = session_cache_keys[best_idx]
            logger.info(f"Cache hit: Semantic match found (similarity: {best_score:.4f}).")
            return self.cache[best_key]["response"]

        return None

    def set(self, query: str, response: str, query_emb: Optional[np.ndarray] = None, ttl: int = 86400, session_id: Optional[str] = None):
        cache_key = self._get_key(query, session_id)
        if query_emb is None:
            query_emb = np.array(self.embeddings.embed_query(query))
        
        # Store in local memory
        self.cache[cache_key] = {
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
                self.redis_client.setex(f"rag:cache:{cache_key}", ttl, val)
                logger.info(f"Cached response in Upstash Redis for query: '{query}' [Key: {cache_key}]")
            except Exception as e:
                logger.error(f"Failed to set Redis cache: {e}")

        logger.info(f"Cached response locally for query: '{query}' [Key: {cache_key}]")

    def clear_session(self, session_id: str):
        """Clears cache entries belonging to a specific session."""
        keys_to_del = [k for k in self.cache.keys() if k.startswith(f"{session_id}:")]
        for k in keys_to_del:
            del self.cache[k]

        if self.redis_client:
            try:
                redis_keys = self.redis_client.keys(f"rag:cache:{session_id}:*")
                if redis_keys:
                    self.redis_client.delete(*redis_keys)
                logger.info(f"Upstash Redis cache cleared for session: {session_id}")
            except Exception as e:
                logger.error(f"Error clearing Redis cache for session {session_id}: {e}")

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


