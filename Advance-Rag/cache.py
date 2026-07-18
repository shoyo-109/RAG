import numpy as np
import logging
from typing import Dict, Optional

logger = logging.getLogger("AdvancedRAG")

class RAGCache:
    """
    Dynamic semantic caching system to cache frequent queries in real-time.
    Uses vectorized cosine similarity of query embeddings to detect similar queries.
    """
    def __init__(self, embeddings, similarity_threshold: float = 0.95):
        self.embeddings = embeddings
        self.threshold = similarity_threshold
        # Stores cache entry: query_text -> {"embedding": np.ndarray, "response": str}
        self.cache: Dict[str, Dict] = {}

    def get(self, query: str, query_emb: Optional[np.ndarray] = None) -> Optional[str]:
        if not self.cache:
            return None
        
        # Check exact match first - takes 0ms
        if query in self.cache:
            logger.info("Cache hit: Exact match found.")
            return self.cache[query]["response"]

        # If exact match fails, do vectorized semantic matching
        if query_emb is None:
            query_emb = np.array(self.embeddings.embed_query(query))
        
        cached_keys = list(self.cache.keys())
        
        # Vectorized similarity computation
        # 1. Stack all cached embeddings to make a 2D matrix of shape (M, D)
        embeddings_matrix = np.stack([self.cache[k]["embedding"] for k in cached_keys])
        
        # 2. Vectorized cosine similarity:
        # dot_products = matrix @ query_emb -> shape (M,)
        dot_products = np.dot(embeddings_matrix, query_emb)
        
        # norms of cached embeddings -> shape (M,)
        norms_cached = np.linalg.norm(embeddings_matrix, axis=1)
        norm_query = np.linalg.norm(query_emb)
        
        # Avoid division by zero
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

    def set(self, query: str, response: str, query_emb: Optional[np.ndarray] = None):
        if query_emb is None:
            query_emb = np.array(self.embeddings.embed_query(query))
        self.cache[query] = {
            "embedding": query_emb,
            "response": response
        }
        logger.info(f"Cached response for query: '{query}'")

    def clear(self):
        self.cache.clear()
        logger.info("Cache cleared.")
