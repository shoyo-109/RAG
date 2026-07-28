import logging
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document

logger = logging.getLogger("VectorMemory")

class VectorMemory:
    """
    Layer 3: Semantic Conversation Memory
    Vectorizes past conversation turns in Qdrant Cloud vector store for semantic recall across extended sessions.
    """
    def __init__(self, vector_store: Optional[Any] = None):
        self.vector_store = vector_store
        self.turn_counter = 0

    def set_vector_store(self, vector_store: Any):
        """Sets or updates the vector store instance."""
        self.vector_store = vector_store

    def add_turn(self, user_query: str, ai_response: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Stores exchange into vector memory for semantic retrieval.
        """
        if not self.vector_store:
            logger.debug("Vector store not initialized in VectorMemory, skipping vector store insert.")
            return

        self.turn_counter += 1
        content = f"User Query: {user_query}\nAI Response: {ai_response}"
        meta = metadata or {}
        meta.update({
            "turn_index": self.turn_counter,
            "type": "chat_history"
        })

        doc = Document(page_content=content, metadata=meta)
        try:
            self.vector_store.add_documents([doc])
            logger.info(f"VectorMemory: Inserted turn #{self.turn_counter} into semantic vector store.")
        except Exception as e:
            logger.error(f"VectorMemory: Failed to store turn in vector memory: {e}")

    def recall_relevant_turns(self, query: str, top_k: int = 2) -> List[str]:
        """
        Retrieves top_k semantically relevant past turns matching the current query.
        """
        if not self.vector_store:
            return []

        try:
            results = self.vector_store.similarity_search(query, k=top_k)
            return [doc.page_content for doc in results if doc.metadata.get("type") == "chat_history"]
        except Exception as e:
            logger.error(f"VectorMemory: Search error: {e}")
            return []

    def format_as_context(self, query: str, top_k: int = 2) -> str:
        """
        Formats recalled turns into a prompt-ready context string.
        """
        relevant_turns = self.recall_relevant_turns(query, top_k=top_k)
        if not relevant_turns:
            return ""
        return "\n---\n".join(relevant_turns)

    def clear(self):
        self.turn_counter = 0
