import logging
from typing import Dict, Any, List, Optional
from .working_memory import WorkingMemory
from .summary_memory import EpisodicSummaryMemory
from .vector_memory import VectorMemory
from .topic_tracker import TopicTracker
from .allocator import ContextAllocator

logger = logging.getLogger("MultiLayerMemoryManager")

class MultiLayerMemoryManager:
    """
    Unified Orchestrator for the 4-Layer Memory Architecture:
    1. Working Memory (Short-term verbatim buffer)
    2. Episodic Summary Memory (Condensed executive summary)
    3. Semantic Vector Memory (Qdrant vector search recall)
    4. Topic Shift Tracker (Topic transition & H3 header signal)
    """
    def __init__(self, capacity: int = 5, max_context_tokens: int = 16000, vector_store: Optional[Any] = None):
        self.working_memory = WorkingMemory(capacity=capacity)
        self.summary_memory = EpisodicSummaryMemory()
        self.vector_memory = VectorMemory(vector_store=vector_store)
        self.topic_tracker = TopicTracker()
        self.allocator = ContextAllocator(max_context_tokens=max_context_tokens)

    def set_vector_store(self, vector_store: Any):
        """Passes vector store reference to vector memory layer."""
        self.vector_memory.set_vector_store(vector_store)

    def process_incoming_query(self, query: str) -> Dict[str, Any]:
        """
        Analyzes incoming query against active conversation state before retrieval.
        Returns topic shift flags and formatted memory context from all layers.
        """
        topic_analysis = self.topic_tracker.analyze_turn(query)
        working_context = self.working_memory.format_as_context()
        episodic_context = self.summary_memory.get_summary()
        semantic_context = self.vector_memory.format_as_context(query, top_k=2)

        return {
            "topic_analysis": topic_analysis,
            "working_context": working_context,
            "episodic_summary": episodic_context,
            "semantic_context": semantic_context
        }

    def record_completed_turn(self, user_query: str, ai_response: str):
        """
        Records completed user/AI turn into working memory, episodic summary, and vector memory.
        """
        self.working_memory.add_turn(user_query, ai_response)
        self.summary_memory.update_summary(user_query, ai_response)
        self.vector_memory.add_turn(user_query, ai_response)

    def prepare_allocated_context(self, rag_text: str, user_query: str) -> Dict[str, Any]:
        """
        Bundles RAG context and multi-layer memory context under strict token budget limits.
        """
        memory_state = self.process_incoming_query(user_query)
        allocated = self.allocator.allocate(
            rag_text=rag_text,
            working_memory_text=memory_state["working_context"],
            episodic_summary_text=memory_state["episodic_summary"],
            semantic_memory_text=memory_state["semantic_context"]
        )

        return {
            "rag_context": allocated["rag_context"],
            "memory_context": allocated["memory_context"],
            "should_render_top_header": memory_state["topic_analysis"]["should_render_top_header"],
            "active_topic": memory_state["topic_analysis"]["active_topic"]
        }

    def clear(self):
        self.working_memory.clear()
        self.summary_memory.clear()
        self.vector_memory.clear()
        self.topic_tracker.clear()

