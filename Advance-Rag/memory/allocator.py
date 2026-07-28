import logging
from typing import Dict, Any

logger = logging.getLogger("ContextAllocator")

class ContextAllocator:
    """
    Context Window Allocator Guardrail
    Enforces strict token budgeting:
    - Fixed System Prompt: ~100 tokens max
    - RAG Context: Up to 60% of total available budget
    - Memory Context (Working + Episodic + Semantic): 40% of budget
    """
    def __init__(self, max_context_tokens: int = 16000):
        self.max_context_tokens = max_context_tokens
        self.system_prompt_budget = 150
        self.available_budget = max_context_tokens - self.system_prompt_budget
        self.rag_budget = int(self.available_budget * 0.60)
        self.memory_budget = int(self.available_budget * 0.40)

    def allocate(self, rag_text: str, working_memory_text: str, episodic_summary_text: str, semantic_memory_text: str = "") -> Dict[str, str]:
        """
        Trims text inputs if they exceed their designated token budgets.
        Uses 4 chars per token approximation.
        """
        rag_char_limit = self.rag_budget * 4
        memory_char_limit = self.memory_budget * 4

        trimmed_rag = rag_text[:rag_char_limit] if len(rag_text) > rag_char_limit else rag_text

        memory_parts = []
        if episodic_summary_text:
            memory_parts.append(f"Episodic Summary:\n{episodic_summary_text}")
        if semantic_memory_text:
            memory_parts.append(f"Relevant Past Recall:\n{semantic_memory_text}")
        if working_memory_text:
            memory_parts.append(f"Recent Exchange:\n{working_memory_text}")

        combined_memory = "\n\n".join(memory_parts).strip()
        trimmed_memory = combined_memory[:memory_char_limit] if len(combined_memory) > memory_char_limit else combined_memory

        logger.info(f"ContextAllocator: RAG chars={len(trimmed_rag)}/{rag_char_limit}, Memory chars={len(trimmed_memory)}/{memory_char_limit}")

        return {
            "rag_context": trimmed_rag,
            "memory_context": trimmed_memory
        }
