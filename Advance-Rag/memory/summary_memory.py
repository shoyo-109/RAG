import logging
from typing import List, Optional

logger = logging.getLogger("EpisodicSummaryMemory")

class EpisodicSummaryMemory:
    """
    Layer 2: Episodic Summary Memory
    Maintains an evolving, highly condensed executive summary of the conversation.
    """
    def __init__(self):
        self.running_summary: str = ""

    def update_summary(self, new_user_query: str, new_ai_response: str):
        """
        Pure Python heuristic summarizer that appends key topics to the running summary.
        Keeps running summary under ~200 words without requiring external LLM calls.
        """
        clean_query = new_user_query.strip()
        if not clean_query:
            return

        summary_bullet = f"Discussed: '{clean_query[:80]}...'" if len(clean_query) > 80 else f"Discussed: '{clean_query}'"
        
        if not self.running_summary:
            self.running_summary = summary_bullet
        else:
            self.running_summary += f" | {summary_bullet}"

        # Keep summary compact
        bullets = self.running_summary.split(" | ")
        if len(bullets) > 6:
            self.running_summary = " | ".join(bullets[-6:])

    def get_summary(self) -> str:
        return self.running_summary

    def clear(self):
        self.running_summary = ""
