from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import time

@dataclass
class ChatTurn:
    user_query: str
    ai_response: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

class WorkingMemory:
    """
    Layer 1: Working Memory Buffer
    Stores recent verbatim user/AI conversation turns in-memory.
    """
    def __init__(self, capacity: int = 5):
        self.capacity = capacity
        self.history: List[ChatTurn] = []

    def add_turn(self, user_query: str, ai_response: str, metadata: Optional[Dict[str, Any]] = None):
        turn = ChatTurn(
            user_query=user_query,
            ai_response=ai_response,
            metadata=metadata or {}
        )
        self.history.append(turn)
        if len(self.history) > self.capacity:
            self.history.pop(0)

    def get_recent_turns(self, limit: Optional[int] = None) -> List[ChatTurn]:
        if limit is None:
            return list(self.history)
        return list(self.history[-limit:])

    def format_as_context(self) -> str:
        if not self.history:
            return ""
        formatted = []
        for turn in self.history:
            formatted.append(f"User: {turn.user_query}\nAssistant: {turn.ai_response}")
        return "\n\n".join(formatted)

    def clear(self):
        self.history.clear()
