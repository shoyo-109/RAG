from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum

class PriorityLevel(Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

class BlockType(Enum):
    EXECUTIVE_SUMMARY = "EXECUTIVE_SUMMARY"
    KEY_HIGHLIGHTS = "KEY_HIGHLIGHTS"
    CATEGORIZED_DATA = "CATEGORIZED_DATA"
    DEEP_DETAILS = "DEEP_DETAILS"
    CLOSING_SYNTHESIS = "CLOSING_SYNTHESIS"

@dataclass
class CategorizedGroup:
    category_name: str
    items: List[str]

@dataclass
class PresentationBlock:
    block_type: BlockType
    priority: PriorityLevel
    content: str
    grouped_data: Optional[List[CategorizedGroup]] = None
    heading: Optional[str] = None
