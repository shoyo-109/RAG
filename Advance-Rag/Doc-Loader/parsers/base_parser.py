from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass, field
try:
    from ..models import DocumentElement, CapabilityState, PipelineContext
except (ImportError, ValueError):
    try:
        from models import DocumentElement, CapabilityState, PipelineContext
    except (ImportError, ValueError):
        import sys, os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from models import DocumentElement, CapabilityState, PipelineContext


@dataclass
class ParseResult:
    elements: List[DocumentElement] = field(default_factory=list)
    confidence: float = 1.0
    status: CapabilityState = CapabilityState.SUCCESS
    error_message: Optional[str] = None
    parser_name: str = "BaseParser"
    parser_version: str = "1.0.0"


class BaseParserStrategy(ABC):
    """
    Abstract base class for all document parsing plugin strategies.
    """
    name: str = "BaseParser"
    version: str = "1.0.0"

    @abstractmethod
    def parse(self, file_path: str, context: PipelineContext) -> ParseResult:
        """
        Parses a document file and returns a canonical ParseResult.
        Must catch internal exceptions safely and return status=CapabilityState.FAILED rather than crashing.
        """
        pass
