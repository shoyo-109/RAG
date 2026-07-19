import time
from enum import Enum
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


class CapabilityState(str, Enum):
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"


class ElementType(str, Enum):
    HEADING = "HEADING"
    PARAGRAPH = "PARAGRAPH"
    TABLE = "TABLE"
    FIGURE = "FIGURE"
    CAPTION = "CAPTION"
    LIST_ITEM = "LIST_ITEM"
    CODE = "CODE"
    UNKNOWN = "UNKNOWN"


@dataclass
class BoundingBox:
    x0: float = 0.0
    y0: float = 0.0
    x1: float = 0.0
    y1: float = 0.0
    page: int = 1


@dataclass
class OCRConfidence:
    word_confidence: float = 1.0
    page_confidence: float = 1.0
    doc_confidence: float = 1.0


@dataclass
class Provenance:
    source_file: str
    page: int = 1
    parser_name: str = "UnknownParser"
    parser_version: str = "1.0.0"
    timestamp: float = field(default_factory=time.time)
    bounding_box: Optional[BoundingBox] = None
    transformation_history: List[str] = field(default_factory=list)


@dataclass
class DocumentElement:
    element_id: str
    element_type: ElementType
    text: str
    html_content: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    parent_id: Optional[str] = None
    child_ids: List[str] = field(default_factory=list)
    provenance: Optional[Provenance] = None


@dataclass
class LayoutGraph:
    elements: List[DocumentElement] = field(default_factory=list)
    root_ids: List[str] = field(default_factory=list)
    parent_child_map: Dict[str, List[str]] = field(default_factory=dict)

    def add_element(self, element: DocumentElement, parent_id: Optional[str] = None):
        self.elements.append(element)
        if parent_id:
            element.parent_id = parent_id
            if parent_id not in self.parent_child_map:
                self.parent_child_map[parent_id] = []
            self.parent_child_map[parent_id].append(element.element_id)
        else:
            self.root_ids.append(element.element_id)


@dataclass
class PipelineContext:
    file_path: str
    filename: str
    mime_type: str = "application/octet-stream"
    file_size_bytes: int = 0
    tenant_id: str = "default_tenant"
    page_count: int = 1
    is_scanned: bool = False
    config: Dict[str, Any] = field(default_factory=dict)
    capability_states: Dict[str, CapabilityState] = field(default_factory=dict)
    selected_parser: Optional[str] = None
    fallback_history: List[Dict[str, Any]] = field(default_factory=list)
    telemetry: Dict[str, Any] = field(default_factory=dict)
