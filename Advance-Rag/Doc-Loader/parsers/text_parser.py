import uuid
import logging
from typing import List
from .base_parser import BaseParserStrategy, ParseResult
try:
    from ..models import DocumentElement, ElementType, Provenance, CapabilityState, PipelineContext
except (ImportError, ValueError):
    from models import DocumentElement, ElementType, Provenance, CapabilityState, PipelineContext

logger = logging.getLogger("TextParserStrategy")


class TextParserStrategy(BaseParserStrategy):
    name = "TextParserStrategy"
    version = "1.0.0"

    def parse(self, file_path: str, context: PipelineContext) -> ParseResult:
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

            if not content.strip():
                return ParseResult(
                    elements=[],
                    confidence=0.0,
                    status=CapabilityState.LOW_CONFIDENCE,
                    error_message="File content is empty",
                    parser_name=self.name,
                    parser_version=self.version
                )

            # Split paragraphs
            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
            elements: List[DocumentElement] = []

            for idx, p in enumerate(paragraphs):
                el_id = f"text_el_{uuid.uuid4().hex[:8]}"
                # Classify header vs paragraph
                is_heading = len(p) < 80 and not p.endswith(".") and (p.startswith("#") or idx == 0)
                el_type = ElementType.HEADING if is_heading else ElementType.PARAGRAPH
                clean_text = p.lstrip("#").strip()

                prov = Provenance(
                    source_file=file_path,
                    page=1,
                    parser_name=self.name,
                    parser_version=self.version
                )

                elements.append(DocumentElement(
                    element_id=el_id,
                    element_type=el_type,
                    text=clean_text,
                    metadata={"paragraph_index": idx},
                    confidence=1.0,
                    provenance=prov
                ))

            return ParseResult(
                elements=elements,
                confidence=1.0,
                status=CapabilityState.SUCCESS,
                parser_name=self.name,
                parser_version=self.version
            )
        except Exception as e:
            logger.error(f"TextParserStrategy failed: {e}")
            return ParseResult(
                elements=[],
                confidence=0.0,
                status=CapabilityState.FAILED,
                error_message=str(e),
                parser_name=self.name,
                parser_version=self.version
            )
