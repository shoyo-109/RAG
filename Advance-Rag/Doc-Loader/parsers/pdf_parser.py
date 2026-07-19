import uuid
import logging
from typing import List
from .base_parser import BaseParserStrategy, ParseResult
try:
    from ..models import DocumentElement, ElementType, Provenance, CapabilityState, PipelineContext
except (ImportError, ValueError):
    from models import DocumentElement, ElementType, Provenance, CapabilityState, PipelineContext

logger = logging.getLogger("PDFParserStrategy")


class PDFParserStrategy(BaseParserStrategy):
    name = "PDFParserStrategy"
    version = "1.0.0"

    def parse(self, file_path: str, context: PipelineContext) -> ParseResult:
        try:
            elements: List[DocumentElement] = []
            pages_read = 0

            # Attempt 1: PyMuPDF (fitz)
            try:
                import fitz
                doc = fitz.open(file_path)
                pages_read = len(doc)
                context.page_count = pages_read

                for page_idx in range(pages_read):
                    page = doc[page_idx]
                    text = page.get_text()
                    if not text.strip():
                        continue

                    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
                    for p in paragraphs:
                        el_id = f"pdf_el_{uuid.uuid4().hex[:8]}"
                        prov = Provenance(
                            source_file=file_path,
                            page=page_idx + 1,
                            parser_name="PyMuPDF",
                            parser_version=getattr(fitz, "__version__", "1.0")
                        )
                        elements.append(DocumentElement(
                            element_id=el_id,
                            element_type=ElementType.PARAGRAPH,
                            text=p,
                            metadata={"page": page_idx + 1},
                            confidence=0.95,
                            provenance=prov
                        ))
            except ImportError:
                # Attempt 2: pypdf fallback
                from pypdf import PdfReader
                reader = PdfReader(file_path)
                pages_read = len(reader.pages)
                context.page_count = pages_read

                for page_idx, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    if not text.strip():
                        continue

                    el_id = f"pdf_el_{uuid.uuid4().hex[:8]}"
                    prov = Provenance(
                        source_file=file_path,
                        page=page_idx + 1,
                        parser_name="pypdf",
                        parser_version="1.0"
                    )
                    elements.append(DocumentElement(
                        element_id=el_id,
                        element_type=ElementType.PARAGRAPH,
                        text=text.strip(),
                        metadata={"page": page_idx + 1},
                        confidence=0.85,
                        provenance=prov
                    ))

            # Evaluate confidence: if 0 elements extracted from multi-page PDF, it is likely scanned!
            if not elements:
                logger.info("PDFParserStrategy extracted 0 text elements. Likely a scanned PDF (LOW_CONFIDENCE).")
                return ParseResult(
                    elements=[],
                    confidence=0.1,
                    status=CapabilityState.LOW_CONFIDENCE,
                    error_message="PDF contains no selectable text (scanned PDF)",
                    parser_name=self.name,
                    parser_version=self.version
                )

            return ParseResult(
                elements=elements,
                confidence=0.95,
                status=CapabilityState.SUCCESS,
                parser_name=self.name,
                parser_version=self.version
            )

        except Exception as e:
            logger.error(f"PDFParserStrategy failed: {e}")
            return ParseResult(
                elements=[],
                confidence=0.0,
                status=CapabilityState.FAILED,
                error_message=str(e),
                parser_name=self.name,
                parser_version=self.version
            )
