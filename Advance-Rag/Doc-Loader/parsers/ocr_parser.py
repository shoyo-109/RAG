import os
import uuid
import logging
from typing import List
from .base_parser import BaseParserStrategy, ParseResult
try:
    from ..models import DocumentElement, ElementType, Provenance, CapabilityState, PipelineContext
except (ImportError, ValueError):
    from models import DocumentElement, ElementType, Provenance, CapabilityState, PipelineContext

logger = logging.getLogger("OCRParserStrategy")


class OCRParserStrategy(BaseParserStrategy):
    name = "OCRParserStrategy"
    version = "1.0.0"

    def parse(self, file_path: str, context: PipelineContext) -> ParseResult:
        elements: List[DocumentElement] = []
        ext = os.path.splitext(file_path)[-1].lower()

        try:
            # Check if unstructured is available
            try:
                from unstructured.partition.auto import partition
                raw_elements = partition(filename=file_path, strategy="hi_res" if ext in [".pdf", ".png", ".jpg", ".jpeg"] else "auto")
                
                for idx, el in enumerate(raw_elements):
                    text = str(el).strip()
                    if not text:
                        continue

                    category = getattr(el, "category", "Paragraph").upper()
                    el_type = ElementType.PARAGRAPH
                    if "HEADING" in category or "TITLE" in category:
                        el_type = ElementType.HEADING
                    elif "TABLE" in category:
                        el_type = ElementType.TABLE

                    prov = Provenance(
                        source_file=file_path,
                        page=getattr(el.metadata, "page_number", 1),
                        parser_name="UnstructuredHiRes",
                        parser_version="1.0"
                    )

                    elements.append(DocumentElement(
                        element_id=f"ocr_el_{uuid.uuid4().hex[:8]}",
                        element_type=el_type,
                        text=text,
                        html_content=getattr(el.metadata, "text_as_html", None),
                        metadata={"category": category},
                        confidence=0.90,
                        provenance=prov
                    ))

                if elements:
                    return ParseResult(
                        elements=elements,
                        confidence=0.90,
                        status=CapabilityState.SUCCESS,
                        parser_name=self.name,
                        parser_version=self.version
                    )
            except Exception as unstruct_err:
                logger.warning(f"Unstructured Hi-Res partition unavailable/failed: {unstruct_err}. Falling back to pytesseract...")

            # Fallback to direct pytesseract / PIL
            import pytesseract
            from PIL import Image

            if ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
                image = Image.open(file_path)
                ocr_text = pytesseract.image_to_string(image)
                
                if ocr_text.strip():
                    prov = Provenance(
                        source_file=file_path,
                        page=1,
                        parser_name="PyTesseractDirect",
                        parser_version="1.0"
                    )
                    elements.append(DocumentElement(
                        element_id=f"ocr_el_{uuid.uuid4().hex[:8]}",
                        element_type=ElementType.PARAGRAPH,
                        text=ocr_text.strip(),
                        confidence=0.75,
                        provenance=prov
                    ))

                    return ParseResult(
                        elements=elements,
                        confidence=0.75,
                        status=CapabilityState.SUCCESS,
                        parser_name=self.name,
                        parser_version=self.version
                    )

            return ParseResult(
                elements=[],
                confidence=0.0,
                status=CapabilityState.FAILED,
                error_message="OCR processing produced no usable text",
                parser_name=self.name,
                parser_version=self.version
            )

        except Exception as e:
            logger.error(f"OCRParserStrategy failed: {e}")
            return ParseResult(
                elements=[],
                confidence=0.0,
                status=CapabilityState.FAILED,
                error_message=str(e),
                parser_name=self.name,
                parser_version=self.version
            )
