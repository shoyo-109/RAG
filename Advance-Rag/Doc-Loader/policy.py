import os
import logging
from typing import List
try:
    from .models import PipelineContext
    from .capability_registry import CapabilityRegistry
except (ImportError, ValueError):
    from models import PipelineContext
    from capability_registry import CapabilityRegistry

logger = logging.getLogger("PolicyEngine")


class PolicyEngine:
    """
    Configuration-driven policy engine that determines the prioritized list of candidate parser strategies.
    """

    @staticmethod
    def get_candidate_strategies(context: PipelineContext) -> List[str]:
        ext = os.path.splitext(context.file_path)[-1].lower()
        candidates = []

        # Scanned Images or Raw Graphic formats
        if ext in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
            candidates = ["OCRParserStrategy", "TextParserStrategy"]

        # PDF Documents
        elif ext == ".pdf":
            # Priority 1 depends on whether capability scanner or user detected scanned status
            if context.is_scanned:
                candidates = ["OCRParserStrategy", "PDFParserStrategy", "TextParserStrategy"]
            else:
                candidates = ["PDFParserStrategy", "OCRParserStrategy", "TextParserStrategy"]

        # Office Documents (.docx, .pptx, .xlsx)
        elif ext in [".docx", ".pptx", ".xlsx", ".xls"]:
            candidates = ["OfficeParserStrategy", "OCRParserStrategy", "TextParserStrategy"]

        # Plain Text, Markdown, HTML, CSV
        else:
            candidates = ["TextParserStrategy"]

        logger.info(f"PolicyEngine resolved candidates for {context.filename} ({ext}): {candidates}")
        return candidates
