import logging
from typing import List, Tuple
try:
    from .models import DocumentElement, LayoutGraph, ElementType, PipelineContext
    from .validator import DocumentValidator
except (ImportError, ValueError):
    from models import DocumentElement, LayoutGraph, ElementType, PipelineContext
    from validator import DocumentValidator

logger = logging.getLogger("QualityAssuranceLayer")


class QualityAssuranceLayer:
    """
    Quality Assurance Layer that validates output, cleans text, and constructs structural LayoutGraph.
    """

    @classmethod
    def process_and_construct_layout(cls, elements: List[DocumentElement], context: PipelineContext) -> Tuple[List[DocumentElement], LayoutGraph]:
        is_valid, issues = DocumentValidator.validate_elements(elements)
        if not is_valid:
            logger.warning(f"QA Validation warnings for {context.filename}: {issues}")

        graph = LayoutGraph()
        last_heading_id = None
        sanitized_elements: List[DocumentElement] = []

        for el in elements:
            # Clean control characters
            clean_text = "".join(ch for ch in el.text if ord(ch) >= 32 or ch in "\n\t")
            el.text = clean_text.strip()

            if not el.text:
                continue

            if el.element_type == ElementType.HEADING:
                last_heading_id = el.element_id
                graph.add_element(el, parent_id=None)
            else:
                graph.add_element(el, parent_id=last_heading_id)

            sanitized_elements.append(el)

        logger.info(f"QA processed {len(sanitized_elements)} elements for {context.filename}. Constructed layout graph with {len(graph.root_ids)} root nodes.")
        return sanitized_elements, graph
