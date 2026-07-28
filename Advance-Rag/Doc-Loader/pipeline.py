import os
import time
import logging
import json
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document

try:
    from .models import PipelineContext, CapabilityState
    from .security import IngestionSecurityGuardrails
    from .factory import StrategyFactory
    from .qa_layer import QualityAssuranceLayer
    from .observability import IngestionObservability
except (ImportError, ValueError):
    from models import PipelineContext, CapabilityState
    from security import IngestionSecurityGuardrails
    from factory import StrategyFactory
    from qa_layer import QualityAssuranceLayer
    from observability import IngestionObservability

logger = logging.getLogger("IngestionPipeline")


class IngestionPipeline:
    """
    Production-grade document ingestion pipeline orchestrator.
    Combines Validation, Capability-Driven Strategy Factory, 3-Level Fallbacks,
    Canonical Normalization, QA, and LangChain Document conversion.
    """

    @classmethod
    def load_document(cls, file_path: str, tenant_id: str = "default_tenant", config: Optional[Dict[str, Any]] = None) -> List[Document]:
        start_time = time.time()
        filename = os.path.basename(file_path)

        # 1. Security Check & Validation
        is_safe, security_msg, detected_mime = IngestionSecurityGuardrails.validate_file(file_path)
        if not is_safe:
            logger.error(f"Security validation failed for {filename}: {security_msg}")
            raise ValueError(f"Security Validation Failed: {security_msg}")

        file_size = os.path.getsize(file_path)

        # 2. Construct PipelineContext
        context = PipelineContext(
            file_path=file_path,
            filename=filename,
            mime_type=detected_mime,
            file_size_bytes=file_size,
            tenant_id=tenant_id,
            config=config or {}
        )

        # 3. Strategy Factory Execution with 3-Level Fallbacks
        parse_result, fallback_history = StrategyFactory.execute_with_fallbacks(context)
        context.fallback_history = fallback_history

        # 4. Quality Assurance & Layout Graph Construction
        sanitized_elements, layout_graph = QualityAssuranceLayer.process_and_construct_layout(
            parse_result.elements, context
        )

        # 5. Convert to LangChain Document format with Hierarchical Parent-Child Metadata
        element_to_parent_text = {}
        element_to_breadcrumb = {}

        active_breadcrumb = "General Document Content"
        section_elements = []

        def flush_section():
            if not section_elements:
                return
            section_full_text = "\n".join([e.text for e in section_elements if e.text])
            for e in section_elements:
                element_to_parent_text[e.element_id] = section_full_text
                element_to_breadcrumb[e.element_id] = active_breadcrumb

        for el in sanitized_elements:
            el_type_val = getattr(el.element_type, "value", str(el.element_type)).lower() if hasattr(el, "element_type") else ""
            is_heading = (
                el_type_val in ["heading", "title", "header"]
            ) or (el.text and (
                el.text.strip().startswith("#") or 
                el.text.strip().isupper() or 
                (len(el.text.split()) <= 5 and any(kw in el.text.lower() for kw in ["experience", "education", "projects", "skills", "summary", "activities"]))
            ))

            if is_heading:
                flush_section()
                active_breadcrumb = el.text.strip("# ").strip()
                section_elements = [el]
            else:
                section_elements.append(el)

        flush_section()

        langchain_docs: List[Document] = []

        for el in sanitized_elements:
            # Document metadata preserving provenance & layout hierarchy
            doc_metadata: Dict[str, Any] = {
                "source": filename,
                "element_id": el.element_id,
                "element_type": getattr(el.element_type, "value", str(el.element_type)) if hasattr(el, "element_type") else "text",
                "tenant_id": tenant_id,
                "parser_used": context.selected_parser or parse_result.parser_name,
                "confidence": el.confidence,
                "parent_id": el.parent_id or "",
                "child_ids": json.dumps(el.child_ids),
                "parent_text": element_to_parent_text.get(el.element_id, el.text),
                "breadcrumb_path": element_to_breadcrumb.get(el.element_id, "General Context")
            }

            if el.provenance:
                doc_metadata["page"] = el.provenance.page
                doc_metadata["parser_version"] = el.provenance.parser_version

            if el.html_content:
                doc_metadata["table_html"] = el.html_content

            if el.metadata:
                doc_metadata.update(el.metadata)

            langchain_docs.append(Document(
                page_content=el.text,
                metadata=doc_metadata
            ))

        # If zero documents extracted, create fallback document with filename notice
        if not langchain_docs:
            logger.warning(f"No text extracted from {filename}. Creating fallback document entry.")
            langchain_docs.append(Document(
                page_content=f"[Document: {filename} - No extractable text content]",
                metadata={"source": filename, "is_fallback": True}
            ))

        # 6. Record Telemetry
        IngestionObservability.record_telemetry(
            context=context,
            start_time=start_time,
            element_count=len(langchain_docs),
            success=parse_result.status == CapabilityState.SUCCESS
        )

        logger.info(f"IngestionPipeline completed for {filename}: generated {len(langchain_docs)} LangChain Document chunks.")
        return langchain_docs
