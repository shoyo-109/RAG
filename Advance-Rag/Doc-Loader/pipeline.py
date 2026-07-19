import os
import time
import logging
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

        # 5. Convert to LangChain Document format
        langchain_docs: List[Document] = []

        for el in sanitized_elements:
            # Document metadata preserving provenance & layout hierarchy
            doc_metadata: Dict[str, Any] = {
                "source": filename,
                "element_id": el.element_id,
                "element_type": el.element_type.value,
                "tenant_id": tenant_id,
                "parser_used": context.selected_parser or parse_result.parser_name,
                "confidence": el.confidence,
                "parent_id": el.parent_id,
                "child_ids": el.child_ids,
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
