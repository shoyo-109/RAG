import logging
import json
import time
from typing import Dict, Any
try:
    from .models import PipelineContext
except (ImportError, ValueError):
    from models import PipelineContext

logger = logging.getLogger("IngestionObservability")


class IngestionObservability:
    """
    Logs structured telemetry for LangSmith and local audit logs.
    """

    @classmethod
    def record_telemetry(cls, context: PipelineContext, start_time: float, element_count: int, success: bool):
        latency_ms = (time.time() - start_time) * 1000.0

        telemetry_data: Dict[str, Any] = {
            "event": "document_ingestion",
            "filename": context.filename,
            "mime_type": context.mime_type,
            "file_size_bytes": context.file_size_bytes,
            "selected_parser": context.selected_parser,
            "fallback_history": context.fallback_history,
            "element_count": element_count,
            "latency_ms": round(latency_ms, 2),
            "success": success
        }

        context.telemetry = telemetry_data
        logger.info(f"INGESTION_TELEMETRY: {json.dumps(telemetry_data)}")
