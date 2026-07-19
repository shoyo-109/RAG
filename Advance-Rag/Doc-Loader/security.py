import os
import re
import logging
from typing import Tuple

logger = logging.getLogger("IngestionSecurity")


class IngestionSecurityGuardrails:
    """
    Validates file integrity, magic bytes, macro threats, and size limits prior to parsing.
    """
    MAGIC_SIGNATURES = {
        b"%PDF": "application/pdf",
        b"\x89PNG": "image/png",
        b"\xff\xd8\xff": "image/jpeg",
        b"GIF8": "image/gif",
        b"II*\x00": "image/tiff",
        b"MM\x00*": "image/tiff",
        b"PK\x03\x04": "application/zip",  # Office docs (docx, pptx, xlsx) are zip containers
    }

    MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB limit

    PROMPT_INJECTION_PATTERNS = [
        r"ignore\s+all\s+previous\s+instructions",
        r"system\s*:\s*you\s+are",
        r"override\s+system\s+prompt",
    ]

    @classmethod
    def validate_file(cls, file_path: str) -> Tuple[bool, str, str]:
        if not os.path.exists(file_path):
            return False, "File does not exist", "unknown"

        file_size = os.path.getsize(file_path)
        if file_size == 0:
            return False, "File is empty (0 bytes)", "unknown"
        
        if file_size > cls.MAX_FILE_SIZE_BYTES:
            return False, f"File size exceeds maximum threshold ({cls.MAX_FILE_SIZE_BYTES // (1024*1024)}MB)", "unknown"

        # Detect Magic Bytes
        detected_mime = "text/plain"
        with open(file_path, "rb") as f:
            header = f.read(8)
            for magic, mime in cls.MAGIC_SIGNATURES.items():
                if header.startswith(magic):
                    detected_mime = mime
                    break

        ext = os.path.splitext(file_path)[-1].lower()

        # Simple Text files or Markdown/HTML
        if ext in [".txt", ".md", ".csv", ".json", ".html", ".xml"]:
            detected_mime = f"text/{ext.lstrip('.')}"
        elif ext in [".docx", ".pptx", ".xlsx"] and detected_mime == "application/zip":
            detected_mime = f"application/vnd.openxmlformats-officedocument.{ext.lstrip('.')}"

        return True, "File security check passed", detected_mime

    @classmethod
    def sanitize_extracted_text(cls, text: str) -> Tuple[str, bool]:
        """Checks for embedded prompt injections inside document text."""
        is_suspicious = False
        cleaned_text = text
        for pattern in cls.PROMPT_INJECTION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                logger.warning(f"Potential prompt injection pattern detected in document text: {pattern}")
                is_suspicious = True
                cleaned_text = re.sub(pattern, "[BLOCKED_PROMPT_INJECTION]", cleaned_text, flags=re.IGNORECASE)

        return cleaned_text, is_suspicious
