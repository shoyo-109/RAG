import re
from typing import Tuple, Optional

class InputSanitizer:
    """Sanitize user input before processing to prevent prompt injection."""
    INJECTION_PATTERNS = [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"forget\s+(all\s+)?previous",
        r"new\s+instructions:",
        r"system\s*prompt",
        r"---\s*end\s*(of)?\s*prompt",
        r"pretend\s+you\s+are",
        r"act\s+as\s+(if\s+)?you",
        r"bypass\s+(all\s+)?restrictions",
    ]

    def __init__(self):
        self.patterns = [re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS]

    def is_suspicious(self, text: str) -> Tuple[bool, Optional[str]]:
        for pattern in self.patterns:
            if pattern.search(text):
                return True, f"Suspicious pattern detected: {pattern.pattern}"
        return False, None

    def sanitize(self, text: str) -> str:
        # Remove common delimiters
        text = re.sub(r"[-]{3,}", "", text)
        text = re.sub(r"[=]{3,}", "", text)
        # Escape brackets to prevent template breaking
        text = text.replace("{{", "{ {").replace("}}", "} }")
        return text.strip()
