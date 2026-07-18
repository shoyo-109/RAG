import re
from typing import Tuple, Optional

try:
    from .pii_detector import PIIDetector
except (ImportError, ValueError):
    from pii_detector import PIIDetector



class OutputValidator:
    """Validate LLM outputs before returning to user."""
    def __init__(self):
        self.pii_detector = PIIDetector()

    def validate(self, output: str) -> Tuple[bool, str, Optional[str]]:
        # Check PII leakage in LLM output
        pii_found = self.pii_detector.detect(output)
        if pii_found:
            cleaned = self.pii_detector.mask(output)
            return False, cleaned, f"PII leakage detected and masked: {list(pii_found.keys())}"

        # Check harmful content
        harmful_patterns = [
            r"here('s| is) (how|the way) to (hack|steal|attack)",
            r"password is",
            r"api[_\s]?key",
        ]
        for pattern in harmful_patterns:
            if re.search(pattern, output, re.IGNORECASE):
                return False, "[CONTENT BLOCKED]", "Harmful content signature detected"

        return True, output, None
