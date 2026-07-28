import re

class PIIDetector:
    """
    Detect and mask personally identifiable information (PII) and secret credentials
    (API keys, JWT tokens, SSNs, credit cards, emails, IP addresses) before caching or logging.
    """
    PATTERNS = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
        "ip_address": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
        "api_key": r"\b(sk-[A-Za-z0-9]{32,}|nvapi-[A-Za-z0-9_-]{32,}|AKIA[0-9A-Z]{16}|lsv2_pt_[A-Za-z0-9_-]+)\b",
        "secret_token": r"\b(bearer\s+[A-Za-z0-9\-._~+/]+=*|token:[A-Za-z0-9_-]{20,})\b",
    }

    def __init__(self):
        combined_parts = [f"(?P<{name}>{pattern})" for name, pattern in self.PATTERNS.items()]
        self.combined_regex = re.compile("|".join(combined_parts), re.IGNORECASE)

    def detect(self, text: str) -> dict:
        found = {}
        if not text:
            return found
        for match in self.combined_regex.finditer(text):
            for pii_type in self.PATTERNS.keys():
                val = match.group(pii_type)
                if val:
                    if pii_type not in found:
                        found[pii_type] = []
                    if val not in found[pii_type]:
                        found[pii_type].append(val)
        return found

    def mask(self, text: str) -> str:
        if not text:
            return ""
        def repl(match):
            for pii_type in self.PATTERNS.keys():
                if match.group(pii_type):
                    return f"[{pii_type.upper()} REDACTED]"
            return match.group(0)
        return self.combined_regex.sub(repl, text)

