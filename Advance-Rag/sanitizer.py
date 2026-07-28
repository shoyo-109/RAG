import re
from typing import Tuple, Optional

class InputSanitizer:
    """
    Production Cybersecurity Input Sanitizer:
    - Defends against OWASP LLM01 (Prompt Injection & Jailbreaks)
    - Strips zero-width unicode characters & null bytes used in evasion attacks
    - Enforces strict input length constraints against Denial of Service (DoS)
    - Neutralizes XSS, HTML/Script tags, and template injection delimiters
    """
    MAX_INPUT_LENGTH = 2000

    INJECTION_PATTERNS = [
        r"ignore\s+(all\s+)?previous\s+(instructions|prompts|rules)",
        r"forget\s+(all\s+)?(previous|prior)\s+(instructions|context)",
        r"new\s+instructions\s*:",
        r"system\s*prompt",
        r"---\s*end\s*(of)?\s*prompt",
        r"pretend\s+you\s+are",
        r"act\s+as\s+(an?\s+)?(unrestricted|jailbroken|unfiltered)\s+AI",
        r"you\s+are\s+now\s+(in\s+)?(developer|dan|jailbreak)\s+mode",
        r"bypass\s+(all\s+)?(restrictions|filters|safety\s+protocols)",
        r"override\s+(all\s+)?(security|safety)\s+settings",
        r"do\s+anything\s+now",
        r"show\s+me\s+your\s+(initial|system)\s+prompt",
        r"reveal\s+your\s+hidden\s+instructions",
    ]

    def __init__(self):
        self.patterns = [re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS]

    def _normalize_text(self, text: str) -> str:
        if not text:
            return ""
        # 1. Strip null bytes and control characters (\x00-\x08, \x0b-\x1f, \x7f)
        text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", text)
        # 2. Strip zero-width unicode characters used in evasion attacks (\u200B, \u200C, \u200D, \uFEFF)
        text = re.sub(r"[\u200B-\u200D\uFEFF]", "", text)
        return text

    def is_suspicious(self, text: str) -> Tuple[bool, Optional[str]]:
        if not text:
            return False, None

        normalized = self._normalize_text(text)

        # DoS Prevention: Check input length constraint
        if len(normalized) > self.MAX_INPUT_LENGTH:
            return True, f"Input size ({len(normalized)} chars) exceeds maximum allowed threshold of {self.MAX_INPUT_LENGTH} characters."

        # Prompt Injection & Jailbreak check
        for pattern in self.patterns:
            if pattern.search(normalized):
                return True, f"Security Violation: Prompt Injection / Jailbreak attempt detected ('{pattern.pattern}')."

        # Check for XSS or Script injection signatures
        if re.search(r"<\s*script[^>]*>", normalized, re.IGNORECASE) or re.search(r"javascript\s*:", normalized, re.IGNORECASE):
            return True, "Security Violation: Script injection attempt detected."

        return False, None

    def sanitize(self, text: str) -> str:
        if not text:
            return ""

        # 1. Normalize unicode & control characters
        text = self._normalize_text(text)

        # 2. Truncate excess length if over limit
        if len(text) > self.MAX_INPUT_LENGTH:
            text = text[:self.MAX_INPUT_LENGTH]

        # 3. Strip dangerous prompt delimiter lines
        text = re.sub(r"[-]{3,}", "", text)
        text = re.sub(r"[=]{3,}", "", text)

        # 4. Escape curly braces to prevent LangChain/Format-String template hijacking
        text = text.replace("{{", "{ {").replace("}}", "} }")

        # 5. Sanitize HTML tags to prevent XSS output reflection
        text = re.sub(r"<[^>]*>", "", text)

        return text.strip()

