import string
import logging
from typing import List, Tuple
try:
    from .models import DocumentElement
except (ImportError, ValueError):
    from models import DocumentElement

logger = logging.getLogger("IngestionValidator")


class DocumentValidator:
    """
    Validates extracted canonical DocumentElements for quality assurance prior to chunking.
    """

    PRINTABLE_SET = set(string.printable)

    @classmethod
    def validate_elements(cls, elements: List[DocumentElement]) -> Tuple[bool, List[str]]:
        issues = []

        if not elements:
            return False, ["No document elements extracted (empty document)."]

        total_text_length = sum(len(el.text) for el in elements if el.text)
        if total_text_length < 5:
            issues.append("Document extracted text is virtually empty (<5 characters).")

        # Check for unicode corruption / non-printable characters ratio
        non_printable_count = 0
        all_text = "".join(el.text for el in elements if el.text)
        for char in all_text:
            if char not in cls.PRINTABLE_SET and not char.isspace():
                non_printable_count += 1

        if all_text:
            corruption_ratio = non_printable_count / len(all_text)
            if corruption_ratio > 0.35:
                issues.append(f"High text corruption ratio ({corruption_ratio:.1%}) detected.")

        is_valid = len(issues) == 0
        return is_valid, issues
