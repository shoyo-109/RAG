import os
import shutil
import logging
import importlib
from typing import Dict, List, Optional
try:
    from .models import CapabilityState
except (ImportError, ValueError):
    from models import CapabilityState

logger = logging.getLogger("CapabilityRegistry")


class CapabilityDetector:
    """
    Probes system environment and installed Python packages to determine available capabilities.
    """

    @staticmethod
    def check_tesseract() -> CapabilityState:
        # Check environment variable overrides
        tesseract_cmd = os.getenv("TESSERACT_CMD")
        if tesseract_cmd and os.path.exists(tesseract_cmd):
            return CapabilityState.AVAILABLE
        
        # Check system PATH
        if shutil.which("tesseract"):
            return CapabilityState.AVAILABLE

        # Check pytesseract import and binary setup
        try:
            import pytesseract
            if pytesseract.pytesseract.tesseract_cmd and os.path.exists(pytesseract.pytesseract.tesseract_cmd):
                return CapabilityState.AVAILABLE
        except Exception:
            pass

        return CapabilityState.UNAVAILABLE

    @staticmethod
    def check_poppler() -> CapabilityState:
        poppler_path = os.getenv("POPPLER_PATH")
        if poppler_path and os.path.exists(poppler_path):
            return CapabilityState.AVAILABLE
        
        if shutil.which("pdftoppm") or shutil.which("pdfinfo"):
            return CapabilityState.AVAILABLE
        
        return CapabilityState.UNAVAILABLE

    @staticmethod
    def check_python_package(package_name: str) -> CapabilityState:
        try:
            importlib.import_module(package_name)
            return CapabilityState.AVAILABLE
        except ImportError:
            return CapabilityState.UNAVAILABLE


class CapabilityRegistry:
    """
    Singleton / Central Registry holding system availability states.
    Never crashes due to missing dependencies.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CapabilityRegistry, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self._registry: Dict[str, Dict[str, CapabilityState]] = {}
        self.refresh()
        self._initialized = True

    def refresh(self):
        """Performs a full scan of system capabilities."""
        logger.info("Scanning system capability registry...")
        
        self._registry = {
            "ocr": {
                "tesseract": CapabilityDetector.check_tesseract(),
                "pytesseract": CapabilityDetector.check_python_package("pytesseract"),
                "paddleocr": CapabilityDetector.check_python_package("paddleocr"),
            },
            "pdf_render": {
                "poppler": CapabilityDetector.check_poppler(),
                "pymupdf": CapabilityDetector.check_python_package("fitz"),
                "pdf2image": CapabilityDetector.check_python_package("pdf2image"),
                "pypdf": CapabilityDetector.check_python_package("pypdf"),
            },
            "unstructured": {
                "core": CapabilityDetector.check_python_package("unstructured"),
                "pdf": CapabilityDetector.check_python_package("unstructured.partition.pdf"),
                "docx": CapabilityDetector.check_python_package("docx"),
                "pptx": CapabilityDetector.check_python_package("pptx"),
                "xlsx": CapabilityDetector.check_python_package("openpyxl"),
            },
            "text": {
                "bs4": CapabilityDetector.check_python_package("bs4"),
                "markdown": CapabilityDetector.check_python_package("markdown"),
            }
        }
        
        logger.info(f"Capability scan completed: {self.get_summary()}")

    def is_available(self, category: str, capability: str) -> bool:
        cat_dict = self._registry.get(category, {})
        state = cat_dict.get(capability, CapabilityState.UNAVAILABLE)
        return state == CapabilityState.AVAILABLE

    def get_state(self, category: str, capability: str) -> CapabilityState:
        return self._registry.get(category, {}).get(capability, CapabilityState.UNAVAILABLE)

    def get_best_available(self, category: str, priority_list: List[str]) -> Optional[str]:
        """Returns the highest priority capability that is currently AVAILABLE."""
        for cap in priority_list:
            if self.is_available(category, cap):
                return cap
        return None

    def get_summary(self) -> Dict[str, Dict[str, str]]:
        summary = {}
        for cat, caps in self._registry.items():
            summary[cat] = {cap: state.value for cap, state in caps.items()}
        return summary
