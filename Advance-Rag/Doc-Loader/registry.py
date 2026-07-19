import logging
from typing import Dict, Type
try:
    from .parsers.base_parser import BaseParserStrategy
    from .parsers.text_parser import TextParserStrategy
    from .parsers.pdf_parser import PDFParserStrategy
    from .parsers.ocr_parser import OCRParserStrategy
    from .parsers.office_parser import OfficeParserStrategy
except (ImportError, ValueError):
    from parsers.base_parser import BaseParserStrategy
    from parsers.text_parser import TextParserStrategy
    from parsers.pdf_parser import PDFParserStrategy
    from parsers.ocr_parser import OCRParserStrategy
    from parsers.office_parser import OfficeParserStrategy

logger = logging.getLogger("ParserRegistry")


class ParserRegistry:
    """
    Registry holding instantiable parser strategy implementations.
    """
    _strategies: Dict[str, BaseParserStrategy] = {}

    @classmethod
    def register(cls, name: str, strategy: BaseParserStrategy):
        cls._strategies[name] = strategy

    @classmethod
    def get(cls, name: str) -> BaseParserStrategy:
        if name not in cls._strategies:
            raise KeyError(f"Strategy {name} is not registered in ParserRegistry.")
        return cls._strategies[name]

    @classmethod
    def has(cls, name: str) -> bool:
        return name in cls._strategies


# Auto-register standard strategies
ParserRegistry.register("TextParserStrategy", TextParserStrategy())
ParserRegistry.register("PDFParserStrategy", PDFParserStrategy())
ParserRegistry.register("OCRParserStrategy", OCRParserStrategy())
ParserRegistry.register("OfficeParserStrategy", OfficeParserStrategy())
