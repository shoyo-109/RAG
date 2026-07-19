import logging
from typing import List, Tuple
try:
    from .models import PipelineContext, CapabilityState
    from .capability_registry import CapabilityRegistry
    from .policy import PolicyEngine
    from .registry import ParserRegistry
    from .parsers.base_parser import ParseResult, BaseParserStrategy
except (ImportError, ValueError):
    from models import PipelineContext, CapabilityState
    from capability_registry import CapabilityRegistry
    from policy import PolicyEngine
    from registry import ParserRegistry
    from parsers.base_parser import ParseResult, BaseParserStrategy

logger = logging.getLogger("StrategyFactory")


class StrategyFactory:
    """
    Factory implementing 3-Level Dynamic Fallbacks:
    - Level 1: Dependency Fallback (Skip UNAVAILABLE parsers)
    - Level 2: Runtime Fallback (Catch exceptions & try next candidate)
    - Level 3: Quality Fallback (If LOW_CONFIDENCE, try next candidate)
    """

    @classmethod
    def execute_with_fallbacks(cls, context: PipelineContext) -> Tuple[ParseResult, List[dict]]:
        capability_registry = CapabilityRegistry()
        candidates = PolicyEngine.get_candidate_strategies(context)
        fallback_history: List[dict] = []

        best_result: ParseResult = ParseResult(
            elements=[],
            confidence=0.0,
            status=CapabilityState.FAILED,
            error_message="No parser candidates succeeded"
        )

        for strategy_name in candidates:
            if not ParserRegistry.has(strategy_name):
                logger.warning(f"Strategy {strategy_name} requested by policy is not registered.")
                continue

            strategy: BaseParserStrategy = ParserRegistry.get(strategy_name)

            # Level 1: Dependency Check
            # If strategy is OCRParserStrategy, check if tesseract or unstructured core is available
            if strategy_name == "OCRParserStrategy":
                if not (capability_registry.is_available("ocr", "tesseract") or capability_registry.is_available("unstructured", "core")):
                    logger.info("Level 1 Dependency Fallback: Skipping OCRParserStrategy (OCR capability UNAVAILABLE).")
                    fallback_history.append({
                        "strategy": strategy_name,
                        "status": CapabilityState.UNAVAILABLE.value,
                        "reason": "System binaries / OCR packages missing"
                    })
                    continue

            logger.info(f"Executing parser strategy: {strategy_name}...")
            context.selected_parser = strategy_name

            # Level 2: Runtime Fallback Execution
            try:
                result = strategy.parse(context.file_path, context)
            except Exception as e:
                logger.error(f"Level 2 Runtime Fallback triggered for {strategy_name}: {e}")
                fallback_history.append({
                    "strategy": strategy_name,
                    "status": CapabilityState.FAILED.value,
                    "reason": f"Runtime exception: {str(e)}"
                })
                continue

            # Check Execution Status
            if result.status == CapabilityState.FAILED:
                logger.warning(f"Strategy {strategy_name} returned FAILED: {result.error_message}")
                fallback_history.append({
                    "strategy": strategy_name,
                    "status": CapabilityState.FAILED.value,
                    "reason": result.error_message or "Execution failed"
                })
                continue

            # Level 3: Quality Check
            if result.status == CapabilityState.LOW_CONFIDENCE or result.confidence < 0.50:
                logger.warning(f"Level 3 Quality Fallback triggered for {strategy_name} (Confidence: {result.confidence:.2f}). Trying next candidate...")
                fallback_history.append({
                    "strategy": strategy_name,
                    "status": CapabilityState.LOW_CONFIDENCE.value,
                    "confidence": result.confidence,
                    "reason": result.error_message or "Low quality text extraction"
                })
                # Keep partial result as backup in case later candidates also fail
                if result.confidence > best_result.confidence:
                    best_result = result
                continue

            # Success!
            logger.info(f"Strategy {strategy_name} succeeded with confidence {result.confidence:.2f}.")
            fallback_history.append({
                "strategy": strategy_name,
                "status": CapabilityState.SUCCESS.value,
                "confidence": result.confidence
            })
            return result, fallback_history

        # If all candidates failed or returned low confidence, return the best partial result
        logger.warning(f"All candidates attempted for {context.filename}. Returning best result (Confidence: {best_result.confidence:.2f}).")
        return best_result, fallback_history
