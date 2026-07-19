# Capability Registry & Dynamic Fallback Design

## Production-Grade RAG Ingestion Implementation Guide

> Purpose: Implement a resilient ingestion pipeline that **never
> crashes** due to missing dependencies and always attempts the best
> available parsing strategy.

------------------------------------------------------------------------

# Core Principle

The pipeline should **not** ask:

> "Is Tesseract installed?"

Instead it should ask:

> "What capabilities are available on this machine, and what is the best
> strategy I can execute right now?"

------------------------------------------------------------------------

# Capability Registry

The application performs a one-time startup scan.

``` text
Application Starts
        │
        ▼
Capability Detector
        │
        ├── Tesseract Available?
        ├── Poppler Available?
        ├── PaddleOCR Available?
        ├── Vision Model Available?
        ├── Docling Available?
        ├── PyMuPDF Available?
        └── GPU/CUDA Available?
```

Example:

``` python
CapabilityRegistry = {
    "ocr": {
        "tesseract": "AVAILABLE",
        "paddleocr": "UNAVAILABLE"
    },
    "pdf_render": {
        "poppler": "AVAILABLE",
        "pymupdf": "AVAILABLE"
    },
    "layout": {
        "docling": "AVAILABLE"
    }
}
```

------------------------------------------------------------------------

# Capability States

-   AVAILABLE
-   UNAVAILABLE
-   RUNNING
-   SUCCESS
-   FAILED
-   LOW_CONFIDENCE

These states drive routing decisions.

------------------------------------------------------------------------

# Dynamic Routing Flow

``` text
Upload
    │
    ▼
Validation
    │
    ▼
Classification
    │
    ▼
Capability Registry
    │
    ▼
Policy Engine
    │
    ▼
Factory
    │
    ▼
Registry
    │
    ▼
Parser Strategy
    │
    ▼
Confidence Evaluation
    │
    ├── SUCCESS ───────────────► Normalization
    │
    ├── FAILED ────────────────► Next Strategy
    │
    ├── UNAVAILABLE ───────────► Next Strategy
    │
    └── LOW_CONFIDENCE ────────► Better Strategy
```

------------------------------------------------------------------------

# Three Levels of Fallback

## 1. Dependency Fallback

Example:

``` text
Scanned PDF
      │
      ▼
Need OCR
      │
      ▼
Tesseract Installed?

NO
      │
      ▼
Skip OCR
      │
      ▼
Try Best Available Parser
```

Never throw an exception simply because a dependency is missing.

------------------------------------------------------------------------

## 2. Runtime Fallback

Even if installed:

``` text
OCR
 │
 ▼
Runtime Error
 │
 ▼
FAILED
 │
 ▼
Next OCR Engine
```

Examples: - timeout - corrupt file - parser crash

------------------------------------------------------------------------

## 3. Quality Fallback

Even when execution succeeds:

``` text
OCR
 │
 ▼
Confidence = 42%
 │
 ▼
LOW_CONFIDENCE
 │
 ▼
Try Better OCR
```

Installation ≠ Success ≠ Good Quality.

------------------------------------------------------------------------

# Decision Tree

``` text
Parser Needed
      │
      ▼
Capability Available?
      │
  ┌───┴────┐
 YES      NO
 │          │
 ▼          ▼
Run      Next Strategy
 │
 ▼
Execution Success?
 │
 ┌────┴─────┐
YES        NO
 │          │
 ▼          ▼
Confidence  FAILED
Check         │
 │            ▼
 ├────Good────► Continue
 │
 └──Low──────► Better Strategy
```

------------------------------------------------------------------------

# Logging Requirements (LangSmith)

Log:

-   selected parser
-   parser version
-   fallback reason
-   dependency status
-   execution duration
-   confidence score
-   retry count
-   OCR confidence
-   extraction quality
-   final parser used

------------------------------------------------------------------------

# Implementation Rules

1.  Never crash because a dependency is missing.
2.  Never assume installation means execution success.
3.  Never assume execution success means acceptable quality.
4.  Always preserve partial results whenever possible.
5.  Always emit structured telemetry.
6.  Always record fallback reasons.
7.  Normalize output regardless of parser.
8.  QA validates output before chunking.

------------------------------------------------------------------------

# Suggested Interfaces

``` python
class CapabilityRegistry:
    def refresh(self): ...
    def is_available(self, capability): ...
    def get_best(self, capability): ...
```

``` python
class ParserStrategy:
    def parse(self, context): ...
```

``` python
class StrategyFactory:
    def get_strategy(self, context): ...
```

``` python
class PolicyEngine:
    def evaluate(self, context): ...
```

``` python
class PipelineContext:
    ...
```

------------------------------------------------------------------------

# End-to-End Pipeline

``` text
Upload
    │
Validation
    │
Classification
    │
Capability Registry
    │
Policy Engine
    │
Factory
    │
Registry
    │
Parser Strategy
    │
Confidence Evaluation
    │
Retry / Fallback
    │
Canonical Normalization
    │
Metadata + Provenance
    │
Quality Assurance
    │
Chunking
    │
Embedding
    │
Vector Index
```
