import re
import logging
from typing import Dict, Any, List, Optional
from .schemas import PriorityLevel, BlockType, PresentationBlock, CategorizedGroup
from .tuner import PresentationTuner

logger = logging.getLogger("PresentationBuilder")

class PresentationBuilder:
    """
    Presentation Transformer.
    Transforms raw LLM / RAG outputs into Progressive Disclosure layouts:
    - Layer 1: Executive 1-Line Summary Banner
    - Layer 2: Key Takeaways (3-5 Bullet Points)
    - Layer 3: Categorized Data & Experience Breakdown
    - Layer 4: Concise Closing Synthesis
    """

    @classmethod
    def transform_to_presentation(cls, raw_text: str, active_topic: str = "Knowledge Overview", should_render_header: bool = True) -> str:
        """
        Transforms raw text into styled, structured Markdown using CPU-native algorithms (< 0.5ms).
        Enforces Progressive Disclosure, Density Rhythm, and Header Control.
        """
        if not raw_text:
            return ""

        # Step 1: Fix broken concatenated headers
        cleaned_text = PresentationTuner.clean_concatenated_headers(raw_text)

        # Step 2: Auto-group skill/tool lists into categorized sections
        grouped_text = PresentationTuner.format_technical_lists_as_badges(cleaned_text)

        # Step 3: Enforce visual density rhythm (Paragraph -> Badges -> Bullets -> Summary)
        rhythmic_text = PresentationTuner.enforce_density_rhythm(grouped_text)

        # Step 4: Header control based on TopicTracker signal
        if not should_render_header:
            # Strip top-level H3 header if continuing same topic to maintain smooth chat flow
            rhythmic_text = re.sub(r'^###\s+[^\n]+\n+', '', rhythmic_text.strip())

        return rhythmic_text.strip()

