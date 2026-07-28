import re
import logging
from typing import List, Dict, Any, Optional
from .tuner import PresentationTuner
from .builder import PresentationBuilder

logger = logging.getLogger("IncrementalBlockStreamer")

class IncrementalBlockStreamer:
    """
    CPU-Native Incremental Streaming Block Engine.
    Executes in < 0.1ms per token. Buffers LLM tokens into semantic presentation blocks
    (headings, paragraphs, bullet lists, badge groups) and tunes them incrementally
    before streaming to the UI.
    
    This eliminates raw text flashes and layout shift on the frontend while preserving
    sub-50ms Time-To-First-Token (TTFT) streaming responsiveness.
    """

    def __init__(self, active_topic: str = "Knowledge Overview", should_render_header: bool = True):
        self.buffer = ""
        self.block_index = 0
        self.active_topic = active_topic
        self.should_render_header = should_render_header

    def feed_token(self, token: str) -> List[Dict[str, Any]]:
        """
        Feeds an incoming token into the streaming buffer.
        Returns a list of event dictionaries:
        - {"type": "block_commit", "id": int, "content": str} when a boundary block completes and is tuned
        - {"type": "block_delta", "id": int, "delta": str} for live active block token streaming
        """
        events = []
        if not token:
            return events

        self.buffer += token

        # Check for block completion boundaries:
        # Boundary 1: Double newlines (\n\n) - Paragraph / section break
        # Boundary 2: Header glued to previous text or on new line (e.g. 'text### Heading' or '\n### Heading')
        # Boundary 3: List section break (blank line after bullet items)
        
        # Regex matches block boundaries in buffer
        pattern = r'(\n\s*\n|(?<=[^\n])#+|(?:^|\n)#+\s+)'
        match = re.search(pattern, self.buffer)

        # We need to ensure we don't prematurely slice a header while hashes are still arriving (e.g., '##' before '#' arrives for '###')
        if match and len(self.buffer) > match.end() + 2:
            split_idx = match.start()

            # Extract the completed block before the boundary
            completed_raw = self.buffer[:split_idx].strip()
            
            # If we extracted meaningful content
            if completed_raw:
                self.block_index += 1
                
                # Apply CPU-native presentation tuning to the completed block
                tuned_content = PresentationBuilder.transform_to_presentation(
                    completed_raw,
                    active_topic=self.active_topic,
                    should_render_header=self.should_render_header if self.block_index == 1 else True
                )
                
                events.append({
                    "type": "block_commit",
                    "id": self.block_index,
                    "content": tuned_content
                })

            # Advance buffer past the completed content
            self.buffer = self.buffer[split_idx:].lstrip("\n")

        # Emit block delta for active token rendering
        events.append({
            "type": "block_delta",
            "id": self.block_index + 1,
            "delta": token
        })

        return events

    def flush(self) -> List[Dict[str, Any]]:
        """
        Flushes any remaining text in the buffer when the stream completes.
        """
        events = []
        remaining = self.buffer.strip()
        if remaining:
            self.block_index += 1
            tuned_content = PresentationBuilder.transform_to_presentation(
                remaining,
                active_topic=self.active_topic,
                should_render_header=self.should_render_header if self.block_index == 1 else True
            )
            events.append({
                "type": "block_commit",
                "id": self.block_index,
                "content": tuned_content
            })
            self.buffer = ""
        return events
