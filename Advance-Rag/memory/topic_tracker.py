import re
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("TopicTracker")

class TopicTracker:
    """
    Layer 4: Topic & Heading Shift Tracker
    Analyzes current user query against working memory to detect whether the user
    is continuing the same topic or shifting to a new topic.
    """
    def __init__(self):
        self.current_topic: Optional[str] = None
        self.topic_history: List[str] = []

    def analyze_turn(self, query: str, recent_topics: Optional[List[str]] = None) -> Dict[str, Any]:
        q_lower = query.lower()
        
        # Follow-up indicators (pronouns, continuations)
        continuation_patterns = [
            r'\bhe\b', r'\bhis\b', r'\bhim\b', r'\bshe\b', r'\bher\b', r'\bit\b', r'\bthey\b',
            r'\bthem\b', r'\bwhat about\b', r'\balso\b', r'\band\b', r'\bmore details\b',
            r'\bexplain that\b', r'\bthis\b', r'\bthese\b', r'\bthose\b'
        ]

        is_continuation = any(re.search(pat, q_lower) for pat in continuation_patterns)
        
        # Heading topics detection keywords
        topic_keywords = {
            "Work Experience": ["experience", "job", "work", "role", "company", "intern", "developer", "analyst"],
            "Education": ["education", "degree", "college", "university", "cgpa", "school", "gpa", "b.tech"],
            "Technical Stack": ["skills", "languages", "programming", "frameworks", "tools", "stack", "python", "java"],
            "Projects": ["projects", "project", "rag", "jarvis", "built", "engineered", "application"],
            "Certifications & Awards": ["certifications", "cert", "achievements", "hackathon", "awards", "ctf"]
        }

        detected_topic = None
        for topic, keywords in topic_keywords.items():
            if any(kw in q_lower for kw in keywords):
                detected_topic = topic
                break

        if detected_topic is None:
            detected_topic = "General Inquiry"

        # Determine shift status
        if is_continuation and self.current_topic:
            is_topic_shift = False
            active_topic = self.current_topic
        elif detected_topic != self.current_topic:
            is_topic_shift = True
            active_topic = detected_topic
            self.current_topic = detected_topic
            if detected_topic not in self.topic_history:
                self.topic_history.append(detected_topic)
        else:
            is_topic_shift = False
            active_topic = self.current_topic or detected_topic

        logger.info(f"TopicTracker: Active='{active_topic}', Shift={is_topic_shift}, Continuation={is_continuation}")

        return {
            "active_topic": active_topic,
            "is_topic_shift": is_topic_shift,
            "is_continuation": is_continuation,
            "should_render_top_header": is_topic_shift
        }

    def clear(self):
        self.current_topic = None
        self.topic_history.clear()
