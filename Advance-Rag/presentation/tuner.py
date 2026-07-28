import re
import logging
from typing import List, Dict, Any, Tuple
from .schemas import PriorityLevel, CategorizedGroup

logger = logging.getLogger("PresentationTuner")

class PresentationTuner:
    """
    CPU-Native Presentation Tuner.
    Executes in < 0.5ms on CPU. Enforces all 9 presentation rules without prompt bloat:
    - Normalizes inline concatenated headers (e.g. 'text### Heading' -> '\n\n### Heading\n\n')
    - Groups raw lists into semantic badge categories
    - Enforces Density & Rhythm rules (Paragraph -> Categorized Badges -> Bullets -> Summary)
    """

    # Category taxonomy map for automatic Data Grouping (Rule 6)
    CATEGORY_TAXONOMY = {
        "Languages": ["python", "java", "kotlin", "dart", "c++", "javascript", "typescript", "html", "css", "sql"],
        "Web & Backend": ["react.js", "react", "express.js", "express", "node.js", "node", "rest apis", "flutter", "android sdk"],
        "AI & RAG Stack": ["llm", "llms", "rag", "langchain", "chromadb", "langsmith", "nvidia nim", "nemotron", "gpt-4o-mini", "bm25", "hnsw"],
        "Tools & Cloud": ["git", "github", "intellij idea", "vscode", "antigravity", "supabase", "vercel", "render", "docker"]
    }

    @classmethod
    def clean_concatenated_headers(cls, text: str) -> str:
        """
        Fixes LLM formatting artifacts where headers get glued to previous words.
        e.g. 'ChromaDB.### Education#### Bachelor' -> 'ChromaDB.\n\n### Education\n\n#### Bachelor'
        e.g. 'Professional SummaryBhubhurv' -> '### Professional Summary\n\nBhubhurv'
        """
        if not text:
            return ""

        processed = text.strip()

        # Step 1: Separate concatenated headers like '### Education#### Bachelor' -> '### Education\n\n#### Bachelor'
        processed = re.sub(r'(#+\s*[^#\n]+?)(#+)', r'\1\n\n\2', processed)

        # Step 2: Ensure newline before ### or #### or ## if preceded by text
        processed = re.sub(r'([^\n])\s*(#+)', r'\1\n\n\2', processed)
        
        # Step 3: Ensure single space between header hashes and title text (e.g. '###Title' -> '### Title')
        processed = re.sub(r'(#+)\s*([A-Za-z0-9])', r'\1 \2', processed)

        # Step 4: Fix glued titles at start like 'Professional SummaryBhubhurv' -> '### Professional Summary\n\nBhubhurv'
        processed = re.sub(
            r'^(Professional Summary|Executive Summary|Summary|Education|Experience|Technical Skills|Skills|Projects|Certifications)([A-Z])',
            r'### \1\n\n\2',
            processed
        )

        # Step 5: Fix glued titles in body like 'textProfessional SummaryBhubhurv' -> 'text\n\n### Professional Summary\n\nBhubhurv'
        processed = re.sub(
            r'([a-z0-9\.\:\)])(Professional Summary|Executive Summary|Summary|Education|Experience|Technical Skills|Skills|Projects|Certifications)([A-Z])',
            r'\1\n\n### \2\n\n\3',
            processed
        )

        # Step 6: Ensure double newlines around complete header lines
        processed = re.sub(r'([^\n])\n*(#+\s+[^\n]+)\n*([^\n])', r'\1\n\n\2\n\n\3', processed)

        # Step 7: Clean excessive newlines (> 2)
        processed = re.sub(r'\n{3,}', '\n\n', processed)

        return processed.strip()


    @classmethod
    def group_raw_list_items(cls, items: List[str]) -> List[CategorizedGroup]:
        """
        Rule 6: Data Grouping.
        Clusters a raw list of technical skills or tools into meaningful sub-headers.
        """
        groups: Dict[str, List[str]] = {}

        for item in items:
            clean_item = item.strip().strip(",.*-")
            if not clean_item:
                continue

            matched = False
            item_lower = clean_item.lower()

            for cat_name, kw_list in cls.CATEGORY_TAXONOMY.items():
                if any(kw in item_lower for kw in kw_list):
                    groups.setdefault(cat_name, []).append(clean_item)
                    matched = True
                    break

            if not matched:
                groups.setdefault("General Details", []).append(clean_item)

        return [CategorizedGroup(category_name=k, items=v) for k, v in groups.items()]

    @classmethod
    def format_technical_lists_as_badges(cls, text: str) -> str:
        """
        Rule 6 & 7: Data Grouping & Badges.
        Scans for technical lists (e.g., raw lists of languages or tools) and formats them into categorized badge groups.
        """
        if not text:
            return ""

        # Find raw comma-separated lists under 'Skills' or 'Tools' headers
        lines = text.split("\n")
        new_lines = []
        in_skills_section = False

        for line in lines:
            trimmed = line.strip()
            if any(header in trimmed.lower() for header in ["### skills", "### technical skills", "### tools", "### tech stack"]):
                in_skills_section = True
                new_lines.append(line)
                continue

            if trimmed.startswith("### ") or trimmed.startswith("## "):
                in_skills_section = False
                new_lines.append(line)
                continue

            if in_skills_section and (trimmed.startswith("- ") or "," in trimmed or ":" in trimmed):
                raw_items = [i.strip() for i in re.split(r'[,:\-*]', trimmed) if i.strip()]
                if len(raw_items) >= 3:
                    grouped = cls.group_raw_list_items(raw_items)
                    badge_str_list = []
                    for g in grouped:
                        pills = " ".join([f"`{item}`" for item in g.items])
                        badge_str_list.append(f"- **{g.category_name}**: {pills}")
                    new_lines.extend(badge_str_list)
                    continue

            new_lines.append(line)

        return "\n".join(new_lines)

    @classmethod
    def score_section_priority(cls, section_title: str, content: str) -> PriorityLevel:
        """
        Rule 1 & 9: Priority Scoring.
        Assigns HIGH, MEDIUM, or LOW priority based on section position and content density.
        """
        st_lower = section_title.lower()
        if any(kw in st_lower for kw in ["summary", "overview", "experience", "core"]):
            return PriorityLevel.HIGH
        elif any(kw in st_lower for kw in ["education", "skills", "projects"]):
            return PriorityLevel.MEDIUM
        return PriorityLevel.LOW

    @classmethod
    def enforce_density_rhythm(cls, text: str) -> str:
        """
        Rule 8: Visual Density Rules.
        Ensures a rhythmic pacing: Paragraph -> Grouped Pills/Bullets -> Callout / Summary.
        Prevents > 5 consecutive bullet points without spacing.
        """
        lines = text.split("\n")
        formatted_lines = []
        consecutive_bullets = 0

        for line in lines:
            trimmed = line.strip()
            if trimmed.startswith("- ") or trimmed.startswith("* "):
                consecutive_bullets += 1
                if consecutive_bullets == 6:
                    # Break up bullet density with visual spacing
                    formatted_lines.append("")
                    consecutive_bullets = 1
                formatted_lines.append(line)
            else:
                consecutive_bullets = 0
                formatted_lines.append(line)

        return "\n".join(formatted_lines)

