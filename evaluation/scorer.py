"""Deterministic, auditable metrics for RAG evaluation runs.

The scorer deliberately does not use an LLM judge: every point can be traced to
an answer concept or an evidence concept declared in the benchmark JSON.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _normalise(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value.lower())).strip()


def _coverage(text: str, concept_groups: Iterable[Iterable[str]]) -> Dict[str, Any]:
    normalised = _normalise(text)
    groups = [list(group) for group in concept_groups]
    matched = [any(_normalise(alias) in normalised for alias in group) for group in groups]
    total = len(matched)
    return {
        "matched": sum(matched),
        "total": total,
        "score": (sum(matched) / total) if total else 1.0,
        "missing": [groups[index] for index, value in enumerate(matched) if not value],
    }


def load_benchmark(path: str | Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def score_run(payload: Dict[str, Any], benchmark: Dict[str, Any]) -> Dict[str, Any]:
    """Score a harness payload and return serialisable retrieval/generation metrics.

    Overall score weights evidence retrieval 40%, answer concept coverage 50%,
    and the pipeline grounding verdict 10%.  A brevity guardrail is reported
    separately rather than silently rewarding longer answers.
    """
    cases = {case["id"]: case for case in benchmark["cases"]}
    per_question: List[Dict[str, Any]] = []

    for result in payload.get("results", []):
        case = cases.get(result.get("question_id"))
        if not case:
            continue
        retrieval = result.get("retrieval", {})
        selected = retrieval.get("rewritten_results") if result.get("query_rewriting", {}).get("triggered") else retrieval.get("initial_results", [])
        retrieved_text = "\n".join(chunk.get("text", "") for chunk in selected)
        evidence = _coverage(retrieved_text, case["evidence_concepts"])
        answer = _coverage(result.get("final_answer", ""), case["answer_concepts"])
        grounded = 1.0 if result.get("hallucination_check", {}).get("is_grounded") is True else 0.0
        word_count = len(_normalise(result.get("final_answer", "")).split())
        concise = 1.0 if word_count <= 180 else max(0.0, 1 - ((word_count - 180) / 360))
        overall = (0.40 * evidence["score"]) + (0.50 * answer["score"]) + (0.10 * grounded)
        per_question.append({
            "question_id": case["id"],
            "retrieval_evidence": evidence,
            "generation_concept_coverage": answer,
            "grounded": bool(grounded),
            "answer_word_count": word_count,
            "conciseness": concise,
            "overall": overall,
        })

    def average(values: Iterable[float]) -> float:
        values = list(values)
        return sum(values) / len(values) if values else 0.0

    return {
        "benchmark": benchmark.get("name", "unnamed"),
        "scoring_version": "1.0",
        "question_count": len(per_question),
        "retrieval_evidence_recall": average(item["retrieval_evidence"]["score"] for item in per_question),
        "generation_concept_coverage": average(item["generation_concept_coverage"]["score"] for item in per_question),
        "grounded_rate": average(float(item["grounded"]) for item in per_question),
        "conciseness": average(item["conciseness"] for item in per_question),
        "overall_score": average(item["overall"] for item in per_question),
        "per_question": per_question,
    }
