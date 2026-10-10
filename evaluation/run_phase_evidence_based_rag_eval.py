"""
Evidence-Based RAG Evaluation Local Testing Phase Script (Audited & Fully Corrected)
=========================================================================================
Logical Stages in Local Evaluation Harness:
1. Candidate Recall Diagnostics (Multi-depth N in {10, 20, 30, 40, 50}, RRF fusion, mapping validation, strict denominators)
2. Context Reconstruction (Target-Blind Runtime Context vs Text-Verified Post-Hoc Benchmark Evaluator)
3. Evidence-Aware Ranking (Fair Comparison of Cross-Encoder Baseline vs Target-Blind Diversity/MMR Ranking via compare_rankings)
4. Sufficiency Gate (Target-Blind Runtime Gate + Ground Truth Evaluator)
5. Recovery Retrieval Controller (Stateful depth-independent (question_id, depth) retry guard & Gate Reassessment)
6. Generation or Abstention (Target-Blind generation using runtime_gate; Structured Statuses: SUCCESS, ABSTAINED, CLIENT_UNAVAILABLE, API_ERROR)
7. Citation and Claim Verification (Lexical support heuristic, citation resolution, nullified non-SUCCESS metrics)

Authoritative Benchmark: evaluation/golden_set.json (10 questions, Q01-Q10)
Corpus: The Comedy of Errors document.pdf (203 canonical chunks: C0-C202)
"""

from pygments.lexer import combined
import os
import sys
import json
import math
import copy
import re
import importlib.util
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set

import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
advance_rag_dir = os.path.join(root_dir, "Advance-Rag")
doc_loader_dir = os.path.join(advance_rag_dir, "Doc-Loader")

sys.path.insert(0, doc_loader_dir)
sys.path.insert(0, advance_rag_dir)
sys.path.insert(0, root_dir)

# Load IngestionPipeline & AdvancedRAGPipeline
doc_loader_pipeline_path = os.path.join(doc_loader_dir, "pipeline.py")
spec = importlib.util.spec_from_file_location("doc_loader_pipeline", doc_loader_pipeline_path)
doc_loader_module = importlib.util.module_from_spec(spec)
sys.modules["doc_loader_pipeline"] = doc_loader_module
spec.loader.exec_module(doc_loader_module)
IngestionPipeline = doc_loader_module.IngestionPipeline

from advance_rag import AdvancedRAGPipeline
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI


# ==============================================================================
# INTEGRITY ASSERTIONS & HARNESS FAIL-FAST CHECKS
# ==============================================================================

def validate_benchmark_integrity(
    golden_set: Dict[str, Any],
    corpus_chunks: List[Dict[str, Any]],
    candidate_depths: List[int],
    context_budget: int,
    expected_corpus_size: Optional[int] = 203
) -> bool:
    """
    Fail-fast validation for benchmark dataset, corpus chunk contiguity, non-empty text,
    duplicate evidence IDs, and configuration limits.
    """
    
    questions = golden_set.get("questions", [])

    if not isinstance(questions, list) or not questions:
        raise ValueError("Golden set must contain a non-empty questions list.")

    # Enforce exact question IDs for the production benchmark.
    if expected_corpus_size == 203:
        expected_qids = {f"Q{i:02d}" for i in range(1, 11)}
        actual_qids = {
            q.get("question_id")
            for q in questions
            if isinstance(q, dict)
    }

        if len(questions) != 10 or actual_qids != expected_qids:
            raise ValueError(
                f"Production benchmark must contain exactly Q01-Q10. "
                f"Found IDs: {sorted(str(qid) for qid in actual_qids)}"
            )

    q_ids = []

    for index, q in enumerate(questions):
        if not isinstance(q, dict):
            raise ValueError(f"Question at index {index} must be a dictionary.")

        required_fields = [
            "question_id",
            "question",
            "gold_evidence",
            "evidence_policy",
            "gold_answer",
            "required_facts",
        ]

        for field in required_fields:
            if field not in q:
                raise ValueError(
                    f"Question at index {index} is missing required field '{field}'."
                )

        qid = q["question_id"]

        if not isinstance(qid, str) or not qid.strip():
            raise ValueError(f"Question at index {index} has an invalid question_id.")

        q_ids.append(qid)

        if not isinstance(q["question"], str) or not q["question"].strip():
            raise ValueError(f"{qid}: question must be a non-empty string.")

        if not isinstance(q["gold_evidence"], list):
            raise ValueError(f"{qid}: gold_evidence must be a list.")

        if not isinstance(q["evidence_policy"], dict):
            raise ValueError(f"{qid}: evidence_policy must be a dictionary.")

        if not isinstance(q["gold_answer"], str):
            raise ValueError(f"{qid}: gold_answer must be a string.")

        if not isinstance(q["required_facts"], list):
            raise ValueError(f"{qid}: required_facts must be a list.")

        evidence_ids = []

        for ev_index, evidence in enumerate(q["gold_evidence"]):
            if not isinstance(evidence, dict):
                raise ValueError(
                    f"{qid}: gold_evidence[{ev_index}] must be a dictionary."
                )

            evidence_id = evidence.get("evidence_id")

            if not isinstance(evidence_id, str) or not evidence_id.strip():
                raise ValueError(
                    f"{qid}: gold_evidence[{ev_index}] has an invalid evidence_id."
                )

            evidence_ids.append(evidence_id)

            if "quote" in evidence and not isinstance(evidence["quote"], str):
                raise ValueError(
                    f"{qid}/{evidence_id}: quote must be a string when supplied."
                )

        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError(f"{qid}: duplicate evidence IDs found.")

    if len(q_ids) != len(set(q_ids)):
        raise ValueError("Golden set contains duplicate question IDs.")

    # if not corpus_chunks:
    #     raise ValueError("Ingested corpus contains zero chunks.")
    
    if not isinstance(corpus_chunks, list) or not corpus_chunks:
        raise ValueError("Ingested corpus must be a non-empty list.")

    if (
        expected_corpus_size is not None
        and len(corpus_chunks) != expected_corpus_size
    ):
        raise ValueError(
            f"Expected corpus size {expected_corpus_size}, "
            f"found {len(corpus_chunks)}."
        )

    expected_cids = {f"C{i}" for i in range(len(corpus_chunks))}
    actual_cid_list = []

    for index, chunk in enumerate(corpus_chunks):
        if not isinstance(chunk, dict):
            raise ValueError(f"Corpus chunk at index {index} must be a dictionary.")

        cid = chunk.get("chunk_id")

        if not isinstance(cid, str) or not cid.strip():
            raise ValueError(f"Corpus chunk at index {index} has an invalid chunk_id.")

        actual_cid_list.append(cid)

        chunk_text = chunk.get("text")

        if not isinstance(chunk_text, str) or not chunk_text.strip():
            raise ValueError(f"Chunk {cid} contains empty or invalid text.")

    if len(actual_cid_list) != len(set(actual_cid_list)):
        raise ValueError("Corpus contains duplicate chunk IDs.")

    actual_cids = set(actual_cid_list)

    if actual_cids != expected_cids:
        raise ValueError(
            "Corpus chunk IDs are non-contiguous or invalid. "
            f"Discrepancy: {expected_cids ^ actual_cids}"
        )

    if candidate_depths != [10, 20, 30, 40, 50]:
        raise ValueError(
            f"Expected candidate_depths [10, 20, 30, 40, 50], "
            f"got {candidate_depths}."
        )

    if context_budget != 15:
        raise ValueError(f"Expected context_budget 15, got {context_budget}.")


    return True


# ==============================================================================
# 1. STRICT EVIDENCE MAPPING VALIDATION & DENOMINATOR POLICY
# ==============================================================================

EvidenceKey = Tuple[str, str]  # (question_id, evidence_id)


def get_evidence_mapping(
    mapping: Dict[EvidenceKey, List[str]],
    question_id: str,
    evidence_id: str,
) -> List[str]:
    """Resolves evidence chunk mappings using composite identity tuple (question_id, evidence_id)."""
    return mapping.get((question_id, evidence_id), [])


def validate_evidence_mapping(
    mapped_chunk_ids: List[str],
    canonical_chunk_ids: Set[str],
) -> Dict[str, Any]:
    """
    Validates a gold evidence unit mapping against the canonical corpus.
    Distinguishes VALID, INVALID_MAPPING, and UNMAPPED.
    """
    mapped = set(mapped_chunk_ids)
    invalid = mapped - canonical_chunk_ids
    valid = mapped & canonical_chunk_ids

    if invalid:
        status = "INVALID_MAPPING"
    elif not valid:
        status = "UNMAPPED"
    else:
        status = "VALID"

    return {
        "status": status,
        "valid_chunk_ids": sorted(valid),
        "invalid_chunk_ids": sorted(invalid),
    }


def classify_evidence_coverage(
    required_chunk_ids: Set[str],
    retrieved_chunk_ids: Set[str],
) -> Dict[str, Any]:
    """
    Classifies evidence coverage cleanly into COMPLETE, PARTIAL, MISSING, or UNMAPPED.
    - COMPLETE: required_chunk_ids.issubset(retrieved_chunk_ids)
    - PARTIAL: required_chunk_ids & retrieved_chunk_ids and not COMPLETE
    - MISSING: required_chunk_ids present but none retrieved
    - UNMAPPED: required_chunk_ids is empty or invalid
    """
    if not required_chunk_ids:
        return {
            "status": "UNMAPPED",
            "complete": False,
            "partial": False,
            "matched_chunk_ids": [],
        }

    matched = required_chunk_ids & retrieved_chunk_ids
    complete = required_chunk_ids.issubset(retrieved_chunk_ids)

    return {
        "status": (
            "COMPLETE" if complete
            else "PARTIAL" if matched
            else "MISSING"
        ),
        "complete": complete,
        "partial": bool(matched) and not complete,
        "matched_chunk_ids": sorted(matched),
    }


class GoldEvidenceMapper:
    """Maps logical gold evidence units in golden_set.json to corpus chunk IDs (C0-C202)."""

    def __init__(self, corpus_chunks: List[Dict[str, Any]]):
        self.corpus_chunks = corpus_chunks
        self.canonical_cids = set(c["chunk_id"] for c in corpus_chunks)

    def map_evidence_unit(self, question_id: str, gold_ev: Dict[str, Any]) -> Dict[str, Any]:
        ev_id = gold_ev["evidence_id"]
        quote = gold_ev.get("quote", "")
        quote_norm = re.sub(r"\s+", " ", quote.lower()).strip()

        matched_cids = []
        match_type = "UNMAPPED"

        if not quote_norm:
            return {
                "question_id": question_id,
                "evidence_id": ev_id,
                "quote": quote,
                "matched_chunk_ids": [],
                "match_type": "UNMAPPED_ANNOTATION_ERROR",
                "validation": validate_evidence_mapping([], self.canonical_cids)
            }

        # Substring match
        for chunk in self.corpus_chunks:
            text_norm = re.sub(r"\s+", " ", chunk["text"].lower()).strip()
            if quote_norm in text_norm:
                matched_cids.append(chunk["chunk_id"])
                match_type = "EXACT_OR_NORMALIZED_SUBSTRING"

        # Multi-clause span match if direct substring failed
        if not matched_cids:
            clauses = [c.strip() for c in re.split(r"[;,.]", quote_norm) if len(c.strip()) > 15]
            for chunk in self.corpus_chunks:
                text_norm = re.sub(r"\s+", " ", chunk["text"].lower()).strip()
                if clauses and any(clause in text_norm for clause in clauses):
                    matched_cids.append(chunk["chunk_id"])
                    match_type = "SPLIT_CLAUSE_SPAN_MATCH"

        val_res = validate_evidence_mapping(matched_cids, self.canonical_cids)
        return {
            "question_id": question_id,
            "evidence_id": ev_id,
            "quote": quote,
            "matched_chunk_ids": matched_cids,
            "match_type": match_type if val_res["status"] == "VALID" else "UNMAPPED_ANNOTATION_ERROR",
            "validation": val_res
        }


# ==============================================================================
# BM25 RESOLUTION & DEDUPLICATION HELPERS
# ==============================================================================

def resolve_bm25_document(
    doc: Any,
    canonical_chunks: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Resolves a BM25 LangChain document to a canonical corpus chunk (C0-C202).
    Avoids silent ambiguous mapping when multiple chunks share an element_id.
    """
    el_id = doc.metadata.get("element_id") if hasattr(doc, "metadata") else None
    content = doc.page_content.strip() if hasattr(doc, "page_content") else str(doc).strip()

    # 1. Exact match on element_id and text content
    exact_match = next((c for c in canonical_chunks if c.get("element_id") == el_id and c["text"].strip() == content), None)
    if exact_match:
        return {"status": "RESOLVED", "chunk": exact_match, "reason": "exact_element_and_text"}

    # 2. Unique element_id resolution
    el_matches = [c for c in canonical_chunks if c.get("element_id") == el_id]
    if len(el_matches) == 1:
        return {"status": "RESOLVED", "chunk": el_matches[0], "reason": "unique_element_id"}
    elif len(el_matches) > 1:
        return {"status": "AMBIGUOUS_METADATA_MAPPING", "chunk": None, "reason": f"element_id '{el_id}' matches {len(el_matches)} distinct chunks"}

    # 3. Fallback exact text match
    text_match = next((c for c in canonical_chunks if c["text"].strip() == content), None)
    if text_match:
        return {"status": "RESOLVED", "chunk": text_match, "reason": "exact_text_content_match"}

    return {"status": "UNRESOLVED", "chunk": None, "reason": "no_matching_canonical_chunk"}


def unique_ranked_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicates candidate list by canonical chunk_id while preserving original first rank."""
    seen = set()
    result = []
    for c in chunks:
        cid = c["chunk_id"]
        if cid not in seen:
            seen.add(cid)
            result.append(dict(c))
    return result


def compute_rrf_fusion(
    bm25_cands: List[Dict[str, Any]],
    dense_cands: List[Dict[str, Any]],
    rrf_k: int = 60,
    top_n: int = 20
) -> List[Dict[str, Any]]:
    """Calculates Reciprocal Rank Fusion (RRF) over deduplicated BM25 and Dense candidates."""
    dedup_bm25 = unique_ranked_chunks(bm25_cands)
    dedup_dense = unique_ranked_chunks(dense_cands)

    rrf_scores = {}
    chunk_map = {}
    bm25_ranks = {}
    dense_ranks = {}

    for rank, c in enumerate(dedup_bm25, 1):
        cid = c["chunk_id"]
        chunk_map[cid] = c
        bm25_ranks[cid] = rank
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank))

    for rank, c in enumerate(dedup_dense, 1):
        cid = c["chunk_id"]
        chunk_map[cid] = c
        dense_ranks[cid] = rank
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank))

    sorted_cids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_n]
    fused_cands = []
    for cid in sorted_cids:
        item = copy.deepcopy(chunk_map[cid])
        item["rrf_score"] = float(rrf_scores[cid])
        item["bm25_source_rank"] = bm25_ranks.get(cid)
        item["dense_source_rank"] = dense_ranks.get(cid)
        fused_cands.append(item)

    return fused_cands


# ==============================================================================
# PRODUCTION MULTI-DEPTH RETRIEVAL & RERANKING ORCHESTRATION ENTRY POINT
# ==============================================================================

def orchestrate_question_retrieval(
    question: str,
    candidate_depth: int,
    pipeline: Any,
    embedder: Any,
    cross_encoder: Any,
    corpus_chunks: List[Dict[str, Any]],
    doc_embs: Optional[np.ndarray] = None,
    context_budget: int = 15
) -> Dict[str, Any]:
    """
    Central production retrieval and ranking orchestration function.
    Target-blind runtime execution for a given question and candidate depth N.
    """
    bm25_cands = []
    if hasattr(pipeline, "bm25_retriever") and pipeline.bm25_retriever:
        pipeline.bm25_retriever.k = candidate_depth
        bm25_docs = pipeline.bm25_retriever.invoke(question)
        for doc in bm25_docs:
            res_bm25 = resolve_bm25_document(doc, corpus_chunks)
            if res_bm25["status"] == "RESOLVED" and res_bm25["chunk"] not in bm25_cands:
                bm25_cands.append(res_bm25["chunk"])

    # Dense retrieval at depth N
    q_emb = embedder.encode([question], normalize_embeddings=True)
    if isinstance(q_emb, np.ndarray):
        q_emb = q_emb.flatten()
    if doc_embs is None:
        doc_embs = embedder.encode([c["text"] for c in corpus_chunks], normalize_embeddings=True)
    dense_sims = np.dot(doc_embs, q_emb).flatten()
    dense_ranked_indices = np.argsort(-dense_sims)[:candidate_depth]
    dense_cands = [corpus_chunks[int(i)] for i in dense_ranked_indices]

    # Deduplicate & RRF Fusion at depth N
    union_cands = unique_ranked_chunks(bm25_cands + dense_cands)
    post_fusion_cands = compute_rrf_fusion(bm25_cands, dense_cands, rrf_k=60, top_n=candidate_depth)

    # Reranking (Baseline vs Diversity)
    stage3 = EvidenceAwareRankingStage()
    baseline_cands = stage3.rank_by_cross_encoder(question, post_fusion_cands, cross_encoder, top_k=context_budget)
    diversity_cands = stage3.rank_with_diversity(question, post_fusion_cands, cross_encoder, top_k=context_budget)

    return {
        "candidate_depth": candidate_depth,
        "requested_counts": {"bm25": candidate_depth, "dense": candidate_depth, "context_budget": context_budget},
        "actual_counts": {
            "bm25": len(bm25_cands),
            "dense": len(dense_cands),
            "union": len(union_cands),
            "fusion": len(post_fusion_cands),
            "baseline": len(baseline_cands),
            "diversity": len(diversity_cands)
        },
        "bm25_candidates": bm25_cands,
        "dense_candidates": dense_cands,
        "union_candidates": union_cands,
        "post_fusion_candidates": post_fusion_cands,
        "baseline_candidates": baseline_cands,
        "diversity_candidates": diversity_cands
    }


# ==============================================================================
# STAGE 1: CANDIDATE RECALL DIAGNOSTICS
# ==============================================================================

class CandidateRecallDiagnosticsStage:
    """Stage 1: Candidate Recall Diagnostics across multi-depth N in {10, 20, 30, 40, 50}."""

    def __init__(self, candidate_depths: List[int] = None):
        self.candidate_depths = candidate_depths or [10, 20, 30, 40, 50]

    def evaluate_question(
        self,
        q_id: str,
        gold_evidence_list: List[Dict[str, Any]],
        evidence_mapping: Dict[EvidenceKey, List[str]],
        canonical_cids: Set[str],
        bm25_cands: List[Dict[str, Any]],
        dense_cands: List[Dict[str, Any]],
        union_cands: List[Dict[str, Any]],
        post_fusion_cands: List[Dict[str, Any]],
        post_rerank_cands: List[Dict[str, Any]],
        final_context_cands: List[Dict[str, Any]],
        effective_depth_n: int = 20
    ) -> List[Dict[str, Any]]:
        
        diagnostics = []

        bm25_ids = set(c["chunk_id"] for c in bm25_cands)
        dense_ids = set(c["chunk_id"] for c in dense_cands)
        union_ids = set(c["chunk_id"] for c in union_cands)
        fusion_ids = set(c["chunk_id"] for c in post_fusion_cands)
        rerank_ids = set(c["chunk_id"] for c in post_rerank_cands)
        final_ids = set(c["chunk_id"] for c in final_context_cands)

        bm25_id_list = [c["chunk_id"] for c in bm25_cands]
        dense_id_list = [c["chunk_id"] for c in dense_cands]
        union_id_list = [c["chunk_id"] for c in union_cands]
        fusion_id_list = [c["chunk_id"] for c in post_fusion_cands]
        rerank_id_list = [c["chunk_id"] for c in post_rerank_cands]
        final_id_list = [c["chunk_id"] for c in final_context_cands]

        for ev in gold_evidence_list:
            ev_id = ev["evidence_id"]
            raw_mapped = get_evidence_mapping(evidence_mapping, q_id, ev_id)
            val = validate_evidence_mapping(raw_mapped, canonical_cids)
            mapped_chunks = set(val["valid_chunk_ids"])

            if val["status"] != "VALID":
                first_missing = "gold_mapping"
                failure_class = "ANNOTATION_OR_MAPPING_ERROR"
                cov_status = val["status"]
                is_complete = is_partial = False
                bm25_rank = dense_rank = union_rank = fusion_rank = rerank_rank = final_rank = None
                in_bm25 = in_dense = in_union = in_fusion = in_rerank = in_final = False
            else:
                bm25_cov = classify_evidence_coverage(mapped_chunks, bm25_ids)
                dense_cov = classify_evidence_coverage(mapped_chunks, dense_ids)
                union_cov = classify_evidence_coverage(mapped_chunks, union_ids)
                fusion_cov = classify_evidence_coverage(mapped_chunks, fusion_ids)
                rerank_cov = classify_evidence_coverage(mapped_chunks, rerank_ids)
                final_cov = classify_evidence_coverage(mapped_chunks, final_ids)

                in_bm25 = bm25_cov["complete"]
                in_dense = dense_cov["complete"]
                in_union = union_cov["complete"]
                in_fusion = fusion_cov["complete"]
                in_rerank = rerank_cov["complete"]
                in_final = final_cov["complete"]

                cov_status = final_cov["status"]
                is_complete = final_cov["complete"]
                is_partial = final_cov["partial"]

                bm25_rank = min([bm25_id_list.index(cid) + 1 for cid in mapped_chunks if cid in bm25_id_list], default=None)
                dense_rank = min([dense_id_list.index(cid) + 1 for cid in mapped_chunks if cid in dense_id_list], default=None)
                union_rank = min([union_id_list.index(cid) + 1 for cid in mapped_chunks if cid in union_id_list], default=None)
                fusion_rank = min([fusion_id_list.index(cid) + 1 for cid in mapped_chunks if cid in fusion_id_list], default=None)
                rerank_rank = min([rerank_id_list.index(cid) + 1 for cid in mapped_chunks if cid in rerank_id_list], default=None)
                final_rank = min([final_id_list.index(cid) + 1 for cid in mapped_chunks if cid in final_id_list], default=None)

                if not union_cov["complete"]:
                    first_missing = "candidate_pool"
                    if not bm25_cov["complete"] and not dense_cov["complete"]:
                        failure_class = "CANDIDATE_GENERATION_FAILURE"
                    elif not bm25_cov["complete"]:
                        failure_class = "BM25_RETRIEVAL_MISS"
                    else:
                        failure_class = "DENSE_RETRIEVAL_MISS"
                elif not fusion_cov["complete"]:
                    first_missing = "post_fusion_candidates"
                    failure_class = "FUSION_OR_TRUNCATION_LOSS"
                elif not rerank_cov["complete"]:
                    first_missing = "post_reranking_candidates"
                    failure_class = "RERANKING_OR_SELECTION_LOSS"
                elif not final_cov["complete"]:
                    first_missing = "final_context"
                    failure_class = "FINAL_CONTEXT_TRUNCATION"
                else:
                    first_missing = None
                    failure_class = "EVIDENCE_SUFFICIENCY_PASS"

            diagnostics.append({
                "question_id": q_id,
                "evidence_id": ev_id,
                "candidate_depth_n": effective_depth_n,
                "mapped_chunk_ids": sorted(mapped_chunks),
                "mapping_validation_status": val["status"],
                "coverage_status": cov_status,
                "strict_complete_coverage": is_complete,
                "partial_coverage": is_partial,
                "present_in_bm25": in_bm25,
                "bm25_rank": bm25_rank,
                "present_in_dense": in_dense,
                "dense_rank": dense_rank,
                "present_in_union": in_union,
                "union_rank": union_rank,
                "present_after_fusion": in_fusion,
                "fusion_rank": fusion_rank,
                "present_after_reranking": in_rerank,
                "reranked_rank": rerank_rank,
                "present_in_final_context": in_final,
                "final_context_rank": final_rank,
                "first_stage_missing": first_missing,
                "failure_classification": failure_class
            })

        return diagnostics


# ==============================================================================
# STAGE 2: CONTEXT RECONSTRUCTION (TARGET-BLIND RUNTIME & TEXT-VERIFIED EVALUATOR)
# ==============================================================================

class RuntimeContextReconstruction:
    """Target-Blind Runtime Context Reconstruction. Does NOT accept gold labels or mapping information."""

    def reconstruct_runtime_context(
        self,
        retrieved_chunks: List[Dict[str, Any]],
        corpus_chunks: List[Dict[str, Any]] = None,
        enable_document_expansion: bool = False
    ) -> Dict[str, Any]:
        
        dedup_retrieved = unique_ranked_chunks(retrieved_chunks)
        sorted_chunks = sorted(dedup_retrieved, key=lambda x: x.get("chunk_idx", 0))
        original_cids = [c["chunk_id"] for c in sorted_chunks]
        expanded_chunks = copy.deepcopy(sorted_chunks)
        expanded_cids = []

        if enable_document_expansion and corpus_chunks:
            retrieved_cids = set(original_cids)
            corpus_map = {c["chunk_id"]: c for c in corpus_chunks}
            for c in sorted_chunks:
                idx = c.get("chunk_idx", 0)
                for adj_idx in [idx - 1, idx + 1]:
                    adj_cid = f"C{adj_idx}"
                    if adj_cid in corpus_map and adj_cid not in retrieved_cids:
                        expanded_chunks.append(copy.deepcopy(corpus_map[adj_cid]))
                        retrieved_cids.add(adj_cid)
                        expanded_cids.append(adj_cid)

            expanded_chunks = sorted(expanded_chunks, key=lambda x: x.get("chunk_idx", 0))

        reconstructed_text = "\n\n".join([f"[{c['chunk_id']}] {c['text']}" for c in expanded_chunks])
        return {
            "original_chunk_ids": original_cids,
            "expanded_chunk_ids": sorted(expanded_cids),
            "chunks_included": [c["chunk_id"] for c in expanded_chunks],
            "reconstructed_text": reconstructed_text,
            "expansion_applied": bool(expanded_cids)
        }


class BenchmarkContextEvaluator:
    """Evaluates reconstructed runtime context post-hoc against golden_set.json ground truth with text verification."""

    def evaluate_context_coverage(
        self,
        reconstructed_context_info: Dict[str, Any],
        gold_evidence_list: List[Dict[str, Any]],
        evidence_mapping: Dict[EvidenceKey, List[str]],
        question_id: str,
        canonical_cids: Set[str]
    ) -> Dict[str, Any]:
        
        retrieved_cids = set(reconstructed_context_info["chunks_included"])
        original_cids = set(reconstructed_context_info.get("original_chunk_ids", []))
        expanded_cids = set(reconstructed_context_info.get("expanded_chunk_ids", []))
        recon_text_raw = reconstructed_context_info.get("reconstructed_text", "").lower()
        clean_recon_text = re.sub(r"\[c\d+\]", " ", recon_text_raw)
        clean_recon_text = re.sub(r"\s+", " ", clean_recon_text).strip()

        reconstructed_units = []
        valid_count = 0
        complete_count = 0
        expanded_count = 0

        for ev in gold_evidence_list:
            ev_id = ev["evidence_id"]
            quote = ev.get("quote", "")
            quote_norm = re.sub(r"\s+", " ", quote.lower()).strip()

            raw_mapped = get_evidence_mapping(evidence_mapping, question_id, ev_id)
            val = validate_evidence_mapping(raw_mapped, canonical_cids)
            required_cids = set(val["valid_chunk_ids"])
            
            if val["status"] != "VALID":
                status = "UNMAPPED_ANNOTATION_ERROR"
            else:
                valid_count += 1
                text_contains_quote = False
                if quote_norm and quote_norm in clean_recon_text:
                    text_contains_quote = True
                elif quote_norm:
                    clauses = [c.strip() for c in re.split(r"[;,.]", quote_norm) if len(c.strip()) > 15]
                    if clauses and any(clause in clean_recon_text for clause in clauses):
                        text_contains_quote = True
                else:
                    text_contains_quote = True

                if required_cids.issubset(retrieved_cids) and text_contains_quote:
                    if required_cids.issubset(original_cids):
                        if len(required_cids) > 1:
                            status = "EVIDENCE_RECONSTRUCTED_FROM_RETRIEVED_CHUNKS"
                        else:
                            status = "EVIDENCE_COMPLETE_IN_RETRIEVED_CHUNKS"
                        complete_count += 1
                    else:
                        status = "EVIDENCE_RECOVERED_VIA_DOCUMENT_EXPANSION"
                        expanded_count += 1
                else:
                    status = "EVIDENCE_INCOMPLETE_OR_UNAVAILABLE"

            reconstructed_units.append({
                "evidence_id": ev_id,
                "status": status,
                "mapping_validation": val["status"],
                "required_chunk_ids": sorted(required_cids),
                "retrieved_chunk_ids": sorted(required_cids & retrieved_cids)
            })

        return {
            "reconstructed_units": reconstructed_units,
            "valid_evidence_units_count": valid_count,
            "complete_evidence_coverage_count": complete_count,
            "expanded_evidence_coverage_count": expanded_count,
            "total_annotated_units": len(gold_evidence_list)
        }


# ==============================================================================
# STAGE 3: EVIDENCE-AWARE RANKING & DETERMINISTIC COMPARISON
# ==============================================================================

class EvidenceAwareRankingStage:
    """Stage 3: Reranks candidate passages using target-blind diversity & relevance heuristics."""

    def rank_by_cross_encoder(
        self,
        question: str,
        candidate_chunks: List[Dict[str, Any]],
        cross_encoder: CrossEncoder,
        top_k: int = 15
    ) -> List[Dict[str, Any]]:
        """Baseline Cross-Encoder relevance ranking with finite score validation."""
        if not candidate_chunks:
            return []

        cands_copy = copy.deepcopy(candidate_chunks)
        ce_pairs = [(question, c["text"]) for c in cands_copy]
        ce_scores = cross_encoder.predict(ce_pairs, batch_size=32, show_progress_bar=False)

        for c, sc in zip(cands_copy, ce_scores):
            sc_val = float(sc)
            if not math.isfinite(sc_val):
                raise ValueError(f"Cross-encoder returned non-finite score '{sc_val}' for chunk {c.get('chunk_id')}")
            c["ce_score"] = sc_val

        return sorted(cands_copy, key=lambda x: x["ce_score"], reverse=True)[:top_k]

    def rank_with_diversity(
        self,
        question: str,
        candidate_chunks: List[Dict[str, Any]],
        cross_encoder: CrossEncoder,
        top_k: int = 15
    ) -> List[Dict[str, Any]]:
        """Diversity / MMR heuristic ranking against redundant parent sections."""
        baseline = self.rank_by_cross_encoder(question, candidate_chunks, cross_encoder, top_k=len(candidate_chunks))
        selected = []
        unselected = copy.deepcopy(baseline)

        while unselected and len(selected) < top_k:
            if not selected:
                selected.append(unselected.pop(0))
            else:
                best_score = -float("inf")
                best_idx = 0
                for idx, cand in enumerate(unselected):
                    rel = cand.get("ce_score", 0.0)
                    redundancy = 0.0
                    for sel in selected:
                        if cand.get("parent_id") and cand.get("parent_id") == sel.get("parent_id"):
                            redundancy = max(redundancy, 0.8)
                    
                    adjusted_score = 0.7 * rel - 0.3 * redundancy
                    if adjusted_score > best_score:
                        best_score = adjusted_score
                        best_idx = idx
                
                selected.append(unselected.pop(best_idx))

        return selected

    def rank(
        self,
        question: str,
        candidate_chunks: List[Dict[str, Any]],
        cross_encoder: CrossEncoder,
        top_k: int = 15
    ) -> List[Dict[str, Any]]:
        """Default target-blind MMR ranker."""
        return self.rank_with_diversity(question, candidate_chunks, cross_encoder, top_k=top_k)


def compare_rankings(
    baseline_chunks: List[Dict[str, Any]],
    diversity_chunks: List[Dict[str, Any]],
    gold_evidence_list: List[Dict[str, Any]],
    evidence_mapping: Dict[EvidenceKey, List[str]],
    question_id: str,
    canonical_chunk_ids: Set[str],
    top_k: int = 15,
) -> Dict[str, Any]:
    """Pure, deterministic function comparing baseline vs diversity ranking metrics."""
    stage4 = SufficiencyGateStage()
    dummy_gate = {"sufficiency_pass": True, "gate_decision": "PASS", "gate_reason": "ok"}

    base_eval = stage4.evaluate_benchmark_ground_truth(question_id, {}, gold_evidence_list, evidence_mapping, canonical_chunk_ids, baseline_chunks, dummy_gate)
    div_eval = stage4.evaluate_benchmark_ground_truth(question_id, {}, gold_evidence_list, evidence_mapping, canonical_chunk_ids, diversity_chunks, dummy_gate)

    base_rec = base_eval["evidence_recall"]
    div_rec = div_eval["evidence_recall"]

    base_missing = set(base_eval["missing_evidence_ids"])
    div_missing = set(div_eval["missing_evidence_ids"])

    evidence_gained = sorted(base_missing - div_missing)
    evidence_lost = sorted(div_missing - base_missing)

    seen_parents_base = set()
    red_base = 0
    for c in baseline_chunks:
        pid = c.get("parent_id")
        if pid and pid in seen_parents_base:
            red_base += 1
        elif pid:
            seen_parents_base.add(pid)
    base_red = (red_base / len(baseline_chunks)) if baseline_chunks else 0.0

    seen_parents_div = set()
    red_div = 0
    for c in diversity_chunks:
        pid = c.get("parent_id")
        if pid and pid in seen_parents_div:
            red_div += 1
        elif pid:
            seen_parents_div.add(pid)
    div_red = (red_div / len(diversity_chunks)) if diversity_chunks else 0.0

    return {
        "top_k": top_k,
        "baseline_evidence_recall": base_rec,
        "diversity_evidence_recall": div_rec,
        "evidence_gained_by_diversity": evidence_gained,
        "evidence_lost_by_diversity": evidence_lost,
        "baseline_redundancy_rate": base_red,
        "diversity_redundancy_rate": div_red,
        "baseline_context_size": len(baseline_chunks),
        "diversity_context_size": len(diversity_chunks),
        "valid_evidence_denominator": div_eval["valid_evidence_count"],
        "invalid_evidence_count": div_eval["invalid_evidence_count"]
    }



def aggregate_ranking_comparisons(
    all_depth_results: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute dataset-level ranking metrics and expose the denominator
    used for each mean.
    """
    base_recalls = []
    div_recalls = []
    base_redundancy_rates = []
    div_redundancy_rates = []

    gained_counts = 0
    lost_counts = 0
    observation_count = 0

    for depth_key, results in all_depth_results.items():
        for result in results:
            observation_count += 1
            comparison = result.get("ranking_comparison") or {}

            base_recall = comparison.get("baseline_evidence_recall")
            div_recall = comparison.get("diversity_evidence_recall")

            base_red = comparison.get("baseline_redundancy_rate")
            div_red = comparison.get("diversity_redundancy_rate")

            if base_recall is not None:
                base_recalls.append(float(base_recall))

            if div_recall is not None:
                div_recalls.append(float(div_recall))

            if base_red is not None:
                base_redundancy_rates.append(float(base_red))

            if div_red is not None:
                div_redundancy_rates.append(float(div_red))

            gained_counts += len(
                comparison.get("evidence_gained_by_diversity") or []
            )
            lost_counts += len(
                comparison.get("evidence_lost_by_diversity") or []
            )

    return {
        "observation_count": observation_count,

        "mean_baseline_evidence_recall": (
            float(np.mean(base_recalls)) if base_recalls else None
        ),
        "baseline_recall_observation_count": len(base_recalls),

        "mean_diversity_evidence_recall": (
            float(np.mean(div_recalls)) if div_recalls else None
        ),
        "diversity_recall_observation_count": len(div_recalls),

        "mean_baseline_redundancy_rate": (
            float(np.mean(base_redundancy_rates))
            if base_redundancy_rates else None
        ),
        "baseline_redundancy_observation_count": len(
            base_redundancy_rates
        ),

        "mean_diversity_redundancy_rate": (
            float(np.mean(div_redundancy_rates))
            if div_redundancy_rates else None
        ),
        "diversity_redundancy_observation_count": len(
            div_redundancy_rates
        ),

        "total_evidence_units_gained_by_diversity": gained_counts,
        "total_evidence_units_lost_by_diversity": lost_counts,
    }


# ==============================================================================
# STAGE 4: SUFFICIENCY GATE & STAGE 5: RECOVERY RETRIEVAL CONTROLLER
# ==============================================================================

class SufficiencyGateStage:
    """Stage 4: Target-Blind Runtime Gate + Ground Truth Evaluator."""

    def evaluate_runtime_gate(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]],
        min_chunks: int = 2
    ) -> Dict[str, Any]:
        """Target-blind runtime gate decision based strictly on Cross-Encoder score distribution."""
        if not retrieved_chunks:
            return {
                "gate_decision": "ABSTAIN",
                "gate_reason": "Zero chunks retrieved in context.",
                "sufficiency_pass": False
            }

        top_scores = [c.get("ce_score", 0.0) for c in retrieved_chunks[:5]]
        max_score = max(top_scores) if top_scores else -10.0
        avg_score = float(np.mean(top_scores)) if top_scores else -10.0

        if len(retrieved_chunks) >= min_chunks and max_score > -2.0 and avg_score > -5.0:
            return {
                "gate_decision": "PASS",
                "gate_reason": f"Context contains {len(retrieved_chunks)} relevant passages (max score: {max_score:.2f}).",
                "sufficiency_pass": True
            }
        elif len(retrieved_chunks) > 0 and max_score > -6.0:
            return {
                "gate_decision": "RECOVER",
                "gate_reason": f"Marginal context retrieved ({len(retrieved_chunks)} chunks, max score: {max_score:.2f}). Recovery recommended.",
                "sufficiency_pass": False
            }
        else:
            return {
                "gate_decision": "ABSTAIN",
                "gate_reason": f"Context relevance too low (max score: {max_score:.2f}).",
                "sufficiency_pass": False
            }

    def evaluate_benchmark_ground_truth(
        self,
        q_id: str,
        evidence_policy: Dict[str, Any],
        gold_evidence_list: List[Dict[str, Any]],
        evidence_mapping: Dict[EvidenceKey, List[str]],
        canonical_cids: Set[str],
        retrieved_chunks: List[Dict[str, Any]],
        runtime_gate_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluator check comparing runtime gate decision against golden_set.json ground truth with valid denominators."""
        retrieved_cids = set(c["chunk_id"] for c in retrieved_chunks)
        
        valid_ev_ids = []
        invalid_ev_ids = []
        present_ev_ids = []
        missing_ev_ids = []

        for ev in gold_evidence_list:
            ev_id = ev["evidence_id"]
            raw_mapped = get_evidence_mapping(evidence_mapping, q_id, ev_id)
            val = validate_evidence_mapping(raw_mapped, canonical_cids)
            
            if val["status"] != "VALID":
                invalid_ev_ids.append(ev_id)
            else:
                valid_ev_ids.append(ev_id)
                mapped_cids = set(val["valid_chunk_ids"])
                if mapped_cids.issubset(retrieved_cids):
                    present_ev_ids.append(ev_id)
                else:
                    missing_ev_ids.append(ev_id)

        valid_count = len(valid_ev_ids)
        present_count = len(present_ev_ids)
        evidence_recall = (present_count / valid_count) if valid_count > 0 else None
        gold_pass = (present_count == valid_count) and valid_count > 0

        runtime_pass = runtime_gate_result["sufficiency_pass"]
        decision_agrees_with_gold = (runtime_pass == gold_pass)

        return {
            "question_id": q_id,
            "runtime_gate_decision": runtime_gate_result["gate_decision"],
            "runtime_gate_reason": runtime_gate_result["gate_reason"],
            "sufficiency_pass": runtime_pass,
            "gold_strict_sufficiency_pass": gold_pass,
            "decision_agrees_with_gold": decision_agrees_with_gold,
            "valid_evidence_count": valid_count,
            "invalid_evidence_count": len(invalid_ev_ids),
            "present_required_evidence_count": present_count,
            "missing_evidence_ids": missing_ev_ids,
            "invalid_evidence_ids": invalid_ev_ids,
            "evidence_recall": evidence_recall
        }


class RecoveryRetrievalController:
    """Stateful Recovery Retrieval Controller keyed by (question_id, candidate_depth) run identity."""

    def __init__(self, pipeline: Any, embedder: Any, corpus_chunks: List[Dict[str, Any]], max_retries: int = 1):
        self.pipeline = pipeline
        self.embedder = embedder
        self.corpus_chunks = corpus_chunks
        self.max_retries = max_retries
        self.attempts_by_run: Dict[Tuple[str, int], int] = {}

    def execute_recovery_and_reassess(
        self,
        question_id: str,
        question: str,
        initial_candidates: List[Dict[str, Any]],
        cross_encoder: CrossEncoder,
        gate_stage: SufficiencyGateStage,
        candidate_depth: int = 10,
        expanded_depth: int = 40,
        context_budget: int = 15
    ) -> Dict[str, Any]:
        if not isinstance(context_budget, int) or context_budget <= 0:
            raise ValueError(
                f"context_budget must be a positive integer, got {context_budget}."
            )

        run_key = (question_id, candidate_depth)
        attempts = self.attempts_by_run.get(run_key, 0)

        if attempts >= self.max_retries:
            initial_gate = gate_stage.evaluate_runtime_gate(question, initial_candidates)
            return {
                "recovery_executed": False,
                "recovery_status": "RETRY_LIMIT_REACHED",
                "attempt_count": attempts,
                "max_retries": self.max_retries,
                "expanded_depth": expanded_depth,
                "new_chunks_added_count": 0,
                "recovered_candidates": initial_candidates,
                "new_chunk_ids": [],
                "post_recovery_gate": initial_gate
            }

        self.attempts_by_run[run_key] = attempts + 1
        initial_cids = set(c["chunk_id"] for c in initial_candidates)

        # Deeper dense retrieval
        q_emb = self.embedder.encode([question], normalize_embeddings=True)
        if isinstance(q_emb, np.ndarray):
            q_emb = q_emb.flatten()
        doc_embs = self.embedder.encode([c["text"] for c in self.corpus_chunks], normalize_embeddings=True)
        dense_sims = np.dot(doc_embs, q_emb).flatten()
        deeper_indices = np.argsort(-dense_sims)[:expanded_depth]
        deeper_cands = [self.corpus_chunks[int(i)] for i in deeper_indices]

        # Combine & deduplicate candidates
        combined = unique_ranked_chunks(initial_candidates + deeper_cands)

        # Rerank recovered set & enforce context_budget
        ce_pairs = [(question, c["text"]) for c in combined]
        ce_scores = cross_encoder.predict(ce_pairs, batch_size=32, show_progress_bar=False)
        
        if len(ce_scores) != len(combined):
            raise ValueError(
                "Cross-encoder score count does not match recovery candidate count."
            )

        for c, sc in zip(combined, ce_scores):
            score = float(sc)

            if not math.isfinite(score):
                raise ValueError(
                    "Cross-encoder returned a non-finite score "
                    f"for recovery chunk {c.get('chunk_id')}: {score}"
                )

            c["ce_score"] = score


        recovered_ranked = sorted(combined, key=lambda x: x["ce_score"], reverse=True)[:context_budget]
        new_cids = set(c["chunk_id"] for c in recovered_ranked) - initial_cids

        # Re-evaluate runtime sufficiency gate on updated context
        post_recovery_gate = gate_stage.evaluate_runtime_gate(question, recovered_ranked)

        return {
            "recovery_executed": True,
            "recovery_status": "RECOVERY_EXECUTED",
            "attempt_count": attempts + 1,
            "max_retries": self.max_retries,
            "expanded_depth": expanded_depth,
            "new_chunks_added_count": len(new_cids),
            "recovered_candidates": recovered_ranked,
            "new_chunk_ids": sorted(new_cids),
            "post_recovery_gate": post_recovery_gate
        }


# ==============================================================================
# STAGE 6: GENERATION OR ABSTENTION (STRUCTURED STATUSES)
# ==============================================================================

class GenerationOrAbstentionStage:
    """Stage 6: Structured generation response handling (SUCCESS, ABSTAINED, CLIENT_UNAVAILABLE, API_ERROR)."""

    def __init__(self, llm_client: OpenAI = None):
        self.llm_client = llm_client

    def abstain(self, reason: str = "Runtime context remained insufficient.") -> Dict[str, Any]:
        """Returns structured ABSTAINED disposition."""
        return {
            "status": "ABSTAINED",
            "generated": False,
            "answer_text": f"I cannot answer this question: {reason}",
            "error_message": None
        }

    def generate(
        self,
        question: str,
        reconstructed_context: str,
        gate_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        
        gate_decision = gate_result.get("runtime_gate_decision", gate_result.get("gate_decision"))

        if gate_decision not in ["PASS"]:
            return self.abstain(reason=f"Final gate decision was {gate_decision}.")

        if not self.llm_client:
            return {
                "status": "CLIENT_UNAVAILABLE",
                "generated": False,
                "answer_text": "Generation skipped: No LLM client configured in current environment.",
                "error_message": "OpenAI client instance is None."
            }

        prompt = (
            "Answer the question thoroughly based ONLY on the provided context.\n"
            "Every statement of fact MUST include inline citations to the chunk ID e.g. [C4].\n\n"
            f"CONTEXT:\n{reconstructed_context}\n\n"
            f"QUESTION: {question}\n\nANSWER:"
        )

        try:
            resp = self.llm_client.chat.completions.create(
                model="nvidia/nemotron-3-ultra-550b-a55b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            answer_text = resp.choices[0].message.content.strip()
            return {
                "status": "SUCCESS",
                "generated": True,
                "answer_text": answer_text,
                "error_message": None
            }
        except Exception as e:
            return {
                "status": "API_ERROR",
                "generated": False,
                "answer_text": f"Generation failed due to LLM API error.",
                "error_message": str(e)
            }


# ==============================================================================
# STAGE 7: CITATION AND CLAIM VERIFICATION
# ==============================================================================

class CitationAndClaimVerificationStage:
    """Stage 7: Citation and Claim Verification using explicit lexical support heuristics."""

    def verify(
        self,
        q_id: str,
        generation_result: Dict[str, Any],
        retrieved_chunks: List[Dict[str, Any]],
        gold_answer: str,
        required_facts: List[str]
    ) -> Dict[str, Any]:
        
        gen_status = generation_result.get("status", "UNKNOWN")

        if not generation_result.get("generated") or gen_status != "SUCCESS":
            return {
                "claims": [],
                "claim_support_rate": None,
                "citation_accuracy": None,
                "unsupported_claim_rate": None,
                "answer_correctness": None,
                "has_evaluable_claims": False,
                "verification_status": "NOT_EVALUATED",
                "generation_status": gen_status,
                "verification_method": "lexical_support_heuristic",
                "verification_limitations": "Lexical token overlap heuristic; non-SUCCESS generation not evaluated."
            }

        answer_text = generation_result.get("answer_text", "")
        sentences = [s.strip() for s in re.split(r"[.!?]", answer_text) if len(s.strip()) > 10]

        if not sentences:
            return {
                "claims": [],
                "claim_support_rate": None,
                "citation_accuracy": None,
                "unsupported_claim_rate": None,
                "answer_correctness": 0.0,
                "has_evaluable_claims": False,
                "verification_status": "EVALUATED_EMPTY_ANSWER",
                "generation_status": gen_status,
                "verification_method": "lexical_support_heuristic",
                "verification_limitations": "Lexical token overlap heuristic; empty answer produced."
            }

        claims = []
        retrieved_map = {c["chunk_id"]: c for c in retrieved_chunks}

        supported_claims = 0
        accurate_citations = 0
        total_citations = 0

        for idx, sentence in enumerate(sentences, 1):
            citations = re.findall(r"\[(C\d+)\]", sentence)
            total_citations += len(citations)

            citation_resolves = all(cid in retrieved_map for cid in citations) if citations else False
            citation_supports = False
            contradicts = False

            if citation_resolves and citations:
                cited_text = "\n".join([retrieved_map[cid]["text"] for cid in citations]).lower()
                clean_claim = re.sub(r"\[C\d+\]", "", sentence).strip().lower()
                
                claim_words = set(re.findall(r"\b[a-z]{4,}\b", clean_claim))
                cited_words = set(re.findall(r"\b[a-z]{4,}\b", cited_text))

                overlap = len(claim_words.intersection(cited_words))
                if overlap >= min(2, len(claim_words)):
                    citation_supports = True
                    accurate_citations += len(citations)

            if citation_supports and not contradicts:
                supported_claims += 1

            claims.append({
                "question_id": q_id,
                "claim_id": f"{q_id}_claim_{idx}",
                "claim_text": sentence,
                "cited_evidence_ids": citations,
                "citation_present": len(citations) > 0,
                "citation_resolves": citation_resolves,
                "citation_supports_claim": citation_supports,
                "contradicts_verified_evidence": contradicts,
                "claim_supported": citation_supports and not contradicts,
                "verification_method": "lexical_support_heuristic",
                "verification_reason": "Verified lexical token overlap against cited chunk." if (citation_supports and not contradicts) else ("Unresolved or unsupporting citation.")
            })

        total_claims = len(claims)
        support_rate = (supported_claims / total_claims) if total_claims > 0 else None
        citation_acc = (accurate_citations / total_citations) if total_citations > 0 else None
        unsupported_rate = (1.0 - (supported_claims / total_claims)) if total_claims > 0 else None

        # Post-hoc Answer Correctness evaluation against required facts
        fact_matches = 0
        answer_lower = answer_text.lower()
        for fact in required_facts:
            fact_words = set(re.findall(r"\b[a-z]{4,}\b", fact.lower()))
            if len(fact_words.intersection(set(re.findall(r"\b[a-z]{4,}\b", answer_lower)))) >= 2:
                fact_matches += 1
        answer_corr = (fact_matches / len(required_facts)) if required_facts else 0.0

        return {
            "claims": claims,
            "claim_support_rate": support_rate,
            "citation_accuracy": citation_acc,
            "unsupported_claim_rate": unsupported_rate,
            "answer_correctness": answer_corr,
            "has_evaluable_claims": total_claims > 0,
            "verification_status": "EVALUATED_SUCCESS",
            "generation_status": gen_status,
            "verification_method": "lexical_support_heuristic",
            "verification_limitations": "Lexical token overlap heuristic; not general semantic entailment."
        }


# ==============================================================================
# MAIN LOCAL EVALUATION EXECUTION ACROSS MULTI-DEPTHS N in {10, 20, 30, 40, 50}
# ==============================================================================


def generate_from_runtime_gate(
    generation_stage: GenerationOrAbstentionStage,
    question: str,
    reconstructed_context: str,
    runtime_gate: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generation decisions depend only on the target-blind runtime gate.
    Post-hoc gold evaluation must never be passed to this function.
    """
    if runtime_gate.get("gate_decision") == "PASS":
        return generation_stage.generate(
            question,
            reconstructed_context,
            runtime_gate,
        )

    return generation_stage.abstain(
        reason=f"Runtime gate decision: {runtime_gate.get('gate_decision')}."
    )



def _unique_artifact_path(directory: str, filename: str) -> str:
    """Return a filename that does not collide with an existing artifact."""
    stem, extension = os.path.splitext(filename)
    candidate = os.path.join(directory, filename)
    suffix = 1

    while os.path.exists(candidate):
        candidate = os.path.join(
            directory,
            f"{stem}_{suffix}{extension}",
        )
        suffix += 1

    return candidate


def save_evaluation_results(
    results_payload: Dict[str, Any],
    results_dir: Optional[str] = None
) -> Dict[str, str]:
    """
    Saves benchmark evaluation results safely with unique timestamped backups.
    Guarantees:
    1. Backup filenames use microsecond precision (%Y%m%d_%H%M%S_%f) to prevent collisions.
    2. Existing results are backed up and verified BEFORE primary output is replaced.
    3. Failed backups raise RuntimeError and abort write, preserving existing primary results.
    4. Primary file and timestamped run log receive identical JSON content payload.
    """
    if results_dir is None:
        results_dir = os.path.join(current_dir, "results")
    
    os.makedirs(results_dir, exist_ok=True)
    primary_path = os.path.join(results_dir, "phase_evidence_based_eval_results.json")
    
    now = datetime.now()
    timestamp_str = now.strftime("%Y%m%d_%H%M%S_%f")
    backup_dir = os.path.join(results_dir, "backups")
    # backup_path = os.path.join(backup_dir, f"phase_evidence_based_eval_results_{timestamp_str}.json")
    # run_log_path = os.path.join(results_dir, f"phase_evidence_based_eval_results_{timestamp_str}.json")

    
    artifact_filename = (
        f"phase_evidence_based_eval_results_{timestamp_str}.json"
    )

    backup_path = _unique_artifact_path(
        backup_dir,
        artifact_filename,
    )

    run_log_path = _unique_artifact_path(
        results_dir,
        artifact_filename,
    )


    # 1. Atomic Backup of Existing Primary File (If Present)
    if os.path.exists(primary_path):
        os.makedirs(backup_dir, exist_ok=True)
        try:
            with open(primary_path, "rb") as f_in, open(backup_path, "wb") as f_out:
                existing_bytes = f_in.read()
                f_out.write(existing_bytes)
            
            # Verify backup completeness before proceeding
            if not os.path.exists(backup_path) or os.path.getsize(backup_path) != len(existing_bytes):
                raise IOError(f"Backup verification failed: size mismatch on {backup_path}")
        except Exception as e:
            if os.path.exists(backup_path):
                try:
                    os.remove(backup_path)
                except Exception:
                    pass
            raise RuntimeError(
                f"Failed to create backup of existing evaluation results: {e}. "
                "Primary results file left untouched to prevent data loss."
            ) from e

    # 2. Format JSON payload once to guarantee strict identity across output files
    payload_str = json.dumps(results_payload, indent=2)

    # 3. Write timestamped run copy
    with open(run_log_path, "w", encoding="utf-8") as f_run:
        f_run.write(payload_str)

    # 4. Write to a temporary file, then atomically replace primary
    temp_primary_path = f"{primary_path}.tmp"

    try:
        with open(temp_primary_path, "w", encoding="utf-8") as f_prim:
            f_prim.write(payload_str)
            f_prim.flush()
            os.fsync(f_prim.fileno())

        os.replace(temp_primary_path, primary_path)

    except Exception:
        if os.path.exists(temp_primary_path):
            try:
                os.remove(temp_primary_path)
            except OSError:
                pass
        raise
    return {
        "primary_path": primary_path,
        "run_log_path": run_log_path,
        "backup_path": backup_path if os.path.exists(backup_path) else None
    }


def main():
    print("==========================================================================", flush=True)
    print("   EVIDENCE-BASED RAG EVALUATION LOCAL TESTING PHASE", flush=True)
    print("==========================================================================", flush=True)

    golden_set_path = os.path.join(current_dir, "golden_set.json")
    with open(golden_set_path, "r", encoding="utf-8") as f:
        golden_set = json.load(f)

    questions = golden_set["questions"]
    print(f"Authoritative Golden Set: {len(questions)} benchmark questions (Q01-Q10 loaded).", flush=True)

    doc_path = os.path.join(root_dir, "Documents", "The Comedy of Errors document.pdf")
    pipeline = AdvancedRAGPipeline(session_id="evidence_eval_run")
    raw_docs = IngestionPipeline.load_document(doc_path)
    pipeline.add_documents(raw_docs)

    formatted_chunks = []
    canonical_cids = set()
    for idx, doc in enumerate(pipeline.all_chunks):
        cid = f"C{idx}"
        canonical_cids.add(cid)
        formatted_chunks.append({
            "chunk_id": cid,
            "chunk_idx": idx,
            "element_id": doc.metadata.get("element_id", ""),
            "parent_id": f"page_{doc.metadata.get('page', 0)}",
            "page": doc.metadata.get("page", 0),
            "text": doc.page_content
        })

    candidate_depths = [10, 20, 30, 40, 50]
    context_budget = 15

    # Harness Fail-Fast Integrity Check
    validate_benchmark_integrity(golden_set, formatted_chunks, candidate_depths, context_budget, expected_corpus_size=203)
    print(f"Corpus Ingested & Integrity Verified: {len(formatted_chunks)} canonical chunks (C0-C202).", flush=True)

    mapper = GoldEvidenceMapper(formatted_chunks)
    evidence_mapping: Dict[EvidenceKey, List[str]] = {}
    for q in questions:
        q_id = q["question_id"]
        for ev in q["gold_evidence"]:
            m_res = mapper.map_evidence_unit(q_id, ev)
            evidence_mapping[(q_id, ev["evidence_id"])] = m_res["matched_chunk_ids"]

    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    api_key_nv = os.getenv("NVIDIA_API_KEY") or os.getenv("NEMOTRON_API_KEY")
    nv_client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key_nv) if api_key_nv else None

    stage1 = CandidateRecallDiagnosticsStage()
    runtime_recon = RuntimeContextReconstruction()
    benchmark_eval = BenchmarkContextEvaluator()
    stage4 = SufficiencyGateStage()
    recovery_ctrl = RecoveryRetrievalController(pipeline, embedder, formatted_chunks, max_retries=1)
    stage6 = GenerationOrAbstentionStage(llm_client=nv_client)
    stage7 = CitationAndClaimVerificationStage()

    all_depth_results = {}
    all_diagnostics = []

    doc_embs = embedder.encode([c["text"] for c in formatted_chunks], normalize_embeddings=True)

    # Evaluate across candidate depths N in {10, 20, 30, 40, 50}
    for N in candidate_depths:
        print(f"\n--- EVALUATING BENCHMARK AT CANDIDATE DEPTH N = {N} ---", flush=True)
        depth_q_results = []

        for q in questions:
            q_id = q["question_id"]
            q_text = q["question"]
            gold_ev_list = q["gold_evidence"]
            ev_policy = q["evidence_policy"]

            # Production Orchestration Entry Point Call
            orch = orchestrate_question_retrieval(
                q_text, N, pipeline, embedder, cross_encoder, formatted_chunks, doc_embs, context_budget=context_budget
            )

            bm25_cands = orch["bm25_candidates"]
            dense_cands = orch["dense_candidates"]
            union_cands = orch["union_candidates"]
            post_fusion_cands = orch["post_fusion_candidates"]
            baseline_cands = orch["baseline_candidates"]
            mmr_cands = orch["diversity_candidates"]
            ranked_cands = mmr_cands

            # Stage 1: Candidate Diagnostics — Cross-Encoder baseline
            baseline_diag = stage1.evaluate_question(
                q_id,
                gold_ev_list,
                evidence_mapping,
                canonical_cids,
                bm25_cands,
                dense_cands,
                union_cands,
                post_fusion_cands,
                baseline_cands,
                baseline_cands,
                effective_depth_n=N,
            )

            for row in baseline_diag:
                row["ranking_strategy"] = "cross_encoder_baseline"

            # Stage 1: Candidate Diagnostics — Diversity/MMR
            diversity_diag = stage1.evaluate_question(
                q_id,
                gold_ev_list,
                evidence_mapping,
                canonical_cids,
                bm25_cands,
                dense_cands,
                union_cands,
                post_fusion_cands,
                mmr_cands,
                mmr_cands,
                effective_depth_n=N,
            )

            for row in diversity_diag:
                row["ranking_strategy"] = "diversity_mmr"

            all_diagnostics.extend(baseline_diag)
            all_diagnostics.extend(diversity_diag)

            # Stage 2: Target-Blind Runtime Context Reconstruction
            recon_info = runtime_recon.reconstruct_runtime_context(ranked_cands, formatted_chunks, enable_document_expansion=False)

            # Stage 4: Target-Blind Runtime Gate
            runtime_gate = stage4.evaluate_runtime_gate(q_text, ranked_cands)

            # Stage 5: Bounded Recovery Retrieval if RECOVER decision (depth-independent keying!)
            if runtime_gate["gate_decision"] == "RECOVER":
                rec_res = recovery_ctrl.execute_recovery_and_reassess(
                    q_id, q_text, ranked_cands, cross_encoder, stage4,
                    candidate_depth=N, expanded_depth=N+20, context_budget=context_budget
                )
                if rec_res["recovery_executed"]:
                    ranked_cands = rec_res["recovered_candidates"]
                    # Reconstruct context for updated recovery candidates
                    recon_info = runtime_recon.reconstruct_runtime_context(ranked_cands, formatted_chunks, enable_document_expansion=False)
                    runtime_gate = rec_res["post_recovery_gate"]
                
                recovery_diag = {
                    "recovery_executed": rec_res["recovery_executed"],
                    "recovery_status": rec_res["recovery_status"],
                    "attempt_count": rec_res["attempt_count"],
                    "expanded_depth": rec_res["expanded_depth"],
                    "new_chunks_added_count": rec_res["new_chunks_added_count"],
                    "new_chunk_ids": rec_res["new_chunk_ids"],
                    "post_recovery_gate": runtime_gate["gate_decision"]
                }
            else:
                recovery_diag = {
                    "recovery_executed": False,
                    "recovery_status": f"SKIPPED_GATE_{runtime_gate['gate_decision']}",
                    "attempt_count": 0,
                    "expanded_depth": None,
                    "new_chunks_added_count": 0,
                    "new_chunk_ids": [],
                    "post_recovery_gate": runtime_gate["gate_decision"]
                }

            # Post-hoc Context Coverage & Ground Truth Evaluation (recomputed on actual final context!)
            eval_coverage = benchmark_eval.evaluate_context_coverage(recon_info, gold_ev_list, evidence_mapping, q_id, canonical_cids)
            eval_gate = stage4.evaluate_benchmark_ground_truth(q_id, ev_policy, gold_ev_list, evidence_mapping, canonical_cids, ranked_cands, runtime_gate)

            # Stage 6: Generation or Abstention (Target-Blind: passes ONLY runtime_gate, NOT gold eval_gate!)
            gen_res = generate_from_runtime_gate(
                stage6,
                q_text,
                recon_info["reconstructed_text"],
                runtime_gate,
            )
            # Stage 7: Citation & Claim Verification
            ver_res = stage7.verify(q_id, gen_res, ranked_cands, q["gold_answer"], q["required_facts"])

            # Deterministic Baseline vs MMR ranking comparison
            rank_comp = compare_rankings(baseline_cands, mmr_cands, gold_ev_list, evidence_mapping, q_id, canonical_cids, top_k=context_budget)

            bm25_diag_msg = None if len(bm25_cands) >= N else f"Requested BM25 depth {N}, but retriever returned {len(bm25_cands)} candidates."
            dense_diag_msg = None if len(dense_cands) >= N else f"Requested Dense depth {N}, but retriever returned {len(dense_cands)} candidates."

            depth_q_results.append({
                "question_id": q_id,
                "candidate_depth_n": N,
                "requested_counts": orch["requested_counts"],
                "actual_counts": orch["actual_counts"],
                "retrieval_diagnostics": {
                    "bm25_diagnostic": bm25_diag_msg,
                    "dense_diagnostic": dense_diag_msg
                },
                "ranking_comparison": rank_comp,
                "gate_decision": eval_gate["runtime_gate_decision"],
                "decision_agrees_with_gold": eval_gate["decision_agrees_with_gold"],
                "evidence_recall": eval_gate["evidence_recall"],
                "context_reconstruction": eval_coverage,
                "recovery_diagnostics": recovery_diag,
                "generation_status": gen_res["status"],
                "answer_text": gen_res["answer_text"],
                "claim_verification": ver_res
            })

            print(f"Depth N={N} | {q_id} | Gate: {eval_gate['runtime_gate_decision']} (Agrees Gold: {eval_gate['decision_agrees_with_gold']}) | Recall: {(eval_gate['evidence_recall']*100 if eval_gate['evidence_recall'] is not None else 0):.1f}%", flush=True)

        all_depth_results[f"N_{N}"] = depth_q_results

    # Aggregate ranking metrics across dataset
    agg_ranking_metrics = aggregate_ranking_comparisons(all_depth_results)

    results_payload = {
        "metadata": {
            "dataset": golden_set.get("dataset_id", "golden_set_v1"),
            "timestamp": datetime.now().isoformat(),
            "question_count": len(questions),
            "candidate_depths_evaluated": candidate_depths,
            "final_context_k": context_budget
        },
        "aggregate_ranking_metrics": agg_ranking_metrics,
        "diagnostics": all_diagnostics,
        "all_depth_results": all_depth_results
    }

    saved_paths = save_evaluation_results(results_payload)
    print(f"\nSaved multi-depth evaluation results to {saved_paths['primary_path']} and log {saved_paths['run_log_path']}", flush=True)
    if saved_paths.get("backup_path"):
        print(f"Backed up prior evaluation results to {saved_paths['backup_path']}", flush=True)

if __name__ == "__main__":
    main()
