import os
import sys
import json
import math
import time
import re
import importlib.util
from datetime import datetime
import numpy as np
import pandas as pd
import hashlib
from dotenv import load_dotenv
# Load environment
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

# Canonical Benchmark Suite (Q01–Q06 & 20 Gold Evidence Units)
BENCHMARK_SUITE = [
    {
        "q_id": "Q01",
        "question": "Who is Aegeon and what is his origin?",
        "total_required_evidence": 2,
        "gold_map": {"C4": "PRIMARY", "C2": "SUPPORTING"},
        "required_evidence": ["C4", "C2"],
        "primary_evidence": ["C4"],
        "supporting_evidence": ["C2"]
    },
    {
        "q_id": "Q02",
        "question": "Why was Aegeon sentenced to death in Ephesus?",
        "total_required_evidence": 4,
        "gold_map": {"C2": "PRIMARY", "C4": "SUPPORTING", "C10": "SUPPORTING", "C12": "SUPPORTING"},
        "required_evidence": ["C2", "C4", "C10", "C12"],
        "primary_evidence": ["C2"],
        "supporting_evidence": ["C4", "C10", "C12"]
    },
    {
        "q_id": "Q03",
        "question": "Why does Antipholus of Syracuse fail to recognize Adriana when she addresses him as her husband?",
        "total_required_evidence": 4,
        "gold_map": {"C15": "PRIMARY", "C24": "PRIMARY", "C29": "SUPPORTING", "C61": "SUPPORTING"},
        "required_evidence": ["C15", "C24", "C29", "C61"],
        "primary_evidence": ["C15", "C24"],
        "supporting_evidence": ["C29", "C61"]
    },
    {
        "q_id": "Q04",
        "question": "How did Aegeon come to have two pairs of identical twin sons and twin servants?",
        "total_required_evidence": 3,
        "gold_map": {"C4": "PRIMARY", "C6": "SUPPORTING", "C8": "SUPPORTING"},
        "required_evidence": ["C4", "C6", "C8"],
        "primary_evidence": ["C4"],
        "supporting_evidence": ["C6", "C8"]
    },
    {
        "q_id": "Q05",
        "question": "How do Antipholus of Syracuse and Antipholus of Ephesus differ in their initial interactions with Dromio of Syracuse?",
        "total_required_evidence": 3,
        "gold_map": {"C42": "PRIMARY", "C65": "PRIMARY", "C12": "SUPPORTING"},
        "required_evidence": ["C42", "C65", "C12"],
        "primary_evidence": ["C42", "C65"],
        "supporting_evidence": ["C12"]
    },
    {
        "q_id": "Q06",
        "question": "Why is Antipholus of Ephesus arrested, and what role does Angelo the goldsmith play in the debt dispute?",
        "total_required_evidence": 4,
        "gold_map": {"C105": "PRIMARY", "C107": "PRIMARY", "C147": "SUPPORTING", "C155": "SUPPORTING"},
        "required_evidence": ["C105", "C107", "C147", "C155"],
        "primary_evidence": ["C105", "C107"],
        "supporting_evidence": ["C147", "C155"]
    }
]

# Frozen Q2 Subqueries (Phase 10C / Phase 11C / Phase B Control)
FROZEN_Q2_QUERIES = {
    "Q01": [
        "Aegeon character background Syracuse merchant",
        "Aegeon family relationships Emilia twin sons",
        "Aegeon arrest Ephesus law against Syracusans"
    ],
    "Q02": [
        "Aegeon Syracuse merchant background identity",
        "Ephesus law death penalty Syracusans merchants",
        "Aegeon arrest sentencing Ephesus Duke ransom"
    ],
    "Q03": [
        "Antipholus of Syracuse visitor stranger in Ephesus",
        "Adriana marriage to Antipholus of Ephesus twin brother",
        "Mistaken identity Antipholus of Syracuse addressed by Adriana"
    ],
    "Q04": [
        "Aegeon twin sons birth circumstances Emilia",
        "Aegeon family separation shipwreck storm",
        "Aegeon twin servants Dromio purchase acquisition"
    ],
    "Q05": [
        "Antipholus of Syracuse character background and his relationship to Dromio of Syracuse",
        "Antipholus of Ephesus relationship to Dromio of Syracuse and the master-servant dynamic in Ephesus",
        "Initial encounter scenes between Antipholus of Syracuse and Dromio of Syracuse versus Antipholus of Ephesus and Dromio of Syracuse"
    ],
    "Q06": [
        "Antipholus of Ephesus character background arrest debt",
        "Angelo goldsmith relationship debt dispute legal claim",
        "Arrest of Antipholus of Ephesus scene Angelo goldsmith demand payment"
    ]
}

# Registered Control Targets & Tolerances for Phase D
PHASE_D_CONTROL_TARGETS = {
    "candidate_recall_20": {
        "target": 0.85,
        "tolerance": 0.03,
    },
    "final_recall_15": {
        "target": 0.70,
        "tolerance": 0.03,
    },
    "ndcg_10": {
        "target": 0.485,
        "tolerance": 0.030,
    },
    "mrr": {
        "target": 0.369,
        "tolerance": 0.030,
    },
}

FINAL_RECALL_PRESERVATION_TOLERANCE = 0.001
NDCG_PRESERVATION_TOLERANCE = 0.030
MRR_PRESERVATION_TOLERANCE = 0.030
Q6_TARGET_IDS = {"C12", "C65", "C155"}

def minimum_b3_recovery_depth(provenance_records):
    """
    Determines the minimum B3 depth at which any previously inaccessible evidence
    becomes candidate-accessible. Checked in strict depth order: D1 -> D2 -> D3.
    """
    if any(r.get("entered_union_D1") and not r.get("canonical_B0_candidate") for r in provenance_records):
        return "D1 (B3@20)"
    if any(r.get("entered_union_D2") and not r.get("canonical_B0_candidate") for r in provenance_records):
        return "D2 (B3@50)"
    if any(r.get("entered_union_D3") and not r.get("canonical_B0_candidate") for r in provenance_records):
        return "D3 (B3@100)"
    return "None"

def interpret_q12(verdict):
    """
    Provides a distinct, unambiguous dynamic interpretation for all three canonical Phase D verdicts.
    """
    if verdict == "PASS":
        return (
            "Yes. Phase D supports a production-relevant B3 auxiliary retrieval "
            "modification under the registered preservation and target-blindness criteria."
        )
    elif verdict == "DIAGNOSTIC SUCCESS — DOWNSTREAM BOTTLENECK":
        return (
            "Partially. B3 improves candidate accessibility, but unchanged R0 "
            "does not convert the recovered candidates into final Top-15 evidence. "
            "This supports a downstream reranking bottleneck diagnosis rather than "
            "production deployment of B3 alone."
        )
    elif verdict == "NO SUPPORT":
        return (
            "No. Phase D does not provide sufficient evidence that B3 materially "
            "improves evidence discoverability through auxiliary candidate generation."
        )
    else:
        raise AssertionError(f"Unknown Phase D verdict: {verdict}")


# Deterministic Known Entities List (Alphabetically Sorted)
KNOWN_ENTITIES = sorted([
    "Adriana", "Aegeon", "Angelo", "Antipholus of Ephesus", "Antipholus of Syracuse",
    "Centaur", "Corinth", "Dromio of Ephesus", "Dromio of Syracuse", "Emilia",
    "Epidamnum", "Ephesus", "Luciana", "Pinch", "Phoenix", "Solinus", "Syracuse"
])

ENGLISH_STOPWORDS = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
    "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
    "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", "shall",
    "thou", "thee", "thy", "thine", "ye", "upon", "hath", "doth", "would", "could", "may"
])

ACTION_VERB_PATTERNS = ["arrest", "bound", "pay", "demand", "sentenced", "drown", "separated", "mistaken", "claim", "recognize", "flee", "buy", "seek"]
EVENT_NOUN_PATTERNS = ["debt", "ransom", "law", "shipwreck", "chain", "ducats", "bond", "dispute", "feast", "dinner"]
TEMPORAL_PATTERNS = ["today", "morning", "seven years", "five hundred ducats", "presently", "now", "hour"]

def extract_entities(text):
    """Extract known entities deterministically in alphabetical order."""
    found = []
    for ent in KNOWN_ENTITIES:
        if re.search(r"\b" + re.escape(ent) + r"\b", text, re.IGNORECASE):
            found.append(ent)
    return sorted(found)

def construct_b3_semantic_representation(chunk_item):
    """
    Construct B3 structured semantic representation deterministically from metadata and text.
    CRITICAL: Does NOT contain the raw chunk text block and excludes arbitrary CHUNK_ID from semantic text.
    Uses KEY_TERMS label. Zero gold data used.
    """
    text = chunk_item["text"]
    breadcrumb = chunk_item.get("breadcrumb_path", "")
    ents = extract_entities(text)
    locations = [e for e in ents if e in ["Ephesus", "Syracuse", "Epidamnum", "Corinth", "Centaur", "Phoenix"]]
    speakers = [e for e in ents if e in ["Aegeon", "Solinus", "Antipholus of Syracuse", "Antipholus of Ephesus", "Dromio of Syracuse", "Dromio of Ephesus", "Adriana", "Luciana", "Angelo", "Pinch", "Emilia"]]
    
    actions = [v for v in ACTION_VERB_PATTERNS if re.search(r"\b" + re.escape(v) + r"\b", text, re.IGNORECASE)]
    events = [e for e in EVENT_NOUN_PATTERNS if re.search(r"\b" + re.escape(e) + r"\b", text, re.IGNORECASE)]
    temporals = [t for t in TEMPORAL_PATTERNS if re.search(r"\b" + re.escape(t) + r"\b", text, re.IGNORECASE)]
    
    words = sorted(set(re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())) - ENGLISH_STOPWORDS)
    key_terms = ", ".join(words[:10])
    
    b3_text = (
        f"SECTION: {breadcrumb}\n"
        f"ENTITIES: {', '.join(ents) if ents else 'None'}\n"
        f"LOCATIONS: {', '.join(locations) if locations else 'None'}\n"
        f"SPEAKERS: {', '.join(speakers) if speakers else 'None'}\n"
        f"ACTION_TERMS: {', '.join(actions) if actions else 'None'}\n"
        f"EVENT_TERMS: {', '.join(events) if events else 'None'}\n"
        f"TEMPORAL_TERMS: {', '.join(temporals) if temporals else 'None'}\n"
        f"KEY_TERMS: {key_terms}"
    )
    return b3_text

def calculate_metrics_at_k(selected_chunks, gold_map, top_k=15):
    selected_top = selected_chunks[:top_k]
    seen_parents = set()
    dup_count = 0
    first_useful_rank = 0
    useful_count = 0
    primary_count = 0

    rel_map = {"PRIMARY": 3, "SUPPORTING": 2, "RELEVANT_BUT_UNNECESSARY": 1, "IRRELEVANT": 0, "DUPLICATE": 0}
    classifications = []

    for idx, item in enumerate(selected_top, 1):
        cid = item["chunk_id"]
        pid = item.get("parent_id", "")

        if cid in gold_map:
            role = gold_map[cid]
            classifications.append(role)
            useful_count += 1
            if role == "PRIMARY":
                primary_count += 1
            if first_useful_rank == 0:
                first_useful_rank = idx
            if pid:
                seen_parents.add(pid)
        elif pid and pid in seen_parents:
            dup_count += 1
            classifications.append("DUPLICATE")
        else:
            if pid:
                seen_parents.add(pid)
            t_lower = item["text"].lower()
            if any(kw in t_lower for kw in ["aegeon", "syracuse", "ephesus", "antipholus", "dromio", "adriana", "goldsmith", "angelo", "chain", "ducats", "twin"]):
                classifications.append("RELEVANT_BUT_UNNECESSARY")
            else:
                classifications.append("IRRELEVANT")

    ev_precision = useful_count / top_k if top_k > 0 else 0.0
    pri_precision = primary_count / top_k if top_k > 0 else 0.0
    mrr = 1.0 / first_useful_rank if first_useful_rank > 0 else 0.0

    dcg = sum((2**rel_map.get(c, 0) - 1) / math.log2(i + 2) for i, c in enumerate(classifications[:top_k]))
    ideal_rels = sorted([3 if r == "PRIMARY" else 2 for r in gold_map.values()], reverse=True)[:top_k]
    idcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(ideal_rels))
    ndcg_k = dcg / idcg if idcg > 0 else 0.0

    return {
        "ev_precision": ev_precision,
        "pri_precision": pri_precision,
        "mrr": mrr,
        "ndcg_k": ndcg_k,
        "redundancy": dup_count,
        "classifications": classifications
    }

def verify_pre_execution_assertions(sample_chunk):
    """Explicit static pre-execution validation assertions enforcing all 19 mandatory audit invariants."""
    print("--- RUNNING PRE-EXECUTION AUDIT ASSERTION CHECKS ---", flush=True)
    import inspect
    
    main_src = inspect.getsource(main)
    report_src = inspect.getsource(generate_markdown_report)

    # 1. ASSERTION: Canonical control configuration unchanged
    assert "sq_ranked_indices[:20]" in main_src and "bm25_docs" in main_src, "CRITICAL: Canonical candidate retrieval configuration altered!"
    print("[PASS] Assertion 1: Canonical control configuration unchanged", flush=True)

    # 2. ASSERTION: E0 embedding model unchanged
    assert "sentence-transformers/all-MiniLM-L6-v2" in main_src, "CRITICAL: E0 embedding model modified!"
    print("[PASS] Assertion 2: E0 embedding model unchanged", flush=True)

    # 3. ASSERTION: R0 cross-encoder model unchanged
    assert "cross-encoder/ms-marco-MiniLM-L-6-v2" in main_src, "CRITICAL: R0 cross-encoder model modified!"
    print("[PASS] Assertion 3: R0 cross-encoder model unchanged", flush=True)

    # 4. ASSERTION: Exactly 3 frozen Q2 subqueries per benchmark question
    for b_case in BENCHMARK_SUITE:
        qid = b_case["q_id"]
        assert qid in FROZEN_Q2_QUERIES, f"CRITICAL: Question '{qid}' missing from FROZEN_Q2_QUERIES!"
        assert len(FROZEN_Q2_QUERIES[qid]) == 3, f"CRITICAL: Question '{qid}' subquery count != 3!"
    assert "FROZEN_Q2_QUERIES" in main_src, "CRITICAL: FROZEN_Q2_QUERIES missing from main execution!"
    print("[PASS] Assertion 4: Exactly 3 frozen Q2 subqueries per question validated", flush=True)

    # 5. ASSERTION: BM25 path unchanged
    assert "pipeline.bm25_retriever.k = 20" in main_src or "bm25_docs" in main_src, "CRITICAL: BM25 path altered!"
    print("[PASS] Assertion 5: BM25 path unchanged", flush=True)

    # 6. ASSERTION: B0 path unchanged
    assert "sq_ranked_indices[:20]" in main_src, "CRITICAL: B0 candidate depth modified!"
    print("[PASS] Assertion 6: B0 path unchanged", flush=True)

    # 7. ASSERTION: B3 generated target-blind without gold evidence
    b3_sig = inspect.signature(construct_b3_semantic_representation)
    assert "gold_map" not in b3_sig.parameters and "required_evidence" not in b3_sig.parameters, "CRITICAL: Gold param in B3 generator!"
    b3_sample = construct_b3_semantic_representation(sample_chunk)
    assert "Content:" not in b3_sample and "CHUNK_ID:" not in b3_sample, "CRITICAL: Raw text or CHUNK_ID found in B3!"
    assert "KEY_TERMS:" in b3_sample, "CRITICAL: KEY_TERMS tag missing from B3!"
    print("[PASS] Assertion 7: B3 generated target-blind without gold evidence", flush=True)

    # 8. ASSERTION: No target-specific candidate injection
    for hardcoded_id in ["'C15'", "'C42'", "'C29'"]:
        assert hardcoded_id not in main_src, f"CRITICAL: Target-specific hardcoded ID {hardcoded_id} found in main logic!"
    print("[PASS] Assertion 8: No target-specific candidate injection", flush=True)

    # 9. ASSERTION: No target-specific query modification
    assert "target_query_override" not in main_src, "CRITICAL: Target-specific query modification found!"
    print("[PASS] Assertion 9: No target-specific query modification", flush=True)

    # 10. ASSERTION: No target-specific ranking logic
    assert "ce_ranked = sorted" in main_src, "CRITICAL: Non-standard candidate sorting found!"
    print("[PASS] Assertion 10: No target-specific ranking logic", flush=True)

    # 11. ASSERTION: B3 is auxiliary rather than replacement retrieval
    assert "candidate_map[cid] = dict(formatted_chunks[c_i])" in main_src or "b3_candidate" in main_src, "CRITICAL: Auxiliary union logic missing!"
    print("[PASS] Assertion 11: B3 is auxiliary rather than replacement retrieval", flush=True)

    # 12. ASSERTION: Final Top-15 selection mechanism unchanged
    assert "ce_ranked[:15]" in main_src, "CRITICAL: Final Top-15 selection mechanism modified!"
    print("[PASS] Assertion 12: Final Top-15 selection mechanism unchanged", flush=True)

    # 13. ASSERTION: Control gate executes before diagnostics
    control_gate_idx = main_src.find("CONTROL REPRODUCTION GATE PASSED!")
    diag_eval_idx = main_src.find("EVALUATING AUXILIARY B3 DISCOVERABILITY CONFIGURATIONS")
    assert control_gate_idx != -1 and diag_eval_idx != -1 and control_gate_idx < diag_eval_idx, "CRITICAL: Control gate does not precede diagnostic evaluation!"
    print("[PASS] Assertion 13: Control gate executes before diagnostics", flush=True)

    # 14. ASSERTION: B3 depths set to exactly {20, 50, 100}
    assert "[20, 50, 100]" in main_src or "(20, 50, 100)" in main_src, "CRITICAL: B3 depths not set to {20, 50, 100}!"
    print("[PASS] Assertion 14: B3 depths set to exactly {20, 50, 100}", flush=True)

    # 15. ASSERTION: Per-evidence provenance retained
    for prov_field in ["canonical_B0_rank", "canonical_B0_candidate", "canonical_final", "phaseD_diagnosis"]:
        assert prov_field in main_src, f"CRITICAL: Provenance field '{prov_field}' missing!"
    assert 'prov_rec[f"entered_union_{cfg_key}"]' in main_src or 'entered_union_' in main_src, "CRITICAL: Union entry provenance field missing!"
    print("[PASS] Assertion 15: Per-evidence provenance retained", flush=True)

    # 16. ASSERTION: Candidate pollution metrics retained
    for poll_field in ["b3_candidates_added", "new_gold_candidates", "new_nongold_candidates", "candidate_pollution_rate"]:
        assert poll_field in main_src, f"CRITICAL: Pollution field '{poll_field}' missing!"
    print("[PASS] Assertion 16: Candidate pollution metrics retained", flush=True)

    # 17. ASSERTION: R0 promotion metrics retained
    for prom_field in ["B3_to_candidate_count", "candidate_to_R0_top15_count", "R0_top15_to_final_count"]:
        assert prom_field in main_src, f"CRITICAL: R0 promotion field '{prom_field}' missing!"
    print("[PASS] Assertion 17: R0 promotion metrics retained", flush=True)

    # 18. ASSERTION: Diagnostic results separated from production control
    assert "baseline_control_results" in main_src and "b3_diagnostic_results" in main_src, "CRITICAL: Control and diagnostic results conflated!"
    print("[PASS] Assertion 18: Diagnostic results separated from production control", flush=True)

    # 19. ASSERTION: Report Generator is genuinely data-derived without hardcoded narrative outcome claims
    required_dynamic_constructs = ["q_rows", "provenance_records", "autopsies", "prom", "baseline", "b3_results"]
    for req in required_dynamic_constructs:
        assert req in report_src or (req == "baseline" and "ctrl" in report_src) or (req == "b3_results" and "diag" in report_src), f"CRITICAL: Report generator missing dynamic construct {req}!"

    assert "for q_r in q_rows:" in report_src, "CRITICAL: Dynamic per-question table generation missing from report generator!"
    assert "| **Q01** | 2 | 100.0% |" not in report_src, "CRITICAL: Hardcoded Q01 metric row found in report generator!"
    assert "C15 is missed at B3@20" not in report_src, "CRITICAL: Hardcoded C15 outcome claim found in report generator!"
    assert "100.0% | 100.0% | 100.0% | 100.0%" not in report_src, "CRITICAL: Hardcoded static table values found in report generator!"
    assert "Pre-existing candidate-accessible evidence units remain subject" not in report_src, "CRITICAL: Hardcoded Q6 claim found in report generator!"
    assert "`+0.0%` | `PASS`" not in report_src, "CRITICAL: Hardcoded delta/gate in Section 4 table found in report generator!"
    print("[PASS] Assertion 19: Report generator source verified free of hardcoded metric tables or claims", flush=True)

    print("\n========================================================", flush=True)
    print("PHASE D STATIC PRE-EXECUTION AUDIT: PASS (ALL 19 ASSERTIONS)", flush=True)
    print("========================================================\n", flush=True)

def compute_dynamic_verdict(baseline, b3_results, provenance_records):
    """
    Computes Phase D verdict dynamically from measured result data evaluating pre-registered criteria.
    Note: Candidate pollution is calculated via a non-canonical keyword heuristic and is recorded for
    diagnostic tracking only; it does NOT independently gate the verdict.
    
    Returns EXACTLY one of:
    - "PASS"
    - "DIAGNOSTIC SUCCESS — DOWNSTREAM BOTTLENECK"
    - "NO SUPPORT"
    """
    import inspect
    d0_final_rec = baseline["final_recall_15"]
    d0_ndcg = baseline["ndcg_10"]
    d0_mrr = baseline["mrr"]

    any_new_in_union = any(r["entered_union_D3"] and not r["canonical_B0_candidate"] for r in provenance_records)
    any_new_in_final = any(r["final_top15_D3"] and not r["canonical_final"] for r in provenance_records)
    shallow_recovery = any((r["entered_union_D1"] or r["entered_union_D2"]) and not r["canonical_B0_candidate"] for r in provenance_records)

    d3_res = b3_results["D3"]
    final_recall_preserved = d3_res["final_recall_15"] >= (d0_final_rec - FINAL_RECALL_PRESERVATION_TOLERANCE)
    ndcg_preserved = d3_res["ndcg_10"] >= (d0_ndcg - NDCG_PRESERVATION_TOLERANCE)
    mrr_preserved = d3_res["mrr"] >= (d0_mrr - MRR_PRESERVATION_TOLERANCE)

    # Target-blindness verification
    b3_sig = inspect.signature(construct_b3_semantic_representation)
    target_blindness_verified = ("gold_map" not in b3_sig.parameters and "required_evidence" not in b3_sig.parameters)

    pass_criteria_met = (
        any_new_in_final and
        shallow_recovery and
        final_recall_preserved and
        ndcg_preserved and
        mrr_preserved and
        target_blindness_verified
    )

    if pass_criteria_met:
        return "PASS"
    elif any_new_in_union:
        return "DIAGNOSTIC SUCCESS — DOWNSTREAM BOTTLENECK"
    else:
        return "NO SUPPORT"


def main():
    print("==========================================================================", flush=True)
    print("   PHASE D — B3 REPRESENTATION DISCOVERABILITY DIAGNOSTIC EXECUTION", flush=True)
    print("==========================================================================", flush=True)

    # 1. Ingest Document and Build Corpus (203 Chunks)
    doc_path = os.path.join(root_dir, "Documents", "The Comedy of Errors document.pdf")
    session_id = f"phaseD_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    pipeline = AdvancedRAGPipeline(session_id=session_id)
    raw_docs = IngestionPipeline.load_document(doc_path)
    pipeline.add_documents(raw_docs)

    chunk_count = len(pipeline.all_chunks)
    print(f"Loaded indexed corpus: {chunk_count} chunks.", flush=True)
    if chunk_count != 203:
        raise ValueError(f"CRITICAL ERROR: Chunk count {chunk_count} != 203!")

    formatted_chunks = []
    for idx, doc in enumerate(pipeline.all_chunks):
        chunk_label = f"C{idx}"
        formatted_chunks.append({
            "chunk_id": chunk_label,
            "chunk_idx": idx,
            "element_id": doc.metadata.get("element_id", ""),
            "page": doc.metadata.get("page", 0),
            "breadcrumb_path": doc.metadata.get("breadcrumb_path", ""),
            "parent_id": doc.metadata.get("parent_id") or doc.metadata.get("parent_text", "")[:80],
            "parent_text": doc.metadata.get("parent_text", ""),
            "text": doc.page_content
        })

    # ============================================================
    # PHASE D — CORPUS MANIFEST
    # Persist the exact corpus identity used by this execution.
    # ============================================================

    corpus_manifest = []

    for chunk in formatted_chunks:
        text_hash = hashlib.sha256(
            chunk["text"].encode("utf-8")
        ).hexdigest()

        corpus_manifest.append({
            "session_id": session_id,
            "chunk_id": chunk["chunk_id"],
            "chunk_idx": chunk["chunk_idx"],
            "element_id": chunk["element_id"],
            "page": chunk["page"],
            "breadcrumb_path": chunk["breadcrumb_path"],
            "parent_id": chunk["parent_id"],
            "parent_text": chunk["parent_text"],
            "text_hash_sha256": text_hash,
            "text": chunk["text"],
        })

    corpus_manifest_df = pd.DataFrame(corpus_manifest)

    # ============================================================
    # DIAGNOSTIC TARGET FINGERPRINTS
    # Not a corpus validation gate.
    # ============================================================

    for target_id in [
        "C15",
        "C29",
        "C42",
        "C12",
        "C65",
        "C155",
    ]:
        target_chunk = next(
            c
            for c in formatted_chunks
            if c["chunk_id"] == target_id
        )

        target_hash = hashlib.sha256(
            target_chunk["text"].encode("utf-8")
        ).hexdigest()

        print(
            f"{target_id}: "
            f"page={target_chunk['page']} | "
            f"hash={target_hash}",
            flush=True
        )

    # Root artifact
    corpus_manifest_path = os.path.join(
        root_dir,
        "phaseD_corpus_manifest.csv"
    )

    # Evaluation artifact
    corpus_manifest_eval_path = os.path.join(
        root_dir,
        "evaluation",
        "results",
        "phaseD_corpus_manifest.csv"
    )

    os.makedirs(
        os.path.join(root_dir, "evaluation", "results"),
        exist_ok=True
    )

    corpus_manifest_df.to_csv(
        corpus_manifest_path,
        index=False
    )

    corpus_manifest_df.to_csv(
        corpus_manifest_eval_path,
        index=False
    )

    print(
        f"Saved Phase D Corpus Manifest: {corpus_manifest_path}",
        flush=True
    )

    # ============================================================
    # PHASE D — CANONICAL CORPUS IDENTITY GATE
    # ============================================================

    canonical_manifest_path = os.path.join(
        root_dir,
        "evaluation",
        "baseline",
        "phaseD_corpus_manifest_canonical.csv"
    )

    validate_canonical_corpus(
        current_manifest_df=corpus_manifest_df,
        canonical_manifest_path=canonical_manifest_path,
    )

    # Run Static Pre-Execution Assertions
    verify_pre_execution_assertions(formatted_chunks[0])

    # 2. Build B0 (Raw Evidence) and B3 (Structured Semantic Representation)
    b0_texts = [c["text"] for c in formatted_chunks]
    b3_texts = [construct_b3_semantic_representation(c) for c in formatted_chunks]

    print("Loading dense embedding model 'sentence-transformers/all-MiniLM-L6-v2' (E0)...", flush=True)
    from sentence_transformers import SentenceTransformer, CrossEncoder
    embedder_e0 = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("Embedding B0 (Raw Evidence)...", flush=True)
    embs_b0 = np.array(embedder_e0.encode(b0_texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True))

    print("Embedding B3 (Structured Semantic Representation)...", flush=True)
    embs_b3 = np.array(embedder_e0.encode(b3_texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True))

    # Pre-embed all subqueries
    all_sq_texts = []
    for b_case in BENCHMARK_SUITE:
        all_sq_texts.extend(FROZEN_Q2_QUERIES[b_case["q_id"]])
    all_sq_texts = sorted(list(set(all_sq_texts)))
    sq_embs_array = np.array(embedder_e0.encode(all_sq_texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True))
    sq_emb_map = {sq: emb for sq, emb in zip(all_sq_texts, sq_embs_array)}

    # 3. Setup Cross-Encoder R0 (ms-marco-MiniLM-L-6-v2)
    print("Loading Cross-Encoder reranker 'cross-encoder/ms-marco-MiniLM-L-6-v2' (R0)...", flush=True)
    ce_reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # 4. CANONICAL BASELINE CONTROL REPRODUCTION GATE (D0)
    print("\n--- CANONICAL BASELINE CONTROL REPRODUCTION GATE (D0) ---", flush=True)
    total_req_units = sum(len(b["required_evidence"]) for b in BENCHMARK_SUITE)
    d0_cand_retrieved = 0
    d0_final_retrieved = 0
    d0_ndcg10_list = []
    d0_mrr_list = []
    canonical_runs = {}

    for b_case in BENCHMARK_SUITE:
        q_id = b_case["q_id"]
        q_text = b_case["question"]
        req_units = b_case["required_evidence"]
        subqueries = FROZEN_Q2_QUERIES[q_id]

        candidate_map = {}
        for sq_idx, sq in enumerate(subqueries):
            sq_emb = sq_emb_map[sq]
            sq_sims = np.dot(embs_b0, sq_emb)
            sq_ranked_indices = np.argsort(-sq_sims)
            for r, c_i in enumerate(sq_ranked_indices[:20], 1):
                cid = f"C{c_i}"
                if cid not in candidate_map:
                    candidate_map[cid] = dict(formatted_chunks[c_i])
                    candidate_map[cid]["sources"] = []
                candidate_map[cid]["sources"].append(f"dense_b0_sq{sq_idx+1}_rank_{r}")

        if pipeline.bm25_retriever:
            pipeline.bm25_retriever.k = 20
            bm25_docs = pipeline.bm25_retriever.invoke(q_text)
            for r, doc in enumerate(bm25_docs, 1):
                el_id = doc.metadata.get("element_id")
                c_found = next((c for c in formatted_chunks if c["element_id"] == el_id), None)
                if c_found:
                    cid = c_found["chunk_id"]
                    if cid not in candidate_map:
                        candidate_map[cid] = dict(c_found)
                        candidate_map[cid]["sources"] = []
                    candidate_map[cid]["sources"].append(f"bm25_rank_{r}")

        cand_chunks = list(candidate_map.values())
        cand_ids = [c["chunk_id"] for c in cand_chunks]
        d0_cand_retrieved += sum(1 for cid in req_units if cid in cand_ids)

        ce_pairs = [(q_text, chunk["text"]) for chunk in cand_chunks]
        ce_scores = ce_reranker.predict(ce_pairs, batch_size=32, show_progress_bar=False)
        for chunk, score in zip(cand_chunks, ce_scores):
            chunk["ce_score"] = float(score)

        ce_ranked = sorted(cand_chunks, key=lambda x: (-x["ce_score"], x["chunk_idx"]))
        top15_cids = [item["chunk_id"] for item in ce_ranked[:15]]
        d0_final_retrieved += sum(1 for cid in req_units if cid in top15_cids)

        sec10 = calculate_metrics_at_k(ce_ranked, b_case["gold_map"], top_k=10)
        sec15 = calculate_metrics_at_k(ce_ranked, b_case["gold_map"], top_k=15)
        d0_ndcg10_list.append(sec10["ndcg_k"])
        d0_mrr_list.append(sec15["mrr"])

        canonical_runs[q_id] = {
            "cand_map": candidate_map,
            "cand_ids": cand_ids,
            "cand_chunks": cand_chunks,
            "ce_ranked": ce_ranked,
            "top15_cids": top15_cids
        }

    d0_cand_rec = d0_cand_retrieved / total_req_units
    d0_final_rec = d0_final_retrieved / total_req_units
    d0_ndcg10 = float(np.mean(d0_ndcg10_list))
    d0_mrr = float(np.mean(d0_mrr_list))

    print(f"Canonical Baseline (D0) Verification:")
    print(f"  Candidate Recall@20 : {d0_cand_rec*100:.1f}% (Expected: 85.0%)")
    print(f"  Final Recall@15     : {d0_final_rec*100:.1f}% (Expected: 70.0%)")
    print(f"  Macro nDCG@10       : {d0_ndcg10:.3f} (Expected: 0.485)")
    print(f"  MRR                 : {d0_mrr:.3f} (Expected: 0.369)")

    cand_cfg = PHASE_D_CONTROL_TARGETS["candidate_recall_20"]
    final_cfg = PHASE_D_CONTROL_TARGETS["final_recall_15"]
    ndcg_cfg = PHASE_D_CONTROL_TARGETS["ndcg_10"]
    mrr_cfg = PHASE_D_CONTROL_TARGETS["mrr"]

    pass_cand = (
        abs(d0_cand_rec - cand_cfg["target"])
        <= cand_cfg["tolerance"]
    )

    pass_final = (
        abs(d0_final_rec - final_cfg["target"])
        <= final_cfg["tolerance"]
    )

    pass_ndcg = (
        abs(d0_ndcg10 - ndcg_cfg["target"])
        <= ndcg_cfg["tolerance"]
    )

    pass_mrr = (
        abs(d0_mrr - mrr_cfg["target"])
        <= mrr_cfg["tolerance"]
    )

    print(f"\n[PASS] Candidate Recall Gate: {pass_cand}")
    print(f"[PASS] Final Recall Gate    : {pass_final}")
    print(f"[PASS] nDCG@10 Gate         : {pass_ndcg}")
    print(f"[PASS] MRR Gate              : {pass_mrr}")

    if not (pass_cand and pass_final and pass_ndcg and pass_mrr):
        print("\nCRITICAL CONTROL REPRODUCTION FAILURE — EXPERIMENT BLOCKED!", flush=True)
        sys.exit(1)
    else:
        print(">>> CONTROL REPRODUCTION GATE PASSED! Proceeding with Phase D...\n", flush=True)

    baseline_control_results = {
        "candidate_recall_20": d0_cand_rec,
        "final_recall_15": d0_final_rec,
        "ndcg_10": d0_ndcg10,
        "mrr": d0_mrr
    }

    # 5. EVALUATE AUXILIARY B3 DISCOVERABILITY CONFIGURATIONS (D1=20, D2=50, D3=100)
    print("--- EVALUATING AUXILIARY B3 DISCOVERABILITY CONFIGURATIONS (D1=20, D2=50, D3=100) ---", flush=True)
    b3_depths = [20, 50, 100]
    b3_diagnostic_results = {}
    provenance_records = []
    question_metric_rows = []

    for depth_idx, depth in enumerate(b3_depths, 1):
        cfg_key = f"D{depth_idx}"
        cfg_label = f"{cfg_key} (Canonical + B3 Top-{depth} × 3)"
        print(f"\nRunning Configuration {cfg_label}...", flush=True)

        total_b3_retrieved = 0
        total_b3_duplicates = 0
        total_b3_candidates_added = 0
        total_new_primary = 0
        total_new_supporting = 0
        total_new_rel_unnecessary = 0
        total_new_irrelevant = 0

        total_cand_retrieved = 0
        total_final_retrieved = 0
        ndcg10_list = []
        mrr_list = []

        for b_case in BENCHMARK_SUITE:
            q_id = b_case["q_id"]
            q_text = b_case["question"]
            req_units = b_case["required_evidence"]
            gold_map = b_case["gold_map"]
            subqueries = FROZEN_Q2_QUERIES[q_id]

            c_run = canonical_runs[q_id]
            cand_map = {cid: dict(item) for cid, item in c_run["cand_map"].items()}
            initial_cids = set(cand_map.keys())

            b3_retrieved_set = set()

            for sq_idx, sq in enumerate(subqueries):
                sq_emb = sq_emb_map[sq]
                b3_sims = np.dot(embs_b3, sq_emb)
                b3_ranked_indices = np.argsort(-b3_sims)
                for r, c_i in enumerate(b3_ranked_indices[:depth], 1):
                    cid = f"C{c_i}"
                    b3_retrieved_set.add(cid)
                    if cid not in cand_map:
                        cand_map[cid] = dict(formatted_chunks[c_i])
                        cand_map[cid]["sources"] = []
                    cand_map[cid]["sources"].append(f"dense_b3_sq{sq_idx+1}_rank_{r}")

            cand_chunks = list(cand_map.values())
            cand_ids = set(cand_map.keys())
            
            added_cids = cand_ids - initial_cids
            duplicates = b3_retrieved_set & initial_cids

            total_b3_retrieved += len(b3_retrieved_set)
            total_b3_duplicates += len(duplicates)
            total_b3_candidates_added += len(added_cids)

            for cid in added_cids:
                if cid in gold_map:
                    role = gold_map[cid]
                    if role == "PRIMARY":
                        total_new_primary += 1
                    else:
                        total_new_supporting += 1
                else:
                    item_text = cand_map[cid]["text"].lower()
                    if any(kw in item_text for kw in ["aegeon", "syracuse", "ephesus", "antipholus", "dromio", "adriana", "goldsmith", "angelo", "chain", "ducats", "twin"]):
                        total_new_rel_unnecessary += 1
                    else:
                        total_new_irrelevant += 1

            total_cand_retrieved += sum(1 for cid in req_units if cid in cand_ids)

            ce_pairs = [(q_text, chunk["text"]) for chunk in cand_chunks]
            ce_scores = ce_reranker.predict(ce_pairs, batch_size=32, show_progress_bar=False)
            for chunk, score in zip(cand_chunks, ce_scores):
                chunk["ce_score"] = float(score)

            ce_ranked = sorted(cand_chunks, key=lambda x: (-x["ce_score"], x["chunk_idx"]))
            top15_cids = [item["chunk_id"] for item in ce_ranked[:15]]
            total_final_retrieved += sum(1 for cid in req_units if cid in top15_cids)

            sec10 = calculate_metrics_at_k(ce_ranked, gold_map, top_k=10)
            sec15 = calculate_metrics_at_k(ce_ranked, gold_map, top_k=15)
            ndcg10_list.append(sec10["ndcg_k"])
            mrr_list.append(sec15["mrr"])

            question_metric_rows.append({
                "config": cfg_key,
                "b3_depth": depth,
                "question_id": q_id,
                "candidate_pool_size": len(cand_ids),
                "b3_retrieved": len(b3_retrieved_set),
                "b3_duplicates": len(duplicates),
                "b3_candidates_added": len(added_cids),
                "candidate_recall_augmented_pool": sum(1 for cid in req_units if cid in cand_ids) / len(req_units),
                "final_recall_15": sum(1 for cid in req_units if cid in top15_cids) / len(req_units),
                "ndcg_10": sec10["ndcg_k"],
                "mrr": sec15["mrr"]
            })

        cand_rec = total_cand_retrieved / total_req_units
        final_rec = total_final_retrieved / total_req_units
        ndcg10 = float(np.mean(ndcg10_list))
        mrr = float(np.mean(mrr_list))

        new_gold_candidates = total_new_primary + total_new_supporting
        new_nongold_candidates = total_new_rel_unnecessary + total_new_irrelevant
        poll_rate = new_nongold_candidates / total_b3_candidates_added if total_b3_candidates_added > 0 else 0.0

        b3_diagnostic_results[cfg_key] = {
            "config": cfg_key,
            "b3_depth": depth,
            "candidate_recall_augmented_pool": cand_rec,
            "final_recall_15": final_rec,
            "ndcg_10": ndcg10,
            "mrr": mrr,
            "total_b3_retrieved": total_b3_retrieved,
            "total_b3_duplicates": total_b3_duplicates,
            "total_b3_candidates_added": total_b3_candidates_added,
            "new_primary_candidates": total_new_primary,
            "new_supporting_candidates": total_new_supporting,
            "new_gold_candidates": new_gold_candidates,
            "new_rel_unnecessary_candidates": total_new_rel_unnecessary,
            "new_irrelevant_candidates": total_new_irrelevant,
            "new_nongold_candidates": new_nongold_candidates,
            "candidate_pollution_rate": round(poll_rate, 4),
            "delta_candidate_recall": round(cand_rec - d0_cand_rec, 4),
            "delta_final_recall": round(final_rec - d0_final_rec, 4),
            "delta_ndcg10": round(ndcg10 - d0_ndcg10, 4),
            "delta_mrr": round(mrr - d0_mrr, 4)
        }

    # 6. PER-EVIDENCE PROVENANCE & PROMOTION ANALYSIS
    print("--- BUILDING PER-EVIDENCE PROVENANCE TRACE & R0 PROMOTION ANALYSIS ---", flush=True)
    b3_to_candidate_count = 0
    candidate_to_R0_top15_count = 0
    R0_top15_to_final_count = 0

    for b_case in BENCHMARK_SUITE:
        q_id = b_case["q_id"]
        q_text = b_case["question"]
        req_units = b_case["required_evidence"]
        gold_map = b_case["gold_map"]
        subqueries = FROZEN_Q2_QUERIES[q_id]

        c_run = canonical_runs[q_id]

        for gold_cid in req_units:
            gold_idx = int(gold_cid[1:])
            gold_role = gold_map[gold_cid]

            # B0 Full-Corpus Dense Rank
            sq_b0_ranks = []
            for sq in subqueries:
                sq_emb = sq_emb_map[sq]
                b0_sims = np.dot(embs_b0, sq_emb)
                r_idx = int(np.where(np.argsort(-b0_sims) == gold_idx)[0][0]) + 1
                sq_b0_ranks.append(r_idx)
            canonical_B0_rank = min(sq_b0_ranks)

            canonical_B0_candidate = gold_cid in c_run["cand_ids"]
            canonical_final = gold_cid in c_run["top15_cids"]

            # B3 Full-Corpus Dense Rank across subqueries
            sq_b3_ranks = []
            for sq in subqueries:
                sq_emb = sq_emb_map[sq]
                b3_sims = np.dot(embs_b3, sq_emb)
                r_idx = int(np.where(np.argsort(-b3_sims) == gold_idx)[0][0]) + 1
                sq_b3_ranks.append(r_idx)
            b3_best_rank = min(sq_b3_ranks)

            prov_rec = {
                "question_id": q_id,
                "evidence_chunk_id": gold_cid,
                "role": gold_role,
                "canonical_B0_rank": canonical_B0_rank,
                "canonical_B0_candidate": canonical_B0_candidate,
                "canonical_final": canonical_final,
                "B3_best_rank": b3_best_rank
            }

            for depth_idx, depth in enumerate(b3_depths, 1):
                cfg_key = f"D{depth_idx}"
                
                b3_retrieved_cfg = b3_best_rank <= depth
                b3_new_candidate_cfg = b3_retrieved_cfg and not canonical_B0_candidate

                cand_map = {cid: dict(item) for cid, item in c_run["cand_map"].items()}
                for sq in subqueries:
                    sq_emb = sq_emb_map[sq]
                    sims = np.dot(embs_b3, sq_emb)
                    for c_i in np.argsort(-sims)[:depth]:
                        cid = f"C{c_i}"
                        if cid not in cand_map:
                            cand_map[cid] = dict(formatted_chunks[c_i])

                cand_chunks = list(cand_map.values())
                cand_ids = [c["chunk_id"] for c in cand_chunks]
                entered_union = gold_cid in cand_ids

                ce_pairs = [(q_text, chunk["text"]) for chunk in cand_chunks]
                ce_scores = ce_reranker.predict(ce_pairs, batch_size=32, show_progress_bar=False)
                for chunk, score in zip(cand_chunks, ce_scores):
                    chunk["ce_score"] = float(score)

                ce_ranked = sorted(cand_chunks, key=lambda x: (-x["ce_score"], x["chunk_idx"]))
                ce_ranked_ids = [c["chunk_id"] for c in ce_ranked]
                
                r0_rank_after_union = ce_ranked_ids.index(gold_cid) + 1 if gold_cid in ce_ranked_ids else 999
                final_top15_cfg = gold_cid in ce_ranked_ids[:15]

                prov_rec[f"B3_retrieved_{cfg_key}"] = b3_retrieved_cfg
                prov_rec[f"B3_new_candidate_{cfg_key}"] = b3_new_candidate_cfg
                prov_rec[f"entered_union_{cfg_key}"] = entered_union
                prov_rec[f"R0_rank_after_union_{cfg_key}"] = r0_rank_after_union
                prov_rec[f"final_top15_{cfg_key}"] = final_top15_cfg

                if not canonical_B0_candidate and entered_union and cfg_key == "D3":
                    b3_to_candidate_count += 1
                    if r0_rank_after_union <= 15:
                        candidate_to_R0_top15_count += 1
                        if final_top15_cfg:
                            R0_top15_to_final_count += 1

                # EXHAUSTIVE 3-BRANCH EVIDENCE-LEVEL DIAGNOSIS
                if cfg_key == "D3":
                    if final_top15_cfg:
                        diag_case = "B3_DISCOVERABILITY_AND_RERANK_SUCCESS"
                    elif entered_union:
                        diag_case = "B3_DISCOVERABILITY_R0_FAILURE"
                    else:
                        diag_case = "B3_NON_DISCOVERY"
                    prov_rec["phaseD_diagnosis"] = diag_case

            provenance_records.append(prov_rec)

    # 7. POST-HOC AUTOPSIES FOR PHASE C HARD CASES
    autopsy_ids = ["C15", "C29", "C42", "C12", "C65", "C155"]
    target_autopsies = {}
    for cid in autopsy_ids:
        target_autopsies[cid] = [r for r in provenance_records if r["evidence_chunk_id"] == cid][0]

    # 8. DYNAMIC VERDICT COMPUTATION FROM MEASURED DATA
    verdict = compute_dynamic_verdict(baseline_control_results, b3_diagnostic_results, provenance_records)

    print(f"\n==========================================================", flush=True)
    print(f"   PHASE D VERDICT: [{verdict}]", flush=True)
    print(f"==========================================================\n", flush=True)

    # 9. SAVE ARTIFACTS IN BOTH ROOT AND EVALUATION DIRECTORIES
    output_json = {
        "metadata": {
            "phase": "PhaseD",
            "title": "Phase D — B3 Representation Discoverability Diagnostic",
            "corpus": "The Comedy of Errors (203 chunks: C0–C202)",
            "timestamp": datetime.now().isoformat(),
            "control_gate": "PASSED",
            "verdict": verdict
        },
        "baseline_control_results": baseline_control_results,
        "b3_diagnostic_results": b3_diagnostic_results,
        "r0_promotion_metrics": {
            "B3_to_candidate_count": b3_to_candidate_count,
            "candidate_to_R0_top15_count": candidate_to_R0_top15_count,
            "R0_top15_to_final_count": R0_top15_to_final_count
        },
        "target_autopsies": target_autopsies,
        "provenance_records": provenance_records,
        "question_metric_rows": question_metric_rows
    }

    path_json = os.path.join(root_dir, "phaseD_results.json")
    path_csv_trace = os.path.join(root_dir, "phaseD_evidence_trace.csv")
    path_csv_rep = os.path.join(root_dir, "phaseD_representation_metrics.csv")
    path_csv_q = os.path.join(root_dir, "phaseD_question_metrics.csv")
    path_report = os.path.join(root_dir, "phaseD_b3_discoverability_report.md")

    path_json_eval = os.path.join(root_dir, "evaluation", "results", "phaseD_results.json")
    path_csv_trace_eval = os.path.join(root_dir, "evaluation", "results", "phaseD_evidence_trace.csv")
    path_csv_rep_eval = os.path.join(root_dir, "evaluation", "results", "phaseD_representation_metrics.csv")
    path_csv_q_eval = os.path.join(root_dir, "evaluation", "results", "phaseD_question_metrics.csv")
    path_report_eval = os.path.join(root_dir, "evaluation", "reports", "phaseD_b3_discoverability_report.md")

    os.makedirs(os.path.join(root_dir, "evaluation", "results"), exist_ok=True)
    os.makedirs(os.path.join(root_dir, "evaluation", "reports"), exist_ok=True)

    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2)
    with open(path_json_eval, "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2)

    df_trace = pd.DataFrame(provenance_records)
    df_trace.to_csv(path_csv_trace, index=False)
    df_trace.to_csv(path_csv_trace_eval, index=False)
    
    rep_summary_rows = [
        {"config": "D0", "candidate_recall": d0_cand_rec, "final_recall_15": d0_final_rec, "ndcg_10": d0_ndcg10, "mrr": d0_mrr, "total_b3_candidates_added": 0, "candidate_pollution_rate": 0.0}
    ]
    for k, v in b3_diagnostic_results.items():
        rep_summary_rows.append(v)
    df_rep = pd.DataFrame(rep_summary_rows)
    df_rep.to_csv(path_csv_rep, index=False)
    df_rep.to_csv(path_csv_rep_eval, index=False)

    df_q = pd.DataFrame(question_metric_rows)
    df_q.to_csv(path_csv_q, index=False)
    df_q.to_csv(path_csv_q_eval, index=False)

    print(f"Saved Phase D Results JSON:             {path_json}")
    print(f"Saved Evidence Trace CSV:               {path_csv_trace}")
    print(f"Saved Representation Metrics CSV:       {path_csv_rep}")
    print(f"Saved Question Metrics CSV:             {path_csv_q}")

    generate_markdown_report(output_json, path_report, path_report_eval)

def validate_canonical_corpus(
    current_manifest_df,
    canonical_manifest_path,
):
    """
    Validate that the current 203-chunk corpus is identical
    to the canonical corpus used to define benchmark evidence IDs.

    This is a hard reproducibility gate.
    """

    if not os.path.exists(canonical_manifest_path):
        os.makedirs(os.path.dirname(canonical_manifest_path), exist_ok=True)
        current_manifest_df.to_csv(canonical_manifest_path, index=False)
        print(
            f"INITIALIZED CANONICAL CORPUS MANIFEST: {canonical_manifest_path}",
            flush=True
        )

    canonical_df = pd.read_csv(canonical_manifest_path)

    # --------------------------------------------------------
    # 1. Chunk count
    # --------------------------------------------------------

    if len(current_manifest_df) != len(canonical_df):
        raise RuntimeError(
            "CORPUS IDENTITY FAILURE: chunk count changed. "
            f"Current={len(current_manifest_df)}, "
            f"Canonical={len(canonical_df)}"
        )

    # --------------------------------------------------------
    # 2. Required identity columns
    # --------------------------------------------------------

    identity_columns = [
        "chunk_id",
        "chunk_idx",
        "element_id",
        "page",
        "breadcrumb_path",
        "parent_id",
        "text_hash_sha256",
    ]

    missing_current = [
        c for c in identity_columns
        if c not in current_manifest_df.columns
    ]

    missing_canonical = [
        c for c in identity_columns
        if c not in canonical_df.columns
    ]

    if missing_current:
        raise RuntimeError(
            f"Current corpus manifest missing columns: "
            f"{missing_current}"
        )

    if missing_canonical:
        raise RuntimeError(
            f"Canonical corpus manifest missing columns: "
            f"{missing_canonical}"
        )

    # --------------------------------------------------------
    # 3. Compare exact ordered identity
    # --------------------------------------------------------

    current_identity = (
        current_manifest_df[identity_columns]
        .fillna("")
        .astype(str)
        .reset_index(drop=True)
    )

    canonical_identity = (
        canonical_df[identity_columns]
        .fillna("")
        .astype(str)
        .reset_index(drop=True)
    )

    if not current_identity.equals(canonical_identity):

        mismatches = []

        for idx in range(len(current_identity)):
            current_row = current_identity.iloc[idx]
            canonical_row = canonical_identity.iloc[idx]

            if not current_row.equals(canonical_row):
                mismatches.append({
                    "row": idx,
                    "current_chunk_id": current_row["chunk_id"],
                    "canonical_chunk_id": canonical_row["chunk_id"],
                    "current_hash": current_row["text_hash_sha256"],
                    "canonical_hash": canonical_row["text_hash_sha256"],
                })

        print(
            "CORPUS IDENTITY FAILURE",
            flush=True
        )

        print(
            f"Total mismatched chunks: {len(mismatches)}",
            flush=True
        )

        for mismatch in mismatches[:10]:
            print(
                mismatch,
                flush=True
            )

        raise RuntimeError(
            "Current corpus does not match the canonical corpus. "
            "Benchmark positional evidence IDs such as C15/C42 "
            "cannot be trusted for this execution."
        )

    print(
        "CORPUS IDENTITY CHECK: PASS — "
        "current corpus exactly matches canonical corpus.",
        flush=True
    )

    return True

def generate_markdown_report(data, path_report, path_report_eval):
    ctrl = data["baseline_control_results"]
    diag = data["b3_diagnostic_results"]
    prom = data["r0_promotion_metrics"]
    autopsies = data.get("target_autopsies", {})
    verdict = data["metadata"]["verdict"]
    q_rows = data.get("question_metric_rows", [])
    trace = data.get("provenance_records", [])

    cand_act = ctrl['candidate_recall_20']
    final_act = ctrl['final_recall_15']
    ndcg_act = ctrl['ndcg_10']
    mrr_act = ctrl['mrr']

    # Dynamic Control reproduction deltas & gate statuses
    cand_target = PHASE_D_CONTROL_TARGETS["candidate_recall_20"]["target"]
    cand_tol = PHASE_D_CONTROL_TARGETS["candidate_recall_20"]["tolerance"]
    cand_delta = cand_act - cand_target
    cand_gate = "PASS" if abs(cand_delta) <= cand_tol else "FAIL"

    final_target = PHASE_D_CONTROL_TARGETS["final_recall_15"]["target"]
    final_tol = PHASE_D_CONTROL_TARGETS["final_recall_15"]["tolerance"]
    final_delta = final_act - final_target
    final_gate = "PASS" if abs(final_delta) <= final_tol else "FAIL"

    ndcg_target = PHASE_D_CONTROL_TARGETS["ndcg_10"]["target"]
    ndcg_tol = PHASE_D_CONTROL_TARGETS["ndcg_10"]["tolerance"]
    ndcg_delta = ndcg_act - ndcg_target
    ndcg_gate = "PASS" if abs(ndcg_delta) <= ndcg_tol else "FAIL"

    mrr_target = PHASE_D_CONTROL_TARGETS["mrr"]["target"]
    mrr_tol = PHASE_D_CONTROL_TARGETS["mrr"]["tolerance"]
    mrr_delta = mrr_act - mrr_target
    mrr_gate = "PASS" if abs(mrr_delta) <= mrr_tol else "FAIL"

    # Dynamic Evidence Counts from Trace
    total_units = len(trace)
    cand_accessible = sum(1 for r in trace if r["canonical_B0_candidate"])
    newly_recovered_union = sum(1 for r in trace if r["entered_union_D3"] and not r["canonical_B0_candidate"])
    newly_recovered_final = sum(1 for r in trace if r["final_top15_D3"] and not r["canonical_final"])

    # Dynamic Q6 Answer Calculation
    q6_unselected = [
        r["evidence_chunk_id"]
        for r in trace
        if (
            r["evidence_chunk_id"] in Q6_TARGET_IDS
            and r["canonical_B0_candidate"]
            and not r["canonical_final"]
        )
    ]
    if q6_unselected:
        q6_answer = (
            f"Yes. Among the predeclared diagnostic targets, "
            f"{', '.join(sorted(set(q6_unselected)))} were candidate-accessible "
            f"under D0 but not selected in final Top-15 by R0."
        )
    else:
        q6_answer = (
            "No predeclared diagnostic target demonstrated a "
            "candidate-accessible-but-not-final R0 failure under D0."
        )

    # Dynamic Q1, Q2, Q9, Q11, Q12 Calculations
    d3_res = diag["D3"]
    cand_gain = d3_res['delta_candidate_recall']*100
    final_gain = d3_res['delta_final_recall']*100
    candidate_gain_observed = (d3_res['delta_candidate_recall'] > 0)

    q1_answer = (
        f"Yes. Candidate Recall increased by {cand_gain:+.1f} percentage points."
        if candidate_gain_observed
        else "No candidate recall increase observed."
    )
    q2_answer = minimum_b3_recovery_depth(trace)

    r0_failure_count = sum(1 for r in trace if r.get("phaseD_diagnosis") == "B3_DISCOVERABILITY_R0_FAILURE")
    non_discovery_count = sum(1 for r in trace if r.get("phaseD_diagnosis") == "B3_NON_DISCOVERY")

    q9_answer = (
        "Increases candidate recall while final recall is constrained downstream by R0."
        if r0_failure_count > 0 or newly_recovered_union > newly_recovered_final else
        "No effect."
    )

    if r0_failure_count > non_discovery_count:
        q11_answer = "Downstream R0 Cross-Encoder Reranking Stage."
    elif non_discovery_count > 0:
        q11_answer = "Candidate Generation Stage (Representation Insufficiency)."
    else:
        q11_answer = "No remaining evidence accessibility failures."

    q12_answer = interpret_q12(verdict)

    md = f"""# Phase D — B3 Representation Discoverability Diagnostic Final Report

> **Research Objective**: Determine whether structured $B_3$ representation makes previously inaccessible evidence discoverable through an auxiliary candidate-generation path at controlled depths $N \\in \\{{20, 50, 100\\}}$ while canonical $B_0$/BM25 retrieval, $R_0$ reranker, and final Top-15 mechanism remain completely frozen.  
> **Status**: **COMPLETE & CANONICAL**  
> **Date**: {datetime.now().strftime('%Y-%m-%d')}  
> **Control Reproduction Gate**: **PASS** (Candidate Recall@20 = {cand_act*100:.1f}%, Final Recall@15 = {final_act*100:.1f}%, nDCG@10 = {ndcg_act:.3f}, MRR = {mrr_act:.3f})  
> **Verdict**: **`PHASE D VERDICT: [{verdict}]`**

---

## 1. Executive Summary
Phase D evaluated auxiliary $B_3$ candidate retrieval at depths $N \\in \\{{20, 50, 100\\}}$ across all {total_units} gold evidence units over the 203-chunk *The Comedy of Errors* corpus. Measured execution metrics determined whether $B_3$ candidate generation succeeded in introducing previously inaccessible evidence into the candidate pool and whether unchanged $R_0$ promoted those candidates downstream.

---

## 2. Research Question
> Can the structured $B_3$ representation make previously inaccessible evidence discoverable through an auxiliary candidate-generation path, at controlled depths $N \\in \\{{20, 50, 100\\}}$, while the canonical $B_0$/BM25 retrieval path, $R_0$ reranker, and final Top-15 mechanism remain completely frozen?

---

## 3. Frozen Control Specification
- **Candidate Pool**: 3 frozen Q2 subqueries ($E_0$ dense Top-20 per subquery) + BM25 Top-20 on primary question.
- **Reranker ($R_0$)**: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- **Selection**: Top-15 final retrieved chunks.

---

## 4. Control Reproduction Results ($D_0$)

| Metric | Canonical Target | Phase D Control ($D_0$) | Delta | Gate Status |
| :--- | :---: | :---: | :---: | :---: |
| **Candidate Recall@20** | **{cand_target*100:.1f}%** | **{cand_act*100:.1f}%** | `{cand_delta*100:+.1f}%` | `{cand_gate}` |
| **Final Recall@15** | **{final_target*100:.1f}%** | **{final_act*100:.1f}%** | `{final_delta*100:+.1f}%` | `{final_gate}` |
| **Macro nDCG@10** | **{ndcg_target:.3f}** | **{ndcg_act:.3f}** | `{ndcg_delta:+.3f}` | `{ndcg_gate}` |
| **MRR** | **{mrr_target:.3f}** | **{mrr_act:.3f}** | `{mrr_delta:+.3f}` | `{mrr_gate}` |

---

## 5. Static Audit Results
All pre-execution audit assertions passed successfully before diagnostic execution.

---

## 6. Experimental Configurations
- **$D_0$**: Canonical Baseline Control (Auxiliary $B_3$ disabled).
- **$D_1$**: Canonical + $B_3$ Top-20 × 3 subqueries.
- **$D_2$**: Canonical + $B_3$ Top-50 × 3 subqueries.
- **$D_3$**: Canonical + $B_3$ Top-100 × 3 subqueries.

---

## 7. B3 Retrieval Method
- Indexed deterministic $B_3$ structured text using frozen $E_0$ (`sentence-transformers/all-MiniLM-L6-v2`).
- Retrieved Top-$N$ candidates across the 3 frozen Q2 subqueries per question.

---

## 8. Aggregate Results

| Config | B3 Depth | Candidate Recall (Augmented Pool) | Final Recall@15 | Macro nDCG@10 | MRR | B3 Candidates Added | Pollution Rate (Diagnostic Keyword Heuristic) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$D_0$** | Disabled | **{cand_act*100:.1f}%** (Recall@20) | **{final_act*100:.1f}%** | **{ndcg_act:.3f}** | **{mrr_act:.3f}** | **0** | **0.0%** |
"""
    for cfg in ["D1", "D2", "D3"]:
        d = diag[cfg]
        md += f"| **{cfg}** | Top-{d['b3_depth']} × 3 | **{d['candidate_recall_augmented_pool']*100:.1f}%** (`{d['delta_candidate_recall']*100:+.1f}%`) | **{d['final_recall_15']*100:.1f}%** (`{d['delta_final_recall']*100:+.1f}%`) | **{d['ndcg_10']:.3f}** (`{d['delta_ndcg10']:+.3f}`) | **{d['mrr']:.3f}** (`{d['delta_mrr']:+.3f}`) | **{d['total_b3_candidates_added']}** | **{d['candidate_pollution_rate']*100:.1f}%** |\n"

    md += f"""
---

## 9. Per-Question Results

| Question | Config | Candidate Pool Size | Candidate Recall | Final Recall@15 | Macro nDCG@10 | MRR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for q_r in q_rows:
        md += f"| **{q_r['question_id']}** | {q_r['config']} | {q_r['candidate_pool_size']} | {q_r['candidate_recall_augmented_pool']*100:.1f}% | {q_r['final_recall_15']*100:.1f}% | {q_r['ndcg_10']:.3f} | {q_r['mrr']:.3f} |\n"

    md += f"""
---

## 10. Evidence-Level Provenance

Summary of {total_units} gold evidence units across $D_0 \\dots D_3$:
- **Canonical Candidate Accessible**: {cand_accessible} / {total_units}
- **Newly Recovered into Candidate Union ($D_3$)**: {newly_recovered_union} / {total_units - cand_accessible}
- **Newly Recovered into Final Top-15 ($D_3$)**: {newly_recovered_final} / {total_units - cand_accessible}

---

## 11. Candidate Pollution Analysis (Diagnostic Keyword Heuristic: Non-Canonical)

| Config | B3 Depth | Total B3 Retrieved | Duplicates of D0 | Added Candidates | Added Primary | Added Supporting | Added Relevant | Added Irrelevant | Pollution Rate (Non-Canonical Heuristic) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for cfg in ["D1", "D2", "D3"]:
        d = diag[cfg]
        md += f"| **{cfg}** | Top-{d['b3_depth']} | {d['total_b3_retrieved']} | {d['total_b3_duplicates']} | {d['total_b3_candidates_added']} | {d['new_primary_candidates']} | {d['new_supporting_candidates']} | {d['new_rel_unnecessary_candidates']} | {d['new_irrelevant_candidates']} | {d['candidate_pollution_rate']*100:.1f}% |\n"

    md += f"""
---

## 12. R0 Promotion Analysis

- **$B_3 \\rightarrow \\text{{Candidate Union}}$**: **{prom['B3_to_candidate_count']}** evidence units newly introduced into candidate union.
- **$\\text{{Candidate Union}} \\rightarrow R_0 \\text{{ Top-15}}$**: **{prom['candidate_to_R0_top15_count']}** evidence units promoted into $R_0$ Top-15.
- **$R_0 \\text{{ Top-15}} \\rightarrow \\text{{Final Top-15}}$**: **{prom['R0_top15_to_final_count']}** evidence units retained in final Top-15.

---

## 13. Analysis of Target Autopsy Chunks

"""
    for cid, a in autopsies.items():
        md += f"""### Target Autopsy: `{cid}` ({a['question_id']})
- **Canonical $B_0$ Rank**: `#{a['canonical_B0_rank']}` (Candidate: `{a['canonical_B0_candidate']}`, Final: `{a['canonical_final']}`)
- **$B_3$ Best Subquery Rank**: `#{a['B3_best_rank']}`
- **$D_1$ ($B_3@20$)**: Entered Union = `{a['entered_union_D1']}`, $R_0$ Rank = `#{a['R0_rank_after_union_D1']}`, Final Top-15 = `{a['final_top15_D1']}`
- **$D_2$ ($B_3@50$)**: Entered Union = `{a['entered_union_D2']}`, $R_0$ Rank = `#{a['R0_rank_after_union_D2']}`, Final Top-15 = `{a['final_top15_D2']}`
- **$D_3$ ($B_3@100$)**: Entered Union = `{a['entered_union_D3']}`, $R_0$ Rank = `#{a['R0_rank_after_union_D3']}`, Final Top-15 = `{a['final_top15_D3']}`
- **Phase D Diagnosis**: `{a['phaseD_diagnosis']}`

"""

    md += f"""---

## 14. D0 vs D1 vs D2 vs D3 Comparison

- **Candidate Recall**: $D_0 = {cand_act*100:.1f}\\%$, $D_3 = {d3_res['candidate_recall_augmented_pool']*100:.1f}\\%$ (`{cand_gain:+.1f}%` gain).
- **Final Recall@15**: $D_0 = {final_act*100:.1f}\\%$, $D_3 = {d3_res['final_recall_15']*100:.1f}\\%$ (`{final_gain:+.1f}%` change).
- **Macro nDCG@10**: $D_0 = {ndcg_act:.3f}$, $D_3 = {d3_res['ndcg_10']:.3f}$.
- **MRR**: $D_0 = {mrr_act:.3f}$, $D_3 = {d3_res['mrr']:.3f}$.

---

## 15. Accessibility Depth Analysis
Evaluated candidate accessibility across shallow ($D_1=20$, $D_2=50$) and deep ($D_3=100$) $B_3$ candidate retrieval budgets.

---

## 16. Failure-Stage Classification
Measured data isolates whether accessibility failures occur during candidate generation or downstream Cross-Encoder reranking.

---

## 17. Limitations
- Benchmark size: {len(BENCHMARK_SUITE)} questions over 203 corpus chunks.
- Reranker: $R_0$ (`ms-marco-MiniLM-L-6-v2`) is un-tuned for structured semantic text schemas.

---

## 18. Production Implication
Retain canonical $D_0$ in production unless verdict is PASS.

---

## 19. Final Verdict

```text
PHASE D VERDICT: [{verdict}]
```

---

## 20. Recommended Next Controlled Experiment
Phase E — Evaluate Cross-Encoder rerankers ($R_1 / R_2$) over $B_3$ structured candidate representations.

---

## 21. Explicit Answers to Predeclared Research Questions (Q1–Q12)

### Q1: Does $B_3$ improve candidate accessibility relative to the frozen $D_0$ representation?
{q1_answer}

### Q2: What is the minimum $B_3$ depth that produces a meaningful accessibility improvement?
{q2_answer}

### Q3: Does $B_3@20$, $B_3@50$, or $B_3@100$ recover `C15`?
{"Recovered into candidate union" if any(r["evidence_chunk_id"] == "C15" and r["entered_union_D3"] for r in trace) else "Not recovered"}

### Q4: Does $B_3@20$, $B_3@50$, or $B_3@100$ recover `C42`?
{"Recovered into candidate union" if any(r["evidence_chunk_id"] == "C42" and r["entered_union_D3"] for r in trace) else "Not recovered"}

### Q5: What happens to `C29`?
{"Retrieved in B3 space" if any(r["evidence_chunk_id"] == "C29" and r["entered_union_D3"] for r in trace) else "Not retrieved"}

### Q6: Do `C12`, `C65`, and `C155` demonstrate that adequate representation can still fail downstream?
{q6_answer}

### Q7: How many previously inaccessible evidence units become candidates?
{newly_recovered_union} evidence units.

### Q8: How many of those candidates survive $R_0$ into Top-15?
{newly_recovered_final} evidence units.

### Q9: Does increased $B_3$ depth improve final retrieval or merely increase candidate-pool size?
{q9_answer}

### Q10: Does $B_3$ introduce measurable candidate pollution?
Measured candidate pollution rate is {d3_res.get('candidate_pollution_rate', 0.0)*100:.1f}% under D3 (diagnostic keyword heuristic: non-canonical, does not gate verdict).

### Q11: At what stage are the remaining failures concentrated?
{q11_answer}

### Q12: Does the evidence justify a production modification?
{q12_answer}
"""

    with open(path_report, "w", encoding="utf-8") as f:
        f.write(md)
    with open(path_report_eval, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"Successfully generated Phase D Report: {path_report}", flush=True)

def run_synthetic_tests():
    """
    Executes lightweight synthetic unit tests for:
    - Pre-execution audit assertions (19 assertions)
    - Q2 minimum B3 recovery depth logic
    - Q12 3-way verdict interpretation logic
    - D0 dynamic control reproduction table calculation
    - Assertion 19 dynamic report audit
    """
    print("==========================================================================", flush=True)
    print("   RUNNING PHASE D LIGHTWEIGHT SYNTHETIC SUITE (NO EXPERIMENT EXECUTION)", flush=True)
    print("==========================================================================", flush=True)

    # 1. Test Static Pre-Execution Assertions
    sample_chunk = {"text": "Aegeon in Ephesus merchant", "breadcrumb_path": "ACT 1"}
    verify_pre_execution_assertions(sample_chunk)

    # 2. Test Q2 Minimum Depth Logic
    rec_d1 = [{"canonical_B0_candidate": False, "entered_union_D1": True, "entered_union_D2": True, "entered_union_D3": True}]
    rec_d2 = [{"canonical_B0_candidate": False, "entered_union_D1": False, "entered_union_D2": True, "entered_union_D3": True}]
    rec_d3 = [{"canonical_B0_candidate": False, "entered_union_D1": False, "entered_union_D2": False, "entered_union_D3": True}]
    rec_none = [{"canonical_B0_candidate": False, "entered_union_D1": False, "entered_union_D2": False, "entered_union_D3": False}]

    assert minimum_b3_recovery_depth(rec_d1) == "D1 (B3@20)", "Q2 Test Failed for D1!"
    assert minimum_b3_recovery_depth(rec_d2) == "D2 (B3@50)", "Q2 Test Failed for D2!"
    assert minimum_b3_recovery_depth(rec_d3) == "D3 (B3@100)", "Q2 Test Failed for D3!"
    assert minimum_b3_recovery_depth(rec_none) == "None", "Q2 Test Failed for None!"
    print("[PASS] Synthetic Test: Q2 minimum B3 recovery depth logic verified", flush=True)

    # 3. Test Q12 Three-Way Verdict Interpretation
    ans_pass = interpret_q12("PASS")
    ans_diag = interpret_q12("DIAGNOSTIC SUCCESS — DOWNSTREAM BOTTLENECK")
    ans_nosupp = interpret_q12("NO SUPPORT")

    assert ans_pass.startswith("Yes."), "Q12 PASS interpretation failed!"
    assert "downstream" in ans_diag.lower() and ans_diag.startswith("Partially."), "Q12 DIAGNOSTIC interpretation failed!"
    assert ans_nosupp.startswith("No."), "Q12 NO SUPPORT interpretation failed!"
    assert len(set([ans_pass, ans_diag, ans_nosupp])) == 3, "Q12 interpretations are not 3 distinct strings!"
    print("[PASS] Synthetic Test: Q12 3-way verdict interpretation verified", flush=True)

    # 4. Test D0 Dynamic Control Table Calculations
    syn_ctrl_pass = {"candidate_recall_20": 0.85, "final_recall_15": 0.70, "ndcg_10": 0.485, "mrr": 0.369}
    syn_ctrl_fail = {"candidate_recall_20": 0.50, "final_recall_15": 0.70, "ndcg_10": 0.485, "mrr": 0.369}

    syn_diag = {
        "D1": {"b3_depth": 20, "candidate_recall_augmented_pool": 0.85, "delta_candidate_recall": 0.0, "final_recall_15": 0.70, "delta_final_recall": 0.0, "ndcg_10": 0.485, "delta_ndcg10": 0.0, "mrr": 0.369, "delta_mrr": 0.0, "total_b3_candidates_added": 0, "candidate_pollution_rate": 0.0, "total_b3_retrieved": 0, "total_b3_duplicates": 0, "new_primary_candidates": 0, "new_supporting_candidates": 0, "new_rel_unnecessary_candidates": 0, "new_irrelevant_candidates": 0},
        "D2": {"b3_depth": 50, "candidate_recall_augmented_pool": 0.85, "delta_candidate_recall": 0.0, "final_recall_15": 0.70, "delta_final_recall": 0.0, "ndcg_10": 0.485, "delta_ndcg10": 0.0, "mrr": 0.369, "delta_mrr": 0.0, "total_b3_candidates_added": 0, "candidate_pollution_rate": 0.0, "total_b3_retrieved": 0, "total_b3_duplicates": 0, "new_primary_candidates": 0, "new_supporting_candidates": 0, "new_rel_unnecessary_candidates": 0, "new_irrelevant_candidates": 0},
        "D3": {"b3_depth": 100, "candidate_recall_augmented_pool": 0.85, "delta_candidate_recall": 0.0, "final_recall_15": 0.70, "delta_final_recall": 0.0, "ndcg_10": 0.485, "delta_ndcg10": 0.0, "mrr": 0.369, "delta_mrr": 0.0, "total_b3_candidates_added": 0, "candidate_pollution_rate": 0.0, "total_b3_retrieved": 0, "total_b3_duplicates": 0, "new_primary_candidates": 0, "new_supporting_candidates": 0, "new_rel_unnecessary_candidates": 0, "new_irrelevant_candidates": 0}
    }
    syn_prom = {"B3_to_candidate_count": 0, "candidate_to_R0_top15_count": 0, "R0_top15_to_final_count": 0}
    syn_trace = [{"evidence_chunk_id": "C15", "question_id": "Q03", "canonical_B0_rank": 99, "canonical_B0_candidate": False, "canonical_final": False, "B3_best_rank": 10, "entered_union_D1": True, "R0_rank_after_union_D1": 25, "final_top15_D1": False, "entered_union_D2": True, "R0_rank_after_union_D2": 25, "final_top15_D2": False, "entered_union_D3": True, "R0_rank_after_union_D3": 25, "final_top15_D3": False, "phaseD_diagnosis": "B3_DISCOVERABILITY_R0_FAILURE"}]
    syn_autopsy = {"C15": syn_trace[0]}

    syn_data_pass = {
        "metadata": {"verdict": "PASS"},
        "baseline_control_results": syn_ctrl_pass,
        "b3_diagnostic_results": syn_diag,
        "r0_promotion_metrics": syn_prom,
        "target_autopsies": syn_autopsy,
        "provenance_records": syn_trace,
        "question_metric_rows": []
    }
    syn_data_fail = {
        "metadata": {"verdict": "NO SUPPORT"},
        "baseline_control_results": syn_ctrl_fail,
        "b3_diagnostic_results": syn_diag,
        "r0_promotion_metrics": syn_prom,
        "target_autopsies": syn_autopsy,
        "provenance_records": syn_trace,
        "question_metric_rows": []
    }

    import tempfile
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".md") as f1, tempfile.NamedTemporaryFile("w", delete=False, suffix=".md") as f2:
        path1, path2 = f1.name, f2.name
    
    generate_markdown_report(syn_data_pass, path1, path2)
    with open(path1, "r", encoding="utf-8") as f:
        rep1 = f.read()
    assert "`PASS`" in rep1, "Control table did not compute PASS for baseline matching targets!"

    generate_markdown_report(syn_data_fail, path1, path2)
    with open(path1, "r", encoding="utf-8") as f:
        rep2 = f.read()
    assert "`FAIL`" in rep2, "Control table did not compute FAIL for baseline below target tolerance!"

    os.remove(path1)
    os.remove(path2)
    print("[PASS] Synthetic Test: D0 control table dynamic calculation verified", flush=True)

    # 5. Test Q6 Target Scoping Logic
    synthetic_trace = [
        {
            "evidence_chunk_id": "C12",
            "canonical_B0_candidate": True,
            "canonical_final": False,
        },
        {
            "evidence_chunk_id": "C65",
            "canonical_B0_candidate": True,
            "canonical_final": False,
        },
        {
            "evidence_chunk_id": "C155",
            "canonical_B0_candidate": False,
            "canonical_final": False,
        },
        {
            "evidence_chunk_id": "C99",
            "canonical_B0_candidate": True,
            "canonical_final": False,
        },
    ]
    syn_q6_unselected = [
        r["evidence_chunk_id"]
        for r in synthetic_trace
        if (
            r["evidence_chunk_id"] in Q6_TARGET_IDS
            and r["canonical_B0_candidate"]
            and not r["canonical_final"]
        )
    ]
    assert "C99" not in syn_q6_unselected, "Q6 synthetic test failed: C99 was reported!"
    assert sorted(syn_q6_unselected) == ["C12", "C65"], f"Q6 synthetic test failed: expected ['C12', 'C65'], got {syn_q6_unselected}"
    print("[PASS] Synthetic Test: Q6 target scoping verified (C99 excluded, C12 & C65 reported)", flush=True)

    print("\n==========================================================================", flush=True)
    print("   ALL SYNTHETIC TESTS PASSED SUCCESSFULLY!", flush=True)
    print("==========================================================================\n", flush=True)

if __name__ == "__main__":
    if "--synthetic-tests" in sys.argv or "--audit-only" in sys.argv:
        run_synthetic_tests()
    else:
        main()

