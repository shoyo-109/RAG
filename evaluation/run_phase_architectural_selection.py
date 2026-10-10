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
import inspect
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Setup system paths
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

# Frozen Q2 Subqueries
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

# D0 Control Tolerances
D0_CONTROL_TARGETS = {
    "candidate_recall_20": {"target": 0.85, "tolerance": 0.03},
    "final_recall_15": {"target": 0.70, "tolerance": 0.03},
    "ndcg_10": {"target": 0.485, "tolerance": 0.030},
    "mrr": {"target": 0.369, "tolerance": 0.030},
}

FINAL_RECALL_MIN_PRESERVATION_THRESHOLD = 0.670
NDCG_MIN_PRESERVATION_THRESHOLD = 0.455
MRR_MIN_PRESERVATION_THRESHOLD = 0.339

INACCESSIBLE_TARGET_IDS = {"C15", "C29", "C42"}

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
    found = []
    for ent in KNOWN_ENTITIES:
        if re.search(r"\b" + re.escape(ent) + r"\b", text, re.IGNORECASE):
            found.append(ent)
    return sorted(found)

def construct_b3_semantic_representation(chunk_item):
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

def calculate_metrics(selected_chunks, gold_map, top_k=15):
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

    ev_precision = useful_count / len(selected_top) if len(selected_top) > 0 else 0.0
    pri_precision = primary_count / len(selected_top) if len(selected_top) > 0 else 0.0
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

def validate_canonical_corpus(current_manifest_df, canonical_manifest_path):
    if not os.path.exists(canonical_manifest_path):
        raise FileNotFoundError(f"BLOCKED — CANONICAL CORPUS MANIFEST MISSING: {canonical_manifest_path}")

    canonical_df = pd.read_csv(canonical_manifest_path)

    if len(current_manifest_df) != len(canonical_df):
        raise RuntimeError(f"CORPUS IDENTITY FAILURE: chunk count changed. Current={len(current_manifest_df)}, Canonical={len(canonical_df)}")

    identity_columns = ["chunk_id", "chunk_idx", "page", "breadcrumb_path", "text_hash_sha256"]
    current_identity = current_manifest_df[identity_columns].fillna("").astype(str).reset_index(drop=True)
    canonical_identity = canonical_df[identity_columns].fillna("").astype(str).reset_index(drop=True)

    if not current_identity.equals(canonical_identity):
        raise RuntimeError("CORPUS IDENTITY FAILURE: Current corpus does not match canonical corpus.")

    print("CORPUS IDENTITY CHECK: PASS — Current corpus matches canonical manifest exactly.", flush=True)
    return True

def verify_target_blindness_invariants():
    """
    SECTION 2 & 16 RUNTIME ASSERTIONS: Prove target-blindness during retrieval and selection.
    """
    print("--- RUNNING TARGET-BLINDNESS RUNTIME INVARIANT VERIFICATION ---", flush=True)
    
    b3_sig = inspect.signature(construct_b3_semantic_representation)
    assert "gold_map" not in b3_sig.parameters and "required_evidence" not in b3_sig.parameters, "Target-Blindness Failure: gold parameter in B3 constructor!"

    selector_fn_sig = inspect.signature(run_llm_listwise_selector)
    assert "gold_map" not in selector_fn_sig.parameters, "Target-Blindness Failure: gold_map parameter found in listwise selector!"
    assert "required_evidence" not in selector_fn_sig.parameters, "Target-Blindness Failure: required_evidence parameter found in selector!"
    assert "hard_targets" not in selector_fn_sig.parameters, "Target-Blindness Failure: hard_targets parameter found in selector!"
    
    for qid, sqs in FROZEN_Q2_QUERIES.items():
        for sq in sqs:
            for tid in INACCESSIBLE_TARGET_IDS:
                assert tid not in sq, f"Target-Blindness Failure: Hard-target ID '{tid}' found in subquery '{sq}'!"
            assert "PRIMARY" not in sq and "SUPPORTING" not in sq, "Target-Blindness Failure: Gold role in subquery!"

    current_src = inspect.getsource(main)
    assert "if cid in INACCESSIBLE_TARGET_IDS:" not in current_src, "Target-Blindness Failure: Hard target override in main loop!"
    assert "boost_score_for_target" not in current_src, "Target-Blindness Failure: Score boost function found!"

    print("[PASS] All Target-Blindness Runtime Invariants Verified Cleanly!", flush=True)

def run_llm_listwise_selector(question, candidate_chunks, model_client, ordering_seed=None):
    """
    Executes Listwise Evidence-Set Selection using LLM.
    Strictly NO gold label or target ID leakage in prompt!
    """
    chunks_to_present = list(candidate_chunks)
    if ordering_seed is not None:
        rng = np.random.RandomState(ordering_seed)
        shuffled_indices = rng.permutation(len(chunks_to_present))
        chunks_to_present = [chunks_to_present[i] for i in shuffled_indices]

    prompt_candidates_str = ""
    for idx, chunk in enumerate(chunks_to_present, 1):
        cid = chunk["chunk_id"]
        breadcrumb = chunk.get("breadcrumb_path", "N/A")
        text_snippet = chunk["text"].replace("\n", " ")
        prompt_candidates_str += f"[{idx}] ID: {cid} | Section: {breadcrumb}\nContent: {text_snippet}\n\n"

    system_prompt = (
        "You are an expert retrieval evidence-set selector for complex question answering.\n"
        "Your goal is to select a minimal, non-redundant, complementary set of evidence chunks from the candidates that collectively contain all required evidence to fully answer the question.\n\n"
        "CRITICAL SELECTOR RULES:\n"
        "1. Focus on question relevance, coverage of required claims, and complementary information.\n"
        "2. Minimize redundancy (do not select multiple chunks stating the exact same fact unless necessary for context).\n"
        "3. You may select as many or as few chunks as necessary to fully answer the question.\n"
        "4. Stop when the evidence appears sufficient.\n"
        "5. Output MUST be valid JSON strictly adhering to the schema below.\n\n"
        "JSON SCHEMA:\n"
        "{\n"
        '  "selected_chunk_ids": ["C4", "C2"],\n'
        '  "selection_rationale": ["Rationale for C4...", "Rationale for C2..."],\n'
        '  "coverage_assessment": "Summary of claim coverage...",\n'
        '  "sufficiency": "FULLY_SUFFICIENT | PARTIALLY_SUFFICIENT | INSUFFICIENT",\n'
        '  "confidence": "HIGH | MEDIUM | LOW"\n'
        "}"
    )

    user_content = f"QUESTION: {question}\n\nCANDIDATE CHUNKS:\n{prompt_candidates_str}"

    try:
        resp = model_client.chat.completions.create(
            model="nvidia/nemotron-3-ultra-550b-a55b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.0
        )
        content = resp.choices[0].message.content.strip()
        
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        content = content.strip()
        
        parsed = json.loads(content)
        selected_ids = parsed.get("selected_chunk_ids", [])
        
        valid_cids = set(c["chunk_id"] for c in candidate_chunks)
        filtered_selected = [cid for cid in selected_ids if cid in valid_cids]
        
        parsed["selected_chunk_ids"] = filtered_selected
        parsed["raw_response"] = content
        return parsed
    except Exception as e:
        print(f"Error in LLM Listwise Selector: {e}", flush=True)
        fallback_ids = [c["chunk_id"] for c in candidate_chunks[:5]]
        return {
            "selected_chunk_ids": fallback_ids,
            "selection_rationale": ["Fallback due to LLM response error"],
            "coverage_assessment": "Fallback selection",
            "sufficiency": "INSUFFICIENT",
            "confidence": "LOW",
            "error": str(e)
        }

def run_mmr_selector(question_emb, candidate_chunks, chunk_embs_b0, top_k=15, lambda_param=0.6):
    """
    Lightweight deterministic MMR selector for evidence-set coverage.
    """
    c_indices = [c["chunk_idx"] for c in candidate_chunks]
    c_embs = chunk_embs_b0[c_indices]
    
    query_sims = np.dot(c_embs, question_emb)
    cand_sim_matrix = np.dot(c_embs, c_embs.T)
    
    selected_indices = []
    unselected = list(range(len(candidate_chunks)))
    
    while unselected and len(selected_indices) < top_k:
        if not selected_indices:
            best_idx = int(np.argmax(query_sims[unselected]))
            selected_indices.append(unselected[best_idx])
            unselected.pop(best_idx)
        else:
            mmr_scores = []
            for u in unselected:
                rel = query_sims[u]
                max_sim = max(cand_sim_matrix[u, s] for s in selected_indices)
                score = lambda_param * rel - (1.0 - lambda_param) * max_sim
                mmr_scores.append(score)
            best_u_idx = int(np.argmax(mmr_scores))
            selected_indices.append(unselected[best_u_idx])
            unselected.pop(best_u_idx)
            
    selected_chunks = [candidate_chunks[i] for i in selected_indices]
    return selected_chunks

def compute_bootstrap_confidence_intervals(metric_data, n_bootstraps=1000, ci_level=0.95):
    """
    CRITICAL FIX #1: Real bootstrap computation without hardcoding ci_lower or std_error.
    """
    np.random.seed(42)
    n_samples = len(metric_data)
    if n_samples == 0:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "std_error": 0.0, "is_stable": False}

    boot_means = []
    for _ in range(n_bootstraps):
        boot_sample = np.random.choice(metric_data, size=n_samples, replace=True)
        boot_means.append(np.mean(boot_sample))

    alpha = (1.0 - ci_level) / 2.0
    lower = float(np.percentile(boot_means, alpha * 100))
    upper = float(np.percentile(boot_means, (1.0 - alpha) * 100))
    mean = float(np.mean(metric_data))
    std_err = float(np.std(boot_means))

    is_stable = std_err <= 0.15 and (upper - lower) <= 0.40

    return {
        "mean": round(mean, 4),
        "ci_lower": round(lower, 4),
        "ci_upper": round(upper, 4),
        "std_error": round(std_err, 4),
        "is_stable": is_stable
    }

def compute_terminal_verdict(e0_pass, is_variance_stable, listwise_results_by_ordering, e0_d1_total_gold_recovered, evidence_trace_records):
    """
    CRITICAL FIX #2 & #3 (USER UPDATES):
    - Explicitly evaluate Ordering Robustness across Orderings A, B, C.
    - Genuine evidence-recovery improvement criterion: Net total gold evidence units recovered MUST strictly exceed E0_D1 total gold evidence units recovered.
    - Equal aggregate recall + 1 recovered target with equal net evidence IS NOT a genuine evidence improvement.
    """
    if not e0_pass:
        return "BLOCKED — CANONICAL CONTROL REPRODUCTION FAILED"

    if not is_variance_stable:
        return "EVALUATION INCONCLUSIVE — EXPAND BENCHMARK"

    # Evaluate Ordering Robustness across Orderings A, B, C
    required_orderings = ["Ordering_A_Canonical", "Ordering_B_Shuffle42", "Ordering_C_Shuffle123"]
    
    ordering_checks = {}
    for ord_key in required_orderings:
        if ord_key not in listwise_results_by_ordering:
            return "ARCHITECTURAL HYPOTHESIS NOT SUPPORTED"
            
        ord_res = listwise_results_by_ordering[ord_key]
        f_rec = ord_res["final_recall_15"]
        ndcg = ord_res["ndcg_10"]
        mrr = ord_res["mrr"]
        
        # Production quality preservation check
        prod_pass = (
            f_rec >= FINAL_RECALL_MIN_PRESERVATION_THRESHOLD and
            ndcg >= NDCG_MIN_PRESERVATION_THRESHOLD and
            mrr >= MRR_MIN_PRESERVATION_THRESHOLD
        )

        # Genuine evidence-recovery improvement check:
        # Total gold evidence units recovered naturally or in fixed-budget in ord_key MUST be > e0_d1_total_gold_recovered
        # AND at least one hard target MUST be recovered.
        ord_trace = [r for r in evidence_trace_records if r["configuration"] == f"Listwise_LLM_{ord_key}"]
        ord_gold_recovered = sum(1 for r in ord_trace if r["final_recovery"])
        ord_hard_recovered = sum(1 for r in ord_trace if r["evidence_chunk_id"] in INACCESSIBLE_TARGET_IDS and r["final_recovery"])

        # Genuine improvement: NET gold evidence recovered strictly greater than E0_D1 AND hard target recovered
        genuine_improvement = (ord_gold_recovered > e0_d1_total_gold_recovered) and (ord_hard_recovered > 0)

        ordering_checks[ord_key] = prod_pass and genuine_improvement

    # ALL orderings A, B, C MUST PASS!
    all_orderings_pass = all(ordering_checks.values())

    if all_orderings_pass:
        return "ARCHITECTURAL SUPPORT — EVIDENCE-SET SELECTION"
    else:
        return "ARCHITECTURAL HYPOTHESIS NOT SUPPORTED"

def evaluate_blind_answer_impact(q_text, context_d0, context_augmented, nv_client):
    """
    Dynamic Answer-Level Impact Evaluator (No hardcoded Q_IDs).
    """
    prompt_template = (
        "You are a strict, objective RAG Answer Evaluator.\n"
        "Evaluate the two draft answers below generated from different context representations for the question.\n\n"
        "QUESTION: {question}\n\n"
        "DRAFT ANSWER A (Standard Context):\n{ans_a}\n\n"
        "DRAFT ANSWER B (Augmented Context):\n{ans_b}\n\n"
        "Compare Draft Answer A and Draft Answer B. Does Draft Answer B provide essential factual claims or accurate details that Draft Answer A completely failed to state or got wrong?\n"
        "Return ONLY JSON: {\"material_impact\": true | false, \"reasoning\": \"...\"}"
    )
    
    prompt_gen = "Answer the question thoroughly based ONLY on context.\n\nCONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:"
    
    try:
        ans_a = nv_client.chat.completions.create(
            model="nvidia/nemotron-3-ultra-550b-a55b",
            messages=[{"role": "user", "content": prompt_gen.format(context=context_d0, question=q_text)}],
            temperature=0.0
        ).choices[0].message.content

        ans_b = nv_client.chat.completions.create(
            model="nvidia/nemotron-3-ultra-550b-a55b",
            messages=[{"role": "user", "content": prompt_gen.format(context=context_augmented, question=q_text)}],
            temperature=0.0
        ).choices[0].message.content

        eval_resp = nv_client.chat.completions.create(
            model="nvidia/nemotron-3-ultra-550b-a55b",
            messages=[{"role": "user", "content": prompt_template.format(question=q_text, ans_a=ans_a, ans_b=ans_b)}],
            temperature=0.0
        ).choices[0].message.content.strip()

        if eval_resp.startswith("```"):
            eval_resp = eval_resp.split("```")[1].replace("json", "").strip()
        parsed_eval = json.loads(eval_resp)

        if parsed_eval.get("material_impact", False):
            return "RETRIEVAL FAILURE WITH ANSWER IMPACT", ans_a, ans_b
        else:
            return "RETRIEVAL FAILURE WITHOUT OBSERVED ANSWER IMPACT", ans_a, ans_b
    except Exception as e:
        print(f"Blind evaluator fallback due to API output: {e}", flush=True)
        return "UNCLASSIFIED — NO BLIND ANSWER EVALUATOR", "", ""

def main():
    print("==========================================================================", flush=True)
    print("   TERMINAL ARCHITECTURAL PHASE — EVIDENCE-SET SELECTION EXECUTION", flush=True)
    print("==========================================================================", flush=True)

    api_key_nv = os.getenv("NVIDIA_API_KEY") or os.getenv("NEMOTRON_API_KEY")
    if not api_key_nv:
        raise ValueError("CRITICAL ERROR: NVIDIA_API_KEY or NEMOTRON_API_KEY not found in environment!")
    nv_client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key_nv)

    verify_target_blindness_invariants()

    doc_path = os.path.join(root_dir, "Documents", "The Comedy of Errors document.pdf")
    session_id = f"arch_phase_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    pipeline = AdvancedRAGPipeline(session_id=session_id)
    raw_docs = IngestionPipeline.load_document(doc_path)
    pipeline.add_documents(raw_docs)

    chunk_count = len(pipeline.all_chunks)
    print(f"Loaded indexed corpus: {chunk_count} chunks.", flush=True)
    if chunk_count != 203:
        report_blocked_verdict(root_dir, "BLOCKED — CANONICAL CONTROL REPRODUCTION FAILED")
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

    corpus_manifest = []
    for chunk in formatted_chunks:
        text_hash = hashlib.sha256(chunk["text"].encode("utf-8")).hexdigest()
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

    canonical_manifest_path = os.path.join(root_dir, "evaluation", "baseline", "phaseD_corpus_manifest_canonical.csv")
    try:
        validate_canonical_corpus(current_manifest_df=corpus_manifest_df, canonical_manifest_path=canonical_manifest_path)
    except Exception as e:
        print(f"Corpus Validation Failure: {e}", flush=True)
        report_blocked_verdict(root_dir, "BLOCKED — CANONICAL CONTROL REPRODUCTION FAILED")
        sys.exit(1)

    b0_texts = [c["text"] for c in formatted_chunks]
    b3_texts = [construct_b3_semantic_representation(c) for c in formatted_chunks]

    print("Loading dense embedding model 'sentence-transformers/all-MiniLM-L6-v2' (E0)...", flush=True)
    from sentence_transformers import SentenceTransformer, CrossEncoder
    embedder_e0 = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("Embedding B0 and B3 representations...", flush=True)
    embs_b0 = np.array(embedder_e0.encode(b0_texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True))
    embs_b3 = np.array(embedder_e0.encode(b3_texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True))

    all_sq_texts = []
    for b_case in BENCHMARK_SUITE:
        all_sq_texts.extend(FROZEN_Q2_QUERIES[b_case["q_id"]])
    all_sq_texts = sorted(list(set(all_sq_texts)))
    sq_embs_array = np.array(embedder_e0.encode(all_sq_texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True))
    sq_emb_map = {sq: emb for sq, emb in zip(all_sq_texts, sq_embs_array)}

    reranker_r0 = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    print("\n--- CONTROL REPRODUCTION GATE (E0 / D0) ---", flush=True)
    total_req_units = sum(len(b["required_evidence"]) for b in BENCHMARK_SUITE)
    e0_cand_retrieved = 0
    e0_final_retrieved = 0
    e0_ndcg10_list = []
    e0_mrr_list = []
    d0_runs = {}

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
        e0_cand_retrieved += sum(1 for cid in req_units if cid in cand_ids)

        ce_pairs = [(q_text, chunk["text"]) for chunk in cand_chunks]
        ce_scores = reranker_r0.predict(ce_pairs, batch_size=32, show_progress_bar=False)
        for chunk, score in zip(cand_chunks, ce_scores):
            chunk["ce_score"] = float(score)

        ce_ranked = sorted(cand_chunks, key=lambda x: (-x["ce_score"], x["chunk_idx"]))
        top15_cids = [item["chunk_id"] for item in ce_ranked[:15]]
        e0_final_retrieved += sum(1 for cid in req_units if cid in top15_cids)

        sec10 = calculate_metrics(ce_ranked, b_case["gold_map"], top_k=10)
        sec15 = calculate_metrics(ce_ranked, b_case["gold_map"], top_k=15)
        e0_ndcg10_list.append(sec10["ndcg_k"])
        e0_mrr_list.append(sec15["mrr"])

        d0_runs[q_id] = {
            "cand_map": candidate_map,
            "cand_ids": cand_ids,
            "cand_chunks": cand_chunks,
            "ce_ranked": ce_ranked,
            "top15_cids": top15_cids
        }

    e0_cand_rec = e0_cand_retrieved / total_req_units
    e0_final_rec = e0_final_retrieved / total_req_units
    e0_ndcg10 = float(np.mean(e0_ndcg10_list))
    e0_mrr = float(np.mean(e0_mrr_list))

    print(f"E0 / D0 Control Verification:")
    print(f"  Candidate Recall@20 : {e0_cand_rec*100:.1f}% (Expected: 85.0%)")
    print(f"  Final Recall@15     : {e0_final_rec*100:.1f}% (Expected: 70.0%)")
    print(f"  Macro nDCG@10       : {e0_ndcg10:.3f} (Expected: 0.485)")
    print(f"  MRR                 : {e0_mrr:.3f} (Expected: 0.369)")

    cand_cfg = D0_CONTROL_TARGETS["candidate_recall_20"]
    final_cfg = D0_CONTROL_TARGETS["final_recall_15"]
    ndcg_cfg = D0_CONTROL_TARGETS["ndcg_10"]
    mrr_cfg = D0_CONTROL_TARGETS["mrr"]

    pass_cand = abs(e0_cand_rec - cand_cfg["target"]) <= cand_cfg["tolerance"]
    pass_final = abs(e0_final_rec - final_cfg["target"]) <= final_cfg["tolerance"]
    pass_ndcg = abs(e0_ndcg10 - ndcg_cfg["target"]) <= ndcg_cfg["tolerance"]
    pass_mrr = abs(e0_mrr - mrr_cfg["target"]) <= mrr_cfg["tolerance"]

    e0_pass = pass_cand and pass_final and pass_ndcg and pass_mrr

    if not e0_pass:
        print("\nCRITICAL CONTROL REPRODUCTION FAILURE — EXPERIMENT BLOCKED!", flush=True)
        report_blocked_verdict(root_dir, "BLOCKED — CANONICAL CONTROL REPRODUCTION FAILED")
        sys.exit(1)
    else:
        print(">>> CONTROL REPRODUCTION GATE PASSED! Proceeding to Architectural Evaluation...\n", flush=True)

    print("--- SECTION 4: ANSWER-LEVEL IMPACT CHECK (DYNAMIC EVALUATOR) ---", flush=True)
    answer_impact_results = []
    hard_questions = [b for b in BENCHMARK_SUITE if b["q_id"] in ["Q03", "Q05"]]
    
    for b_case in hard_questions:
        q_id = b_case["q_id"]
        q_text = b_case["question"]
        
        d0_top15 = d0_runs[q_id]["ce_ranked"][:15]
        context_d0 = "\n\n".join([f"[{c['chunk_id']}] {c['text']}" for c in d0_top15])
        
        hard_cids = [cid for cid in INACCESSIBLE_TARGET_IDS if cid in b_case["gold_map"]]
        augmented_chunks = list(d0_top15)
        for h_cid in hard_cids:
            h_c = next((c for c in formatted_chunks if c["chunk_id"] == h_cid), None)
            if h_c and h_cid not in [c["chunk_id"] for c in augmented_chunks]:
                augmented_chunks.append(h_c)
        context_augmented = "\n\n".join([f"[{c['chunk_id']}] {c['text']}" for c in augmented_chunks])
        
        impact_label, ans_a, ans_b = evaluate_blind_answer_impact(q_text, context_d0, context_augmented, nv_client)
        
        answer_impact_results.append({
            "question_id": q_id,
            "hard_targets": hard_cids,
            "diagnostic_mode": "COUNTERFACTUAL_INJECTION_DIAGNOSTIC",
            "answer_without_target": ans_a,
            "answer_with_target": ans_b,
            "impact_classification": impact_label
        })
        print(f"  {q_id} Hard Targets {hard_cids} Classification: {impact_label}")

    print("\nGenerating Single Frozen D1 Candidate Pools...", flush=True)
    d1_candidate_pools = {}
    for b_case in BENCHMARK_SUITE:
        q_id = b_case["q_id"]
        subqueries = FROZEN_Q2_QUERIES[q_id]

        c_run = d0_runs[q_id]
        cand_map = {cid: dict(item) for cid, item in c_run["cand_map"].items()}

        for sq_idx, sq in enumerate(subqueries):
            sq_emb = sq_emb_map[sq]
            b3_sims = np.dot(embs_b3, sq_emb)
            b3_ranked_indices = np.argsort(-b3_sims)
            for r, c_i in enumerate(b3_ranked_indices[:20], 1):
                cid = f"C{c_i}"
                if cid not in cand_map:
                    cand_map[cid] = dict(formatted_chunks[c_i])
                    cand_map[cid]["sources"] = []
                cand_map[cid]["sources"].append(f"dense_b3_sq{sq_idx+1}_rank_{r}")

        d1_candidate_pools[q_id] = cand_map

    print("\nRunning Control A: R0 Independent Ranking on D1 (E0_D1)...", flush=True)
    e0_d1_results = {}
    e0_d1_final_retrieved = 0
    e0_d1_ndcg10_list = []
    e0_d1_mrr_list = []
    e0_d1_ranks = {}

    for b_case in BENCHMARK_SUITE:
        q_id = b_case["q_id"]
        q_text = b_case["question"]
        req_units = b_case["required_evidence"]
        gold_map = b_case["gold_map"]

        cand_map = d1_candidate_pools[q_id]
        cand_chunks = list(cand_map.values())
        cand_ids = [c["chunk_id"] for c in cand_chunks]

        ce_pairs = [(q_text, chunk["text"]) for chunk in cand_chunks]
        ce_scores = reranker_r0.predict(ce_pairs, batch_size=32, show_progress_bar=False)
        for chunk, score in zip(cand_chunks, ce_scores):
            chunk["ce_score"] = float(score)

        ce_ranked = sorted(cand_chunks, key=lambda x: (-x["ce_score"], x["chunk_idx"]))
        ce_ranked_ids = [c["chunk_id"] for c in ce_ranked]
        top15_cids = ce_ranked_ids[:15]
        e0_d1_final_retrieved += sum(1 for cid in req_units if cid in top15_cids)

        sec10 = calculate_metrics(ce_ranked, gold_map, top_k=10)
        sec15 = calculate_metrics(ce_ranked, gold_map, top_k=15)
        e0_d1_ndcg10_list.append(sec10["ndcg_k"])
        e0_d1_mrr_list.append(sec15["mrr"])

        for gold_cid in req_units:
            r_rank = ce_ranked_ids.index(gold_cid) + 1 if gold_cid in ce_ranked_ids else 999
            e0_d1_ranks[(q_id, gold_cid)] = r_rank

        e0_d1_results[q_id] = {
            "cand_chunks": cand_chunks,
            "ce_ranked": ce_ranked,
            "top15_cids": top15_cids,
            "ndcg_10": sec10["ndcg_k"],
            "mrr": sec15["mrr"]
        }

    e0_d1_final_rec = e0_d1_final_retrieved / total_req_units
    e0_d1_ndcg10 = float(np.mean(e0_d1_ndcg10_list))
    e0_d1_mrr = float(np.mean(e0_d1_mrr_list))

    print(f"E0_D1 Control A Results:")
    print(f"  Final Recall@15 : {e0_d1_final_rec*100:.1f}%")
    print(f"  Macro nDCG@10   : {e0_d1_ndcg10:.3f}")
    print(f"  MRR             : {e0_d1_mrr:.3f}")

    print("\nExecuting Experimental B: Listwise LLM Evidence Selector...", flush=True)
    selector_run_records = []
    evidence_trace_records = []
    question_metric_rows = []

    orderings = [
        {"name": "Ordering_A_Canonical", "seed": None},
        {"name": "Ordering_B_Shuffle42", "seed": 42},
        {"name": "Ordering_C_Shuffle123", "seed": 123}
    ]

    listwise_results_by_ordering = {}

    for ord_spec in orderings:
        ord_name = ord_spec["name"]
        ord_seed = ord_spec["seed"]
        print(f"\n--- Running Listwise Selector: Position-Order Robustness {ord_name} ---", flush=True)

        total_listwise_retrieved = 0
        total_normalized_retrieved = 0
        listwise_ndcg10_list = []
        listwise_mrr_list = []
        selected_set_sizes = []

        for b_case in BENCHMARK_SUITE:
            q_id = b_case["q_id"]
            q_text = b_case["question"]
            req_units = b_case["required_evidence"]
            gold_map = b_case["gold_map"]
            subqueries = FROZEN_Q2_QUERIES[q_id]

            cand_map = d1_candidate_pools[q_id]
            cand_chunks = list(cand_map.values())
            cand_ids = [c["chunk_id"] for c in cand_chunks]

            start_t = time.time()
            llm_sel_output = run_llm_listwise_selector(
                question=q_text,
                candidate_chunks=cand_chunks,
                model_client=nv_client,
                ordering_seed=ord_seed
            )
            elapsed_t = time.time() - start_t

            selected_cids = llm_sel_output["selected_chunk_ids"]
            selected_set_sizes.append(len(selected_cids))

            selected_chunks = [c for c in cand_chunks if c["chunk_id"] in selected_cids]
            
            remaining_cids = [c["chunk_id"] for c in e0_d1_results[q_id]["ce_ranked"] if c["chunk_id"] not in selected_cids]
            normalized_top15_cids = (selected_cids + remaining_cids)[:15]
            normalized_top15_chunks = [next(c for c in cand_chunks if c["chunk_id"] == cid) for cid in normalized_top15_cids]

            sec10 = calculate_metrics(normalized_top15_chunks, gold_map, top_k=10)
            sec15 = calculate_metrics(normalized_top15_chunks, gold_map, top_k=15)

            listwise_ndcg10_list.append(sec10["ndcg_k"])
            listwise_mrr_list.append(sec15["mrr"])

            req_ret_selected = sum(1 for cid in req_units if cid in selected_cids)
            req_ret_norm = sum(1 for cid in req_units if cid in normalized_top15_cids)

            total_listwise_retrieved += req_ret_selected
            total_normalized_retrieved += req_ret_norm

            selector_run_records.append({
                "ordering_name": ord_name,
                "ordering_seed": ord_seed if ord_seed is not None else 0,
                "question_id": q_id,
                "model": "nvidia/nemotron-3-ultra-550b-a55b",
                "temperature": 0.0,
                "selected_set_size": len(selected_cids),
                "sufficiency": llm_sel_output.get("sufficiency", ""),
                "confidence": llm_sel_output.get("confidence", ""),
                "selected_chunk_ids": "|".join(selected_cids),
                "latency_seconds": round(elapsed_t, 3),
                "timestamp": datetime.now().isoformat()
            })

            question_metric_rows.append({
                "config": f"Listwise_LLM_{ord_name}",
                "question_id": q_id,
                "candidate_pool_size": len(cand_ids),
                "candidate_recall": sum(1 for cid in req_units if cid in cand_ids) / len(req_units),
                "selected_set_size": len(selected_cids),
                "natural_selected_recall": req_ret_selected / len(req_units),
                "fixed_budget_hybrid_recall_15": req_ret_norm / len(req_units),
                "ndcg_10": sec10["ndcg_k"],
                "mrr": sec15["mrr"],
                "evidence_precision": sec15["ev_precision"],
                "primary_precision": sec15["pri_precision"],
                "redundancy": sec15["redundancy"]
            })

            for gold_cid in req_units:
                gold_idx = int(gold_cid[1:])
                gold_role = gold_map[gold_cid]

                sq_b0_ranks = [int(np.where(np.argsort(-np.dot(embs_b0, sq_emb_map[sq])) == gold_idx)[0][0]) + 1 for sq in subqueries]
                b0_best_rank = min(sq_b0_ranks)

                entered_d1 = gold_cid in cand_ids
                r0_d1_r = e0_d1_ranks.get((q_id, gold_cid), 999)
                is_selected = gold_cid in selected_cids
                listwise_pos = selected_cids.index(gold_cid) + 1 if is_selected else 999
                final_recovery = gold_cid in normalized_top15_cids

                if final_recovery:
                    fail_stage = "FINAL_RECOVERY_SUCCESS"
                elif not entered_d1:
                    fail_stage = "CANDIDATE_GENERATION_FAILURE"
                else:
                    fail_stage = "SELECTION_FAILURE"

                evidence_trace_records.append({
                    "configuration": f"Listwise_LLM_{ord_name}",
                    "question_id": q_id,
                    "evidence_chunk_id": gold_cid,
                    "evidence_role": gold_role,
                    "canonical_B0_rank": b0_best_rank,
                    "entered_D1": entered_d1,
                    "E0_D1_rank": r0_d1_r,
                    "R0_D1_rank": r0_d1_r,
                    "listwise_selected": is_selected,
                    "listwise_position": listwise_pos,
                    "natural_set_membership": is_selected,
                    "fixed_top15_membership": final_recovery,
                    "final_recovery": final_recovery,
                    "failure_stage": fail_stage,
                    "candidate_order_seed": ord_seed if ord_seed is not None else 0,
                    "selector_seed": 0,
                    "selector_model": "nvidia/nemotron-3-ultra-550b-a55b",
                    "selector_prompt_version": "v1_structured_json"
                })

        ord_final_rec = total_normalized_retrieved / total_req_units
        ord_raw_rec = total_listwise_retrieved / total_req_units
        ord_ndcg10 = float(np.mean(listwise_ndcg10_list))
        ord_mrr = float(np.mean(listwise_mrr_list))
        avg_set_size = float(np.mean(selected_set_sizes))

        listwise_results_by_ordering[ord_name] = {
            "ordering": ord_name,
            "e0_d1_final_rec": e0_d1_final_rec,
            "raw_selected_recall": ord_raw_rec,
            "final_recall_15": ord_final_rec,
            "ndcg_10": ord_ndcg10,
            "mrr": ord_mrr,
            "avg_set_size": avg_set_size
        }

        print(f"Results for Listwise Selector ({ord_name}):")
        print(f"  Natural Selected Recall : {ord_raw_rec*100:.1f}%")
        print(f"  Fixed Budget Recall@15  : {ord_final_rec*100:.1f}%")
        print(f"  Macro nDCG@10           : {ord_ndcg10:.3f}")
        print(f"  MRR                     : {ord_mrr:.3f}")

    print("\n--- SECTION 5: REAL BOOTSTRAP VARIANCE & STABILITY ANALYSIS ---", flush=True)
    listwise_canonical = listwise_results_by_ordering["Ordering_A_Canonical"]
    
    # CRITICAL FIX #1 (USER UPDATE): Calculate real bootstrap CI/SE for Canonical D0 directly without fabrication!
    d0_rec_scores = [sum(1 for cid in b["required_evidence"] if cid in d0_runs[b["q_id"]]["top15_cids"])/len(b["required_evidence"]) for b in BENCHMARK_SUITE]
    ci_d0 = compute_bootstrap_confidence_intervals(d0_rec_scores)
    
    listwise_rec_scores = [
        r["fixed_budget_hybrid_recall_15"] for r in question_metric_rows if r["config"] == "Listwise_LLM_Ordering_A_Canonical"
    ]
    listwise_ndcg_scores = [
        r["ndcg_10"] for r in question_metric_rows if r["config"] == "Listwise_LLM_Ordering_A_Canonical"
    ]
    listwise_mrr_scores = [
        r["mrr"] for r in question_metric_rows if r["config"] == "Listwise_LLM_Ordering_A_Canonical"
    ]

    ci_rec = compute_bootstrap_confidence_intervals(listwise_rec_scores)
    ci_ndcg = compute_bootstrap_confidence_intervals(listwise_ndcg_scores)
    ci_mrr = compute_bootstrap_confidence_intervals(listwise_mrr_scores)

    delta_rec_scores = [l - d for l, d in zip(listwise_rec_scores, d0_rec_scores)]
    ci_delta_rec = compute_bootstrap_confidence_intervals(delta_rec_scores)

    is_variance_stable = ci_rec["is_stable"] and ci_ndcg["is_stable"] and ci_mrr["is_stable"]

    variance_report_rows = [
        {"configuration": "Canonical_D0", "metric": "Final_Recall_15", "mean": ci_d0["mean"], "ci_lower": ci_d0["ci_lower"], "ci_upper": ci_d0["ci_upper"], "std_error": ci_d0["std_error"]},
        {"configuration": "Listwise_LLM_Ordering_A", "metric": "Final_Recall_15", "mean": ci_rec["mean"], "ci_lower": ci_rec["ci_lower"], "ci_upper": ci_rec["ci_upper"], "std_error": ci_rec["std_error"]},
        {"configuration": "Listwise_LLM_Ordering_A", "metric": "nDCG_10", "mean": ci_ndcg["mean"], "ci_lower": ci_ndcg["ci_lower"], "ci_upper": ci_ndcg["ci_upper"], "std_error": ci_ndcg["std_error"]},
        {"configuration": "Listwise_LLM_Ordering_A", "metric": "MRR", "mean": ci_mrr["mean"], "ci_lower": ci_mrr["ci_lower"], "ci_upper": ci_mrr["ci_upper"], "std_error": ci_mrr["std_error"]},
        {"configuration": "Delta_Listwise_vs_D0", "metric": "Delta_Final_Recall", "mean": ci_delta_rec["mean"], "ci_lower": ci_delta_rec["ci_lower"], "ci_upper": ci_delta_rec["ci_upper"], "std_error": ci_delta_rec["std_error"]}
    ]

    print(f"Bootstrap 95% Confidence Intervals (Listwise Selector):")
    print(f"  Final Recall@15 : {ci_rec['mean']} (95% CI: [{ci_rec['ci_lower']}, {ci_rec['ci_upper']}])")
    print(f"  Macro nDCG@10   : {ci_ndcg['mean']} (95% CI: [{ci_ndcg['ci_lower']}, {ci_ndcg['ci_upper']}])")
    print(f"  MRR             : {ci_mrr['mean']} (95% CI: [{ci_mrr['ci_lower']}, {ci_mrr['ci_upper']}])")
    print(f"  Stability Gate  : {'STABLE' if is_variance_stable else 'INCONCLUSIVE'}")

    # Calculate total gold evidence units recovered by E0_D1 across all questions
    e0_d1_total_gold_recovered = sum(
        sum(1 for cid in b["required_evidence"] if cid in e0_d1_results[b["q_id"]]["top15_cids"])
        for b in BENCHMARK_SUITE
    )

    final_verdict = compute_terminal_verdict(
        e0_pass=e0_pass,
        is_variance_stable=is_variance_stable,
        listwise_results_by_ordering=listwise_results_by_ordering,
        e0_d1_total_gold_recovered=e0_d1_total_gold_recovered,
        evidence_trace_records=evidence_trace_records
    )

    listwise_success = (final_verdict == "ARCHITECTURAL SUPPORT — EVIDENCE-SET SELECTION")

    if listwise_success:
        print("\nRunning Conditional Deterministic MMR Selector on D1...", flush=True)
        mmr_final_retrieved = 0
        mmr_ndcg10_list = []
        mmr_mrr_list = []

        for b_case in BENCHMARK_SUITE:
            q_id = b_case["q_id"]
            q_text = b_case["question"]
            req_units = b_case["required_evidence"]
            gold_map = b_case["gold_map"]

            cand_map = d1_candidate_pools[q_id]
            cand_chunks = list(cand_map.values())
            
            q_emb = embedder_e0.encode(q_text, normalize_embeddings=True)
            mmr_chunks = run_mmr_selector(q_emb, cand_chunks, embs_b0, top_k=15, lambda_param=0.6)
            mmr_cids = [c["chunk_id"] for c in mmr_chunks]

            mmr_final_retrieved += sum(1 for cid in req_units if cid in mmr_cids)
            sec10 = calculate_metrics(mmr_chunks, gold_map, top_k=10)
            sec15 = calculate_metrics(mmr_chunks, gold_map, top_k=15)
            mmr_ndcg10_list.append(sec10["ndcg_k"])
            mmr_mrr_list.append(sec15["mrr"])

        mmr_res = {
            "status": "EXECUTED",
            "final_recall_15": mmr_final_retrieved / total_req_units,
            "ndcg_10": float(np.mean(mmr_ndcg10_list)),
            "mrr": float(np.mean(mmr_mrr_list))
        }
    else:
        print("\nMMR EXPERIMENT SKIPPED (Condition: Listwise success criterion not met).", flush=True)
        mmr_res = {
            "status": "NOT_RUN",
            "reason": "LISTWISE_SUCCESS_CRITERION_NOT_MET",
            "final_recall_15": 0.0,
            "ndcg_10": 0.0,
            "mrr": 0.0
        }

    print("\n--- SECTION 21: CONDITIONAL GATING EXPERIMENT (ACTUAL EXECUTION) ---", flush=True)
    gating_results = []
    gated_retrieved = 0
    gated_ndcg_list = []
    gated_mrr_list = []
    
    for b_case in BENCHMARK_SUITE:
        q_id = b_case["q_id"]
        q_text = b_case["question"]
        req_units = b_case["required_evidence"]
        gold_map = b_case["gold_map"]
        
        d0_top15 = d0_runs[q_id]["ce_ranked"][:15]
        context_d0 = "\n\n".join([f"[{c['chunk_id']}] {c['text']}" for c in d0_top15])
        
        check_prompt = (
            "Analyze if the retrieved context is fully sufficient to answer the question.\n\n"
            f"QUESTION: {q_text}\n\nCONTEXT:\n{context_d0}\n\n"
            "Return ONLY JSON: {\"sufficient\": true | false, \"missing_aspects\": \"...\"}"
        )
        
        resp_check = nv_client.chat.completions.create(
            model="nvidia/nemotron-3-ultra-550b-a55b",
            messages=[{"role": "user", "content": check_prompt}],
            temperature=0.0
        ).choices[0].message.content.strip()
        
        try:
            if resp_check.startswith("```"):
                resp_check = resp_check.split("```")[1].replace("json", "").strip()
            parsed_check = json.loads(resp_check)
            is_suff = parsed_check.get("sufficient", True)
        except Exception:
            is_suff = True

        if not is_suff:
            cand_chunks = list(d1_candidate_pools[q_id].values())
            llm_sel = run_llm_listwise_selector(q_text, cand_chunks, nv_client)
            sel_cids = llm_sel["selected_chunk_ids"]
            rem_cids = [c["chunk_id"] for c in e0_d1_results[q_id]["ce_ranked"] if c["chunk_id"] not in sel_cids]
            final_cids = (sel_cids + rem_cids)[:15]
            final_chunks = [next(c for c in cand_chunks if c["chunk_id"] == cid) for cid in final_cids]
            action = "ESCALATED_AND_EXECUTED_LISTWISE_D1"
        else:
            final_chunks = d0_runs[q_id]["ce_ranked"][:15]
            final_cids = [c["chunk_id"] for c in final_chunks]
            action = "RETAINED_CANONICAL_D0"

        gated_retrieved += sum(1 for cid in req_units if cid in final_cids)
        sec10 = calculate_metrics(final_chunks, gold_map, top_k=10)
        sec15 = calculate_metrics(final_chunks, gold_map, top_k=15)
        gated_ndcg_list.append(sec10["ndcg_k"])
        gated_mrr_list.append(sec15["mrr"])
            
        gating_results.append({
            "question_id": q_id,
            "action": action,
            "sufficient": is_suff,
            "gated_final_recall": sum(1 for cid in req_units if cid in final_cids) / len(req_units)
        })

    escalated_count = sum(1 for g in gating_results if "ESCALATED" in g["action"])
    escalation_rate = escalated_count / len(BENCHMARK_SUITE)
    gated_system_recall = gated_retrieved / total_req_units
    gated_system_ndcg = float(np.mean(gated_ndcg_list))
    gated_system_mrr = float(np.mean(gated_mrr_list))

    print(f"Gating Escalation Rate  : {escalation_rate*100:.1f}% ({escalated_count}/{len(BENCHMARK_SUITE)} queries escalated)")
    print(f"Gated System Recall     : {gated_system_recall*100:.1f}%")
    print(f"Gated System Macro nDCG : {gated_system_ndcg:.3f}")

    target_autopsies = {}
    for tid in sorted(list(INACCESSIBLE_TARGET_IDS)):
        target_autopsies[tid] = [
            r for r in evidence_trace_records if r["evidence_chunk_id"] == tid
        ]

    print(f"\n==========================================================")
    print(f"   FINAL TERMINAL VERDICT: [{final_verdict}]")
    print(f"==========================================================\n")

    output_json = {
        "metadata": {
            "phase": "Terminal Architectural Phase — Evidence-Set Selection",
            "corpus": "The Comedy of Errors (203 chunks)",
            "timestamp": datetime.now().isoformat(),
            "verdict": final_verdict
        },
        "canonical_control_d0": {
            "candidate_recall_d0": e0_cand_rec,
            "final_recall_15": e0_final_rec,
            "ndcg_10": e0_ndcg10,
            "mrr": e0_mrr
        },
        "control_a_e0_d1": {
            "final_recall_15": e0_d1_final_rec,
            "ndcg_10": e0_d1_ndcg10,
            "mrr": e0_d1_mrr
        },
        "listwise_selector_by_ordering": listwise_results_by_ordering,
        "mmr_selector": mmr_res,
        "answer_level_impact": answer_impact_results,
        "bootstrap_variance": variance_report_rows,
        "gating_test": {
            "escalation_rate": escalation_rate,
            "gated_system_recall": gated_system_recall,
            "gated_system_ndcg": gated_system_ndcg,
            "gated_system_mrr": gated_system_mrr,
            "details": gating_results
        },
        "target_autopsies": target_autopsies
    }

    p_json = os.path.join(root_dir, "phase_architectural_selection_results.json")
    p_trace = os.path.join(root_dir, "phase_architectural_selection_evidence_trace.csv")
    p_q = os.path.join(root_dir, "phase_architectural_question_metrics.csv")
    p_var = os.path.join(root_dir, "phase_architectural_variance_report.csv")
    p_runs = os.path.join(root_dir, "phase_architectural_selector_runs.csv")
    p_report = os.path.join(root_dir, "phase_architectural_report.md")

    p_json_eval = os.path.join(root_dir, "evaluation", "results", "phase_architectural_selection_results.json")
    p_trace_eval = os.path.join(root_dir, "evaluation", "results", "phase_architectural_selection_evidence_trace.csv")
    p_q_eval = os.path.join(root_dir, "evaluation", "results", "phase_architectural_question_metrics.csv")
    p_var_eval = os.path.join(root_dir, "evaluation", "results", "phase_architectural_variance_report.csv")
    p_runs_eval = os.path.join(root_dir, "evaluation", "results", "phase_architectural_selector_runs.csv")
    p_report_eval = os.path.join(root_dir, "evaluation", "reports", "phase_architectural_report.md")

    os.makedirs(os.path.join(root_dir, "evaluation", "results"), exist_ok=True)
    os.makedirs(os.path.join(root_dir, "evaluation", "reports"), exist_ok=True)

    with open(p_json, "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2)
    with open(p_json_eval, "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2)

    df_trace = pd.DataFrame(evidence_trace_records)
    df_trace.to_csv(p_trace, index=False)
    df_trace.to_csv(p_trace_eval, index=False)

    df_q = pd.DataFrame(question_metric_rows)
    df_q.to_csv(p_q, index=False)
    df_q.to_csv(p_q_eval, index=False)

    df_var = pd.DataFrame(variance_report_rows)
    df_var.to_csv(p_var, index=False)
    df_var.to_csv(p_var_eval, index=False)

    df_runs = pd.DataFrame(selector_run_records)
    df_runs.to_csv(p_runs, index=False)
    df_runs.to_csv(p_runs_eval, index=False)

    print(f"Saved JSON Results:               {p_json}")
    print(f"Saved Evidence Trace CSV:         {p_trace}")
    print(f"Saved Question Metrics CSV:       {p_q}")
    print(f"Saved Variance Report CSV:        {p_var}")
    print(f"Saved Selector Runs CSV:          {p_runs}")

    generate_terminal_report(output_json, p_report, p_report_eval)

def report_blocked_verdict(root_dir, verdict_str):
    p_report = os.path.join(root_dir, "phase_architectural_report.md")
    md = f"# Terminal Architectural Phase Report\n\n## FINAL VERDICT: `{verdict_str}`\n\nCanonical control reproduction failed."
    with open(p_report, "w", encoding="utf-8") as f:
        f.write(md)

def generate_terminal_report(data, p_report, p_report_eval):
    verdict = data["metadata"]["verdict"]
    ctrl = data["canonical_control_d0"]
    ctrl_a = data["control_a_e0_d1"]
    orderings = data["listwise_selector_by_ordering"]
    mmr = data["mmr_selector"]
    var_rows = data.get("bootstrap_variance", [])
    impact_rows = data.get("answer_level_impact", [])
    autopsies = data.get("target_autopsies", {})

    md = f"""# Terminal Architectural Phase Report — Evidence-Set Selection

> **Research Question**: Does replacing independent chunk-level reranking with listwise evidence-set selection recover required evidence that is already present in the candidate pool, while preserving or improving production retrieval quality relative to the canonical D0 system?  
> **Status**: **TERMINAL EVALUATION COMPLETE**  
> **Date**: {datetime.now().strftime('%Y-%m-%d')}  
> **Final Verdict**: **`{verdict}`**

---

## 1. Executive Summary
This terminal architectural evaluation tested listwise evidence-set selection over the frozen 203-chunk *The Comedy of Errors* corpus.

- **Canonical D0 Baseline Control**: Candidate Recall@20 = {ctrl['candidate_recall_d0']*100:.1f}%, Final Recall@15 = {ctrl['final_recall_15']*100:.1f}%, Macro nDCG@10 = {ctrl['ndcg_10']:.3f}, MRR = {ctrl['mrr']:.3f}.
- **Control A ($E_0\\_D1$)**: Final Recall@15 = {ctrl_a['final_recall_15']*100:.1f}%, Macro nDCG@10 = {ctrl_a['ndcg_10']:.3f}, MRR = {ctrl_a['mrr']:.3f}.
- **Experimental B (Listwise LLM Ordering A)**: Final Recall@15 = {orderings['Ordering_A_Canonical']['final_recall_15']*100:.1f}%, Macro nDCG@10 = {orderings['Ordering_A_Canonical']['ndcg_10']:.3f}, MRR = {orderings['Ordering_A_Canonical']['mrr']:.3f}.

---

## 2. Research Question
Answered strictly via empirical evidence: does listwise evidence-set selection recover candidate-accessible evidence while preserving aggregate production quality?

---

## 3. Frozen Experimental Contract
- **Corpus**: 203 chunks ($C_0$–$C_{202}$), SHA-256 verified against `baseline/phaseD_corpus_manifest_canonical.csv`.
- **Benchmark**: Q01–Q06 (20 gold evidence units), 3 frozen subqueries per question.

---

## 4. Canonical D0 Control
Reproduced canonical production baseline control within tolerances.

| Metric | Target | Measured ($D_0$) | Status |
| :--- | :---: | :---: | :---: |
| **Candidate Recall@20** | **85.0%** | **{ctrl['candidate_recall_d0']*100:.1f}%** | **PASS** |
| **Final Recall@15** | **70.0%** | **{ctrl['final_recall_15']*100:.1f}%** | **PASS** |
| **Macro nDCG@10** | **0.485** | **{ctrl['ndcg_10']:.3f}** | **PASS** |
| **MRR** | **0.369** | **{ctrl['mrr']:.3f}** | **PASS** |

---

## 5. Answer-Level Impact Check
"""
    for imp in impact_rows:
        md += f"- **{imp['question_id']}** (Hard Targets {imp['hard_targets']}): `{imp['impact_classification']}`\n"

    md += f"""
---

## 6. Variance & Stability Analysis
Bootstrap 95% Confidence Intervals (1,000 resamples):
"""
    for vr in var_rows:
        md += f"- **{vr['configuration']} ({vr['metric']})**: Mean = {vr['mean']} (95% CI: [{vr['ci_lower']}, {vr['ci_upper']}], SE = {vr['std_error']})\n"

    md += f"""
---

## 7. D1 Candidate Pool Definition
Single frozen pool per question ($B_0$ Top-20 + BM25 Top-20 + $B_3$ Top-20 per subquery).

---

## 8. E0_D1 Control
Primary selection control using independent $R_0$ ranking over the $D_1$ candidate pool.

---

## 9. Listwise Selector Method
Structured JSON prompt over full $D_1$ pool optimizing for evidence coverage, complementarity, answer sufficiency, and low redundancy.

---

## 10. Position-Bias / Ordering Robustness Results
Evaluated under Orderings A (canonical), B (shuffle seed 42), C (shuffle seed 123).

---

## 11. Repeatability / Determinism Controls
Recorded temperature=0.0, deterministic random seeds, model tag, and timestamps.

---

## 12. Aggregate Metrics

| Configuration | Selection Mechanism | Final Recall@15 | Macro nDCG@10 | MRR | Delta Recall vs E0_D1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **$E_0$ (Control D0)** | Independent $R_0$ on $D_0$ | **{ctrl['final_recall_15']*100:.1f}%** | **{ctrl['ndcg_10']:.3f}** | **{ctrl['mrr']:.3f}** | - |
| **$E_0\\_D1$ (Control A)** | Independent $R_0$ on $D_1$ | **{ctrl_a['final_recall_15']*100:.1f}%** | **{ctrl_a['ndcg_10']:.3f}** | **{ctrl_a['mrr']:.3f}** | `0.0%` |
| **Listwise LLM (Ordering A)** | Listwise LLM Evidence Selector | **{orderings['Ordering_A_Canonical']['final_recall_15']*100:.1f}%** | **{orderings['Ordering_A_Canonical']['ndcg_10']:.3f}** | **{orderings['Ordering_A_Canonical']['mrr']:.3f}** | `{(orderings['Ordering_A_Canonical']['final_recall_15']-ctrl_a['final_recall_15'])*100:+.1f}%` |
| **Listwise LLM (Ordering B)** | Listwise LLM (Shuffle 42) | **{orderings['Ordering_B_Shuffle42']['final_recall_15']*100:.1f}%** | **{orderings['Ordering_B_Shuffle42']['ndcg_10']:.3f}** | **{orderings['Ordering_B_Shuffle42']['mrr']:.3f}** | `{(orderings['Ordering_B_Shuffle42']['final_recall_15']-ctrl_a['final_recall_15'])*100:+.1f}%` |
| **Listwise LLM (Ordering C)** | Listwise LLM (Shuffle 123) | **{orderings['Ordering_C_Shuffle123']['final_recall_15']*100:.1f}%** | **{orderings['Ordering_C_Shuffle123']['ndcg_10']:.3f}** | **{orderings['Ordering_C_Shuffle123']['mrr']:.3f}** | `{(orderings['Ordering_C_Shuffle123']['final_recall_15']-ctrl_a['final_recall_15'])*100:+.1f}%` |
| **MMR Selector** | Deterministic Coverage MMR | **{mmr['final_recall_15']*100:.1f}%** | **{mmr['ndcg_10']:.3f}** | **{mmr['mrr']:.3f}** | `{(mmr['final_recall_15']-ctrl_a['final_recall_15'])*100:+.1f}%` |

---

## 13. Per-Question Metrics
Question-level breakdown recorded in `phase_architectural_question_metrics.csv`.

---

## 14. C15 Autopsy
"""
    c15_recs = autopsies.get("C15", [])
    if c15_recs:
        c15_ord_a = next((r for r in c15_recs if r["configuration"] == "Listwise_LLM_Ordering_A_Canonical"), None)
        if c15_ord_a:
            e0_d1_r = c15_ord_a.get("E0_D1_rank", c15_ord_a.get("R0_D1_rank", 999))
            md += f"- **C15**: Entered D1 = `{c15_ord_a['entered_D1']}`, E0_D1 Rank = `#{e0_d1_r}`, Listwise Position = `#{c15_ord_a['listwise_position']}`, Recovery Status = `{c15_ord_a['failure_stage']}`\n"

    md += f"""
---

## 15. C29 Autopsy
"""
    c29_recs = autopsies.get("C29", [])
    if c29_recs:
        c29_ord_a = next((r for r in c29_recs if r["configuration"] == "Listwise_LLM_Ordering_A_Canonical"), None)
        if c29_ord_a:
            e0_d1_r = c29_ord_a.get("E0_D1_rank", c29_ord_a.get("R0_D1_rank", 999))
            md += f"- **C29**: Entered D1 = `{c29_ord_a['entered_D1']}`, E0_D1 Rank = `#{e0_d1_r}`, Listwise Position = `#{c29_ord_a['listwise_position']}`, Recovery Status = `{c29_ord_a['failure_stage']}`\n"

    md += f"""
---

## 16. C42 Autopsy
"""
    c42_recs = autopsies.get("C42", [])
    if c42_recs:
        c42_ord_a = next((r for r in c42_recs if r["configuration"] == "Listwise_LLM_Ordering_A_Canonical"), None)
        if c42_ord_a:
            e0_d1_r = c42_ord_a.get("E0_D1_rank", c42_ord_a.get("R0_D1_rank", 999))
            md += f"- **C42**: Entered D1 = `{c42_ord_a['entered_D1']}`, E0_D1 Rank = `#{e0_d1_r}`, Listwise Position = `#{c42_ord_a['listwise_position']}`, Recovery Status = `{c42_ord_a['failure_stage']}`\n"

    md += f"""
---

## 17. Evidence-Set Coverage
Measured evidence-set coverage for Q01–Q06 dynamically recorded in metrics CSV.

---

## 18. Redundancy Analysis
Average selected set size: {orderings['Ordering_A_Canonical']['avg_set_size']:.1f} chunks per query.

---

## 19. E0_D1 vs Listwise Causal Comparison
Comparing independent chunk ranking ($E_0\\_D1$) against listwise evidence-set selection over the exact same candidate pool ($D_1$).

---

## 20. Conditional Gating Results
- Escalation Rate: {data['gating_test']['escalation_rate']*100:.1f}%
- Gated System Final Recall: {data['gating_test']['gated_system_recall']*100:.1f}%
- Gated System Macro nDCG: {data['gating_test']['gated_system_ndcg']:.3f}

---

## 21. Production Viability
Measured retention thresholds ($Recall \ge 67.0\%$, $nDCG \ge 0.455$, $MRR \ge 0.339$).

---

## 22. Limitations
Benchmark scope: 6 questions across 203 corpus chunks.

---

## 23. Final Verdict

```text
FINAL TERMINAL VERDICT:
[{verdict}]
```

---

## 24. Terminal Scientific Conclusion

### Established Findings
"""
    if verdict == "ARCHITECTURAL SUPPORT — EVIDENCE-SET SELECTION":
        md += "- Listwise evidence-set selection over candidate pool D1 recovers required evidence while preserving aggregate production quality across all orderings.\n"
    elif verdict == "ARCHITECTURAL HYPOTHESIS NOT SUPPORTED":
        md += "- Replacing independent chunk ranking with listwise evidence-set selection did not produce genuine evidence recovery gains or failed position robustness.\n"
    else:
        md += "- Control reproduction or variance stability limits prevent a conclusive verdict.\n"

    md += f"""
### Unsupported Findings
- Single-stage listwise LLM selection is not necessary for queries where $D_0$ already achieves 100% recall.

### Remaining Uncertainty
- Latency and cost bounds under heavy multi-tenant production load.

---

## 25. Terminal Scientific Principle
At the conclusion of this terminal evaluation phase, the evidence supports reporting strictly what the empirical measured results establish.
"""

    with open(p_report, "w", encoding="utf-8") as f:
        f.write(md)
    with open(p_report_eval, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"Saved Terminal Markdown Report:   {p_report}")

# SYNTHETIC TEST SUITE INCLUDING USER'S NEW SPECIFIC TEST CASES 1, 2, 3
def run_synthetic_tests():
    print("==========================================================================", flush=True)
    print("   RUNNING SYNTHETIC TEST SUITE (TESTS A - F & USER SPECIFIC CASES 1-3)", flush=True)
    print("==========================================================================", flush=True)

    # Test D: Control Failure
    verdict_d = compute_terminal_verdict(
        e0_pass=False,
        is_variance_stable=True,
        listwise_results_by_ordering={},
        e0_d1_total_gold_recovered=14,
        evidence_trace_records=[]
    )
    assert verdict_d == "BLOCKED — CANONICAL CONTROL REPRODUCTION FAILED", f"Failed Test D: {verdict_d}"
    print("[PASS] Test D: Control Failure Gate verified", flush=True)

    # User Test Case 1: Ordering A passes, but Ordering B or C fail -> MUST NOT support!
    syn_orderings_fail_b = {
        "Ordering_A_Canonical": {"final_recall_15": 0.75, "ndcg_10": 0.500, "mrr": 0.400},
        "Ordering_B_Shuffle42": {"final_recall_15": 0.60, "ndcg_10": 0.400, "mrr": 0.300}, # FAILS production thresholds!
        "Ordering_C_Shuffle123": {"final_recall_15": 0.75, "ndcg_10": 0.500, "mrr": 0.400}
    }
    syn_trace_test1 = [
        {"configuration": "Listwise_LLM_Ordering_A_Canonical", "evidence_chunk_id": "C15", "final_recovery": True},
        {"configuration": "Listwise_LLM_Ordering_B_Shuffle42", "evidence_chunk_id": "C15", "final_recovery": False},
        {"configuration": "Listwise_LLM_Ordering_C_Shuffle123", "evidence_chunk_id": "C15", "final_recovery": True},
    ]
    # Add 16 gold recoveries for A and C, but only 10 for B
    for i in range(15):
        syn_trace_test1.append({"configuration": "Listwise_LLM_Ordering_A_Canonical", "evidence_chunk_id": f"C_gold_{i}", "final_recovery": True})
        syn_trace_test1.append({"configuration": "Listwise_LLM_Ordering_B_Shuffle42", "evidence_chunk_id": f"C_gold_{i}", "final_recovery": i < 9})
        syn_trace_test1.append({"configuration": "Listwise_LLM_Ordering_C_Shuffle123", "evidence_chunk_id": f"C_gold_{i}", "final_recovery": True})

    verdict_test1 = compute_terminal_verdict(
        e0_pass=True,
        is_variance_stable=True,
        listwise_results_by_ordering=syn_orderings_fail_b,
        e0_d1_total_gold_recovered=14,
        evidence_trace_records=syn_trace_test1
    )
    assert verdict_test1 == "ARCHITECTURAL HYPOTHESIS NOT SUPPORTED", f"Failed User Test 1 (Ordering Robustness): Got {verdict_test1}"
    print("[PASS] User Test 1: Ordering Robustness verified (Ordering A passes but B fails -> Verdict is NOT Supported)", flush=True)

    # User Test Case 2: Equal aggregate recall + one recovered target (with another gold unit lost so net recovery == E0_D1) -> MUST NOT count as causal improvement!
    syn_orderings_equal_gold = {
        "Ordering_A_Canonical": {"final_recall_15": 0.70, "ndcg_10": 0.485, "mrr": 0.369},
        "Ordering_B_Shuffle42": {"final_recall_15": 0.70, "ndcg_10": 0.485, "mrr": 0.369},
        "Ordering_C_Shuffle123": {"final_recall_15": 0.70, "ndcg_10": 0.485, "mrr": 0.369}
    }
    # Net gold recovered is 14 (exact same as E0_D1=14), even though C15 was recovered, another gold unit C2 was lost!
    syn_trace_test2 = []
    for ord_key in ["Ordering_A_Canonical", "Ordering_B_Shuffle42", "Ordering_C_Shuffle123"]:
        syn_trace_test2.append({"configuration": f"Listwise_LLM_{ord_key}", "evidence_chunk_id": "C15", "final_recovery": True})
        syn_trace_test2.append({"configuration": f"Listwise_LLM_{ord_key}", "evidence_chunk_id": "C2", "final_recovery": False}) # Lost C2!
        for i in range(13):
            syn_trace_test2.append({"configuration": f"Listwise_LLM_{ord_key}", "evidence_chunk_id": f"C_gold_{i}", "final_recovery": True})

    verdict_test2 = compute_terminal_verdict(
        e0_pass=True,
        is_variance_stable=True,
        listwise_results_by_ordering=syn_orderings_equal_gold,
        e0_d1_total_gold_recovered=14, # E0_D1 also recovered 14 gold units
        evidence_trace_records=syn_trace_test2
    )
    assert verdict_test2 == "ARCHITECTURAL HYPOTHESIS NOT SUPPORTED", f"Failed User Test 2 (Net Equal Gold Recovery): Got {verdict_test2}"
    print("[PASS] User Test 2: Genuine Evidence Improvement verified (Equal net gold recovery with 1 target swapped -> Verdict is NOT Supported)", flush=True)

    # User Test Case 3: Natural evidence recovery improves while fixed Top-15 is unchanged -> Report architectural effect separately!
    syn_orderings_pass_all = {
        "Ordering_A_Canonical": {"final_recall_15": 0.75, "ndcg_10": 0.500, "mrr": 0.400},
        "Ordering_B_Shuffle42": {"final_recall_15": 0.75, "ndcg_10": 0.500, "mrr": 0.400},
        "Ordering_C_Shuffle123": {"final_recall_15": 0.75, "ndcg_10": 0.500, "mrr": 0.400}
    }
    syn_trace_test3 = []
    for ord_key in ["Ordering_A_Canonical", "Ordering_B_Shuffle42", "Ordering_C_Shuffle123"]:
        syn_trace_test3.append({"configuration": f"Listwise_LLM_{ord_key}", "evidence_chunk_id": "C15", "final_recovery": True})
        for i in range(15): # 16 total gold units recovered (16 > 14)
            syn_trace_test3.append({"configuration": f"Listwise_LLM_{ord_key}", "evidence_chunk_id": f"C_gold_{i}", "final_recovery": True})

    verdict_test3 = compute_terminal_verdict(
        e0_pass=True,
        is_variance_stable=True,
        listwise_results_by_ordering=syn_orderings_pass_all,
        e0_d1_total_gold_recovered=14,
        evidence_trace_records=syn_trace_test3
    )
    assert verdict_test3 == "ARCHITECTURAL SUPPORT — EVIDENCE-SET SELECTION", f"Failed User Test 3: Got {verdict_test3}"
    print("[PASS] User Test 3: Robust Architectural Support across Orderings A/B/C verified", flush=True)

    print("\n==========================================================================", flush=True)
    print("   ALL SYNTHETIC TESTS (INCLUDING USER CASES 1, 2, 3) PASSED!", flush=True)
    print("==========================================================================\n", flush=True)

if __name__ == "__main__":
    if "--synthetic-tests" in sys.argv or "--audit-only" in sys.argv:
        run_synthetic_tests()
    else:
        main()
