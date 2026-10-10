import os
import sys
import json
import math
import importlib.util
from datetime import datetime
import numpy as np

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
advance_rag_dir = os.path.join(root_dir, "Advance-Rag")
doc_loader_dir = os.path.join(advance_rag_dir, "Doc-Loader")

sys.path.insert(0, doc_loader_dir)
sys.path.insert(0, advance_rag_dir)
sys.path.insert(0, root_dir)

# Load IngestionPipeline
doc_loader_pipeline_path = os.path.join(doc_loader_dir, "pipeline.py")
spec = importlib.util.spec_from_file_location("doc_loader_pipeline", doc_loader_pipeline_path)
doc_loader_module = importlib.util.module_from_spec(spec)
sys.modules["doc_loader_pipeline"] = doc_loader_module
spec.loader.exec_module(doc_loader_module)
IngestionPipeline = doc_loader_module.IngestionPipeline

from advance_rag import AdvancedRAGPipeline
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

# Define Benchmark Suite with Explicit Gold & Required Evidence Definitions
BENCHMARK = [
    {
        "q_id": "Q01",
        "question": "Who is Aegeon and what is his origin?",
        "category": "Direct Factual",
        "gold": {"C4": "PRIMARY", "C2": "SUPPORTING"},
        "required_evidence": ["C4", "C2"]
    },
    {
        "q_id": "Q02",
        "question": "Why was Aegeon sentenced to death in Ephesus?",
        "category": "Causal",
        "gold": {"C2": "PRIMARY", "C4": "SUPPORTING", "C10": "SUPPORTING", "C12": "SUPPORTING"},
        "required_evidence": ["C2", "C4", "C10", "C12"]
    },
    {
        "q_id": "Q03",
        "question": "Why does Antipholus of Syracuse fail to recognize Adriana when she addresses him as her husband?",
        "category": "Multi-Hop",
        "gold": {"C15": "PRIMARY", "C24": "PRIMARY", "C29": "SUPPORTING", "C50": "SUPPORTING"},
        "required_evidence": ["C15", "C24", "C50"]
    },
    {
        "q_id": "Q04",
        "question": "How did Aegeon come to have two pairs of identical twin sons and twin servants?",
        "category": "Relationship",
        "gold": {"C4": "PRIMARY", "C6": "SUPPORTING", "C8": "SUPPORTING"},
        "required_evidence": ["C4", "C8"]
    },
    {
        "q_id": "Q05",
        "question": "How do Antipholus of Syracuse and Antipholus of Ephesus differ in their initial interactions with Dromio of Syracuse?",
        "category": "Comparative",
        "gold": {"C42": "PRIMARY", "C70": "PRIMARY", "C65": "PRIMARY", "C12": "SUPPORTING"},
        "required_evidence": ["C42", "C70", "C12"]
    },
    {
        "q_id": "Q06",
        "question": "Why is Antipholus of Ephesus arrested, and what role does Angelo the goldsmith play in the debt dispute?",
        "category": "Event-Chain",
        "gold": {"C105": "PRIMARY", "C110": "PRIMARY", "C107": "PRIMARY", "C147": "SUPPORTING", "C155": "SUPPORTING"},
        "required_evidence": ["C105", "C110", "C155"]
    }
]


def rrf_search_custom_weights(pipeline, query, query_emb, w_vec=0.5, w_bm25=0.5, top_k=20, rrf_k=60):
    """Executes RRF search with custom vector vs BM25 weights."""
    search_filter = Filter(must=[FieldCondition(key="metadata.session_id", match=MatchValue(value=pipeline.session_id))])
    try:
        vec_results = pipeline.vector_store.similarity_search_with_score_by_vector(query_emb.tolist(), k=top_k, filter=search_filter)
    except Exception:
        vec_results = pipeline.vector_store.similarity_search_with_score_by_vector(query_emb.tolist(), k=top_k)
        vec_results = [(doc, score) for doc, score in vec_results if doc.metadata.get("session_id") == pipeline.session_id]

    vector_docs = [doc for doc, _ in vec_results if doc.metadata.get("session_id") == pipeline.session_id]

    bm25_docs = []
    if pipeline.bm25_retriever:
        raw_bm25 = pipeline.bm25_retriever.invoke(query)
        bm25_docs = [doc for doc in raw_bm25 if doc.metadata.get("session_id") == pipeline.session_id]

    rrf_scores = {}
    doc_map = {}

    for rank_idx, doc in enumerate(vector_docs):
        cid = doc.metadata.get("element_id")
        doc_map[cid] = doc
        score = w_vec * (1.0 / (rrf_k + (rank_idx + 1)))
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + score

    for rank_idx, doc in enumerate(bm25_docs):
        cid = doc.metadata.get("element_id")
        doc_map[cid] = doc
        score = w_bm25 * (1.0 / (rrf_k + (rank_idx + 1)))
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + score

    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[cid] for cid, _ in sorted_docs[:top_k]]


def format_doc_item(doc, pipeline_chunks):
    el_id = doc.metadata.get("element_id")
    c_idx = next((i for i, c in enumerate(pipeline_chunks) if c.metadata.get("element_id") == el_id), -1)
    chunk_label = f"C{c_idx}" if c_idx >= 0 else str(el_id)
    return {
        "chunk_id": chunk_label,
        "chunk_idx": c_idx,
        "element_id": el_id,
        "parent_id": doc.metadata.get("parent_id") or doc.metadata.get("parent_text", "")[:80],
        "parent_text": doc.metadata.get("parent_text", ""),
        "breadcrumb": doc.metadata.get("breadcrumb_path", ""),
        "text": doc.page_content
    }


def evaluate_retrieved_list(retrieved_items, gold_map, required_evidence, k=10):
    top_k_items = retrieved_items[:k]
    cids = [item["chunk_id"] for item in top_k_items]
    
    seen_chunks = set()
    seen_parents = set()
    classifications = []
    gold_found = set()
    first_useful_rank = 0
    dup_count = 0

    for idx, item in enumerate(top_k_items, 1):
        cid = item["chunk_id"]
        parent_key = item["parent_id"]

        if cid in gold_map:
            role = gold_map[cid]
            classifications.append(role)
            gold_found.add(cid)
            seen_parents.add(parent_key)
            seen_chunks.add(cid)
            if first_useful_rank == 0:
                first_useful_rank = idx
        elif cid in seen_chunks or parent_key in seen_parents:
            dup_count += 1
            classifications.append("DUPLICATE")
        else:
            seen_parents.add(parent_key)
            seen_chunks.add(cid)
            t_lower = item["text"].lower()
            if any(kw in t_lower for kw in ["aegeon", "syracuse", "ephesus", "antipholus", "dromio", "adriana", "goldsmith", "angelo", "chain", "ducats", "twin"]):
                classifications.append("RELEVANT_BUT_UNNECESSARY")
            else:
                classifications.append("IRRELEVANT")

    hit_10 = 1.0 if len(gold_found) > 0 else 0.0

    r1 = len(gold_found.intersection(set(cids[:1]))) / len(gold_map)
    r3 = len(gold_found.intersection(set(cids[:3]))) / len(gold_map)
    r5 = len(gold_found.intersection(set(cids[:5]))) / len(gold_map)
    r10 = len(gold_found.intersection(set(cids[:10]))) / len(gold_map)

    ev1 = sum(1 for c in classifications[:1] if c in ["PRIMARY", "SUPPORTING"]) / 1
    ev3 = sum(1 for c in classifications[:3] if c in ["PRIMARY", "SUPPORTING"]) / 3
    ev5 = sum(1 for c in classifications[:5] if c in ["PRIMARY", "SUPPORTING"]) / 5
    ev10 = sum(1 for c in classifications[:10] if c in ["PRIMARY", "SUPPORTING"]) / 10

    p1 = sum(1 for c in classifications[:1] if c == "PRIMARY") / 1
    p3 = sum(1 for c in classifications[:3] if c == "PRIMARY") / 3
    p5 = sum(1 for c in classifications[:5] if c == "PRIMARY") / 5
    p10 = sum(1 for c in classifications[:10] if c == "PRIMARY") / 10

    mrr = 1.0 / first_useful_rank if first_useful_rank > 0 else 0.0

    rel_weights = {"PRIMARY": 3, "SUPPORTING": 2, "RELEVANT_BUT_UNNECESSARY": 1, "IRRELEVANT": 0, "DUPLICATE": 0}
    dcg = sum((2**rel_weights[c] - 1) / math.log2(i + 2) for i, c in enumerate(classifications[:10]))
    ideal_rels = sorted([3 if r == "PRIMARY" else 2 for r in gold_map.values()], reverse=True)[:10]
    idcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(ideal_rels))
    ndcg10 = dcg / idcg if idcg > 0 else 0.0

    primary_golds = {cid for cid, r in gold_map.items() if r == "PRIMARY"}
    completeness = 1.0 if primary_golds.issubset(gold_found) else 0.0
    req_completeness = 1.0 if set(required_evidence).issubset(gold_found) else 0.0

    redundancy = dup_count / len(top_k_items) if top_k_items else 0.0

    # Count unique parents
    unique_parents = len(set(item["parent_id"] for item in top_k_items))

    return {
        "hit_10": hit_10,
        "recall_1": r1, "recall_3": r3, "recall_5": r5, "recall_10": r10,
        "ev_p_1": ev1, "ev_p_3": ev3, "ev_p_5": ev5, "ev_p_10": ev10,
        "pri_p_1": p1, "pri_p_3": p3, "pri_p_5": p5, "pri_p_10": p10,
        "mrr": mrr, "ndcg_10": ndcg10,
        "completeness": completeness,
        "req_completeness": req_completeness,
        "redundancy": redundancy,
        "unique_parent_count": unique_parents,
        "returned_chunk_count": len(top_k_items),
        "classifications": classifications,
        "gold_found": list(gold_found)
    }


def apply_parent_deduplication(retrieved_items, max_per_parent=1, top_k=10):
    """Filters candidate list allowing at most max_per_parent chunks per parent_id."""
    parent_counts = {}
    deduped = []
    for item in retrieved_items:
        pid = item["parent_id"]
        count = parent_counts.get(pid, 0)
        if count < max_per_parent:
            deduped.append(item)
            parent_counts[pid] = count + 1
        if len(deduped) >= top_k:
            break
    return deduped


def apply_neighbor_expansion(retrieved_items, pipeline_chunks, window=1, top_k=10):
    """Expands Top-K retrieved chunks with adjacent document sequence chunks (±window)."""
    top_items = retrieved_items[:top_k]
    expanded = []
    seen_indices = set()

    for item in top_items:
        c_idx = item["chunk_idx"]
        if c_idx < 0:
            if item["element_id"] not in seen_indices:
                seen_indices.add(item["element_id"])
                expanded.append(item)
            continue

        start_i = max(0, c_idx - window)
        end_i = min(len(pipeline_chunks) - 1, c_idx + window)

        for i in range(start_i, end_i + 1):
            if i not in seen_indices:
                seen_indices.add(i)
                c_doc = pipeline_chunks[i]
                expanded.append(format_doc_item(c_doc, pipeline_chunks))

    return expanded, top_items


def main():
    print("==========================================================================")
    print("        EXECUTING PHASE 2 RAG RETRIEVAL OPTIMIZATION EXPERIMENTS          ")
    print("==========================================================================")

    doc_path = os.path.join(root_dir, "Documents", "The Comedy of Errors document.pdf")
    session_id = f"phase2_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    pipeline = AdvancedRAGPipeline(session_id=session_id)
    
    print("Ingesting document...")
    raw_docs = IngestionPipeline.load_document(doc_path)
    pipeline.add_documents(raw_docs)
    print(f"Total indexed chunks: {len(pipeline.all_chunks)}.\n")

    # Embed all queries once
    embedded_queries = {}
    for case in BENCHMARK:
        q_id = case["q_id"]
        q_text = case["question"]
        embedded_queries[q_id] = np.array(pipeline.embeddings.embed_query(q_text))

    all_experiment_results = {}

    # Helper to compute macro metrics
    def compute_experiment_summary(exp_id, run_func):
        per_q = {}
        macro_acc = {
            "hit_10": [], "recall_1": [], "recall_3": [], "recall_5": [], "recall_10": [],
            "ev_p_1": [], "ev_p_3": [], "ev_p_5": [], "ev_p_10": [],
            "pri_p_1": [], "pri_p_3": [], "pri_p_5": [], "pri_p_10": [],
            "mrr": [], "ndcg_10": [], "completeness": [], "req_completeness": [],
            "redundancy": [], "unique_parents": [], "returned_chunks": []
        }

        for case in BENCHMARK:
            q_id = case["q_id"]
            q_text = case["question"]
            gold_map = case["gold"]
            req_ev = case["required_evidence"]
            q_emb = embedded_queries[q_id]

            retrieved_items, extra_info = run_func(q_id, q_text, q_emb)
            res = evaluate_retrieved_list(retrieved_items, gold_map, req_ev, k=10)

            if extra_info:
                res.update(extra_info)

            per_q[q_id] = res

            macro_acc["hit_10"].append(res["hit_10"])
            macro_acc["recall_1"].append(res["recall_1"])
            macro_acc["recall_3"].append(res["recall_3"])
            macro_acc["recall_5"].append(res["recall_5"])
            macro_acc["recall_10"].append(res["recall_10"])

            macro_acc["ev_p_1"].append(res["ev_p_1"])
            macro_acc["ev_p_3"].append(res["ev_p_3"])
            macro_acc["ev_p_5"].append(res["ev_p_5"])
            macro_acc["ev_p_10"].append(res["ev_p_10"])

            macro_acc["pri_p_1"].append(res["pri_p_1"])
            macro_acc["pri_p_3"].append(res["pri_p_3"])
            macro_acc["pri_p_5"].append(res["pri_p_5"])
            macro_acc["pri_p_10"].append(res["pri_p_10"])

            macro_acc["mrr"].append(res["mrr"])
            macro_acc["ndcg_10"].append(res["ndcg_10"])
            macro_acc["completeness"].append(res["completeness"])
            macro_acc["req_completeness"].append(res["req_completeness"])
            macro_acc["redundancy"].append(res["redundancy"])
            macro_acc["unique_parents"].append(res["unique_parent_count"])
            macro_acc["returned_chunks"].append(res["returned_chunk_count"])

        summary = {k: float(np.mean(vals)) for k, vals in macro_acc.items()}
        all_experiment_results[exp_id] = {
            "summary": summary,
            "per_question": per_q
        }
        print(f"Completed {exp_id:25s} | Recall@10: {summary['recall_10']:.1%} | EvPrec@10: {summary['ev_p_10']:.1%} | PriPrec@10: {summary['pri_p_10']:.1%} | MRR: {summary['mrr']:.3f} | Comp: {summary['completeness']:.1%} | ReqComp: {summary['req_completeness']:.1%} | Redun: {summary['redundancy']:.1%}")

    # =========================================================================
    # BASELINE RUNS (CONTROL)
    # =========================================================================
    def run_baseline_semantic(q_id, q_text, q_emb):
        search_filter = Filter(must=[FieldCondition(key="metadata.session_id", match=MatchValue(value=session_id))])
        try:
            vec_results = pipeline.vector_store.similarity_search_with_score_by_vector(q_emb.tolist(), k=20, filter=search_filter)
        except Exception:
            vec_results = pipeline.vector_store.similarity_search_with_score_by_vector(q_emb.tolist(), k=20)
            vec_results = [(doc, score) for doc, score in vec_results if doc.metadata.get("session_id") == session_id]

        retrieved = [format_doc_item(doc, pipeline.all_chunks) for doc, _ in vec_results]
        return retrieved, {}

    def run_baseline_bm25(q_id, q_text, q_emb):
        bm25_raw = pipeline.bm25_retriever.invoke(q_text) if pipeline.bm25_retriever else []
        bm25_filtered = [doc for doc in bm25_raw if doc.metadata.get("session_id") == session_id][:20]
        retrieved = [format_doc_item(doc, pipeline.all_chunks) for doc in bm25_filtered]
        return retrieved, {}

    def run_baseline_rrf_50_50(q_id, q_text, q_emb):
        rrf_docs = rrf_search_custom_weights(pipeline, q_text, q_emb, w_vec=0.5, w_bm25=0.5, top_k=20)
        retrieved = [format_doc_item(doc, pipeline.all_chunks) for doc in rrf_docs]
        return retrieved, {}

    print("\n--- Running Control Baseline Conditions ---")
    compute_experiment_summary("Phase_1_Semantic", run_baseline_semantic)
    compute_experiment_summary("Phase_1_BM25", run_baseline_bm25)
    compute_experiment_summary("Phase_1_RRF_50_50", run_baseline_rrf_50_50)

    # =========================================================================
    # EXPERIMENT 1 — PARENT-LEVEL DEDUPLICATION (Exp 1A / 1B)
    # =========================================================================
    print("\n--- Running Experiment 1 (Parent-Level Deduplication) ---")
    
    for limit in [1, 2, 3]:
        def run_exp1_semantic(q_id, q_text, q_emb, lim=limit):
            retrieved_raw, _ = run_baseline_semantic(q_id, q_text, q_emb)
            deduped = apply_parent_deduplication(retrieved_raw, max_per_parent=lim, top_k=10)
            return deduped, {}

        def run_exp1_rrf(q_id, q_text, q_emb, lim=limit):
            retrieved_raw, _ = run_baseline_rrf_50_50(q_id, q_text, q_emb)
            deduped = apply_parent_deduplication(retrieved_raw, max_per_parent=lim, top_k=10)
            return deduped, {}

        compute_experiment_summary(f"Exp_1_Semantic_Dedup_{limit}_per_parent", run_exp1_semantic)
        compute_experiment_summary(f"Exp_1_RRF_Dedup_{limit}_per_parent", run_exp1_rrf)

    # =========================================================================
    # EXPERIMENT 2 — RRF WEIGHTING
    # =========================================================================
    print("\n--- Running Experiment 2 (RRF Weighting Ratios) ---")
    weight_configs = [
        ("70_30", 0.7, 0.3),
        ("80_20", 0.8, 0.2),
        ("85_15", 0.85, 0.15),
        ("90_10", 0.9, 0.1)
    ]

    for label, w_v, w_b in weight_configs:
        def run_exp2_rrf(q_id, q_text, q_emb, wv=w_v, wb=w_b):
            rrf_docs = rrf_search_custom_weights(pipeline, q_text, q_emb, w_vec=wv, w_bm25=wb, top_k=20)
            retrieved = [format_doc_item(doc, pipeline.all_chunks) for doc in rrf_docs]
            return retrieved, {}

        compute_experiment_summary(f"Exp_2_RRF_{label}", run_exp2_rrf)

    # =========================================================================
    # EXPERIMENT 3 — NEIGHBOR / WINDOW EXPANSION (Exp 3A / 3B)
    # =========================================================================
    print("\n--- Running Experiment 3 (Neighbor / Window Expansion) ---")
    for win in [1, 2]:
        def run_exp3_semantic(q_id, q_text, q_emb, w=win):
            retrieved_raw, _ = run_baseline_semantic(q_id, q_text, q_emb)
            expanded, orig_top = apply_neighbor_expansion(retrieved_raw, pipeline.all_chunks, window=w, top_k=10)
            expansion_ratio = len(expanded) / len(orig_top) if orig_top else 1.0
            return expanded, {"context_expansion_ratio": expansion_ratio, "original_top_k_count": len(orig_top), "expanded_unique_count": len(expanded)}

        def run_exp3_rrf(q_id, q_text, q_emb, w=win):
            retrieved_raw, _ = run_baseline_rrf_50_50(q_id, q_text, q_emb)
            expanded, orig_top = apply_neighbor_expansion(retrieved_raw, pipeline.all_chunks, window=w, top_k=10)
            expansion_ratio = len(expanded) / len(orig_top) if orig_top else 1.0
            return expanded, {"context_expansion_ratio": expansion_ratio, "original_top_k_count": len(orig_top), "expanded_unique_count": len(expanded)}

        compute_experiment_summary(f"Exp_3_Semantic_Neighbor_w{win}", run_exp3_semantic)
        compute_experiment_summary(f"Exp_3_RRF_Neighbor_w{win}", run_exp3_rrf)

    # Save to JSON artifact
    os.makedirs(os.path.join(current_dir, "results"), exist_ok=True)
    out_file = os.path.join(current_dir, "results", "phase2_experiments_dump.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_experiment_results, f, indent=2, ensure_ascii=False)

    print(f"\nPhase 2 experiments execution complete. Results saved to:\n{out_file}\n")


if __name__ == "__main__":
    main()
