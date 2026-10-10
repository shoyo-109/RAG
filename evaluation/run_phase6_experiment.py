import os
import sys
import json
import math
import importlib.util
from datetime import datetime
import numpy as np

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
advance_rag_dir = os.path.join(root_dir, "Advance-Rag")
doc_loader_dir = os.path.join(advance_rag_dir, "Doc-Loader")

sys.path.insert(0, doc_loader_dir)
sys.path.insert(0, advance_rag_dir)
sys.path.insert(0, root_dir)

from dotenv import load_dotenv
load_dotenv()

# Load IngestionPipeline
doc_loader_pipeline_path = os.path.join(doc_loader_dir, "pipeline.py")
spec = importlib.util.spec_from_file_location("doc_loader_pipeline", doc_loader_pipeline_path)
doc_loader_module = importlib.util.module_from_spec(spec)
sys.modules["doc_loader_pipeline"] = doc_loader_module
spec.loader.exec_module(doc_loader_module)
IngestionPipeline = doc_loader_module.IngestionPipeline

from advance_rag import AdvancedRAGPipeline
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

# Canonical Benchmark Suite & Gold Mapping
BENCHMARK_SUITE = [
    {
        "q_id": "Q01",
        "question": "Who is Aegeon and what is his origin?",
        "total_required_evidence": 2, # C4, C2
        "gold_map": {"C4": "PRIMARY", "C2": "SUPPORTING"},
        "required_evidence": ["C4", "C2"],
        "claims": [
            {"claim_id": "Q01_C1", "desc": "Aegeon is a merchant from Syracuse.", "required_chunks": ["C4"]},
            {"claim_id": "Q01_C2", "desc": "Married, Epidamnum trip, twin sons birth.", "required_chunks": ["C4"]}
        ]
    },
    {
        "q_id": "Q02",
        "question": "Why was Aegeon sentenced to death in Ephesus?",
        "total_required_evidence": 4, # C2, C4, C10, C12
        "gold_map": {"C2": "PRIMARY", "C4": "SUPPORTING", "C10": "SUPPORTING", "C12": "SUPPORTING"},
        "required_evidence": ["C2", "C4", "C10", "C12"],
        "claims": [
            {"claim_id": "Q02_C1", "desc": "Ephesus law prohibiting Syracusians on pain of death/1000 marks ransom.", "required_chunks": ["C2"]},
            {"claim_id": "Q02_C2", "desc": "Aegeon's wealth under 100 marks, unable to pay ransom.", "required_chunks": ["C2"]}
        ]
    },
    {
        "q_id": "Q03",
        "question": "Why does Antipholus of Syracuse fail to recognize Adriana when she addresses him as her husband?",
        "total_required_evidence": 4, # C15, C24, C29, C61
        "gold_map": {"C15": "PRIMARY", "C24": "PRIMARY", "C29": "SUPPORTING", "C61": "SUPPORTING"},
        "required_evidence": ["C15", "C24", "C29", "C61"],
        "claims": [
            {"claim_id": "Q03_C1", "desc": "Adriana married to Antipholus of Ephesus.", "required_chunks": ["C24"]},
            {"claim_id": "Q03_C2", "desc": "Antipholus of Syracuse visitor unaware of twin brother.", "required_chunks": ["C15"]},
            {"claim_id": "Q03_C3", "desc": "Antipholus of Syracuse mistaken for twin brother by Adriana.", "required_chunks": ["C61"]}
        ]
    },
    {
        "q_id": "Q04",
        "question": "How did Aegeon come to have two pairs of identical twin sons and twin servants?",
        "total_required_evidence": 3, # C4, C6, C8
        "gold_map": {"C4": "PRIMARY", "C6": "SUPPORTING", "C8": "SUPPORTING"},
        "required_evidence": ["C4", "C6", "C8"],
        "claims": [
            {"claim_id": "Q04_C1", "desc": "Twin sons born in Epidamnum.", "required_chunks": ["C4"]},
            {"claim_id": "Q04_C2", "desc": "Poor woman's twin boys bought as servants.", "required_chunks": ["C4"]},
            {"claim_id": "Q04_C3", "desc": "Storm, spare mast, shipwreck separation.", "required_chunks": ["C6", "C8"]}
        ]
    },
    {
        "q_id": "Q05",
        "question": "How do Antipholus of Syracuse and Antipholus of Ephesus differ in their initial interactions with Dromio of Syracuse?",
        "total_required_evidence": 3, # C42, C65, C12
        "gold_map": {"C42": "PRIMARY", "C65": "PRIMARY", "C12": "SUPPORTING"},
        "required_evidence": ["C42", "C65", "C12"],
        "claims": [
            {"claim_id": "Q05_C1", "desc": "Syracuse twin trusts money, scolds over confusion.", "required_chunks": ["C42", "C12"]},
            {"claim_id": "Q05_C2", "desc": "Ephesus twin locked out of house.", "required_chunks": ["C65"]}
        ]
    },
    {
        "q_id": "Q06",
        "question": "Why is Antipholus of Ephesus arrested, and what role does Angelo the goldsmith play in the debt dispute?",
        "total_required_evidence": 4, # C105, C107, C147, C155
        "gold_map": {"C105": "PRIMARY", "C107": "PRIMARY", "C147": "SUPPORTING", "C155": "SUPPORTING"},
        "required_evidence": ["C105", "C107", "C147", "C155"],
        "claims": [
            {"claim_id": "Q06_C1", "desc": "Angelo delivered chain to Syracuse twin by mistake, demanded 200 ducats.", "required_chunks": ["C105", "C107"]},
            {"claim_id": "Q06_C2", "desc": "Ephesus twin denied chain, arrested for 200 ducats debt.", "required_chunks": ["C147", "C155"]}
        ]
    }
]


def format_chunk_item(doc, pipeline_chunks):
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


def execute_rrf_fusion(vector_results, bm25_results, w_vec=0.8, w_bm25=0.2, rrf_k=60):
    """Executes RRF fusion given vector and BM25 candidate lists."""
    rrf_scores = {}
    doc_map = {}
    vec_rank_map = {}
    bm25_rank_map = {}

    for rank_idx, item in enumerate(vector_results, 1):
        cid = item["chunk_id"]
        doc_map[cid] = item
        vec_rank_map[cid] = rank_idx
        score = w_vec * (1.0 / (rrf_k + rank_idx))
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + score

    for rank_idx, item in enumerate(bm25_results, 1):
        cid = item["chunk_id"]
        doc_map[cid] = item
        bm25_rank_map[cid] = rank_idx
        score = w_bm25 * (1.0 / (rrf_k + rank_idx))
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + score

    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    rrf_ranked = []
    for rrf_rank, (cid, score) in enumerate(sorted_docs, 1):
        doc = doc_map[cid]
        rrf_ranked.append({
            "chunk_id": cid,
            "element_id": doc["element_id"],
            "parent_id": doc["parent_id"],
            "parent_text": doc["parent_text"],
            "breadcrumb": doc["breadcrumb"],
            "text": doc["text"],
            "semantic_rank": vec_rank_map.get(cid, None),
            "bm25_rank": bm25_rank_map.get(cid, None),
            "rrf_score": score,
            "rrf_rank": rrf_rank
        })
    return rrf_ranked


def apply_parent_aware_selection(rrf_ranked_list, max_per_parent=None, top_k=10):
    """
    Filters RRF ranked candidate list using Parent-Aware selection.
    OPERATES AFTER RRF RANKING (preserves original RRF ranking order).
    max_per_parent: None (P0 - No constraint), 1 (P1), 2 (P2), 3 (P3).
    """
    if max_per_parent is None: # P0
        return rrf_ranked_list[:top_k]

    parent_counts = {}
    selected = []
    for item in rrf_ranked_list:
        pid = item["parent_id"]
        count = parent_counts.get(pid, 0)
        if count < max_per_parent:
            selected.append(item)
            parent_counts[pid] = count + 1
        if len(selected) >= top_k:
            break
    return selected


def calculate_secondary_metrics(selected_chunks, gold_map, top_k=10):
    """Calculates Evidence Precision, Primary Precision, MRR, nDCG, Redundancy Ratio."""
    retrieved_cids = [item["chunk_id"] for item in selected_chunks]
    
    seen_parents = set()
    dup_count = 0
    first_useful_rank = 0
    useful_count = 0
    primary_count = 0

    rel_map = {"PRIMARY": 3, "SUPPORTING": 2, "RELEVANT_BUT_UNNECESSARY": 1, "IRRELEVANT": 0, "DUPLICATE": 0}
    classifications = []

    for idx, item in enumerate(selected_chunks[:top_k], 1):
        cid = item["chunk_id"]
        pid = item["parent_id"]

        if cid in gold_map:
            role = gold_map[cid]
            classifications.append(role)
            useful_count += 1
            if role == "PRIMARY":
                primary_count += 1
            if first_useful_rank == 0:
                first_useful_rank = idx
            seen_parents.add(pid)
        elif pid in seen_parents:
            dup_count += 1
            classifications.append("DUPLICATE")
        else:
            seen_parents.add(pid)
            t_lower = item["text"].lower()
            if any(kw in t_lower for kw in ["aegeon", "syracuse", "ephesus", "antipholus", "dromio", "adriana", "goldsmith", "angelo", "chain", "ducats", "twin"]):
                classifications.append("RELEVANT_BUT_UNNECESSARY")
            else:
                classifications.append("IRRELEVANT")

    ev_precision = useful_count / top_k
    pri_precision = primary_count / top_k
    mrr = 1.0 / first_useful_rank if first_useful_rank > 0 else 0.0

    dcg = sum((2**rel_map.get(c, 0) - 1) / math.log2(i + 2) for i, c in enumerate(classifications[:top_k]))
    ideal_rels = sorted([3 if r == "PRIMARY" else 2 for r in gold_map.values()], reverse=True)[:top_k]
    idcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(ideal_rels))
    ndcg10 = dcg / idcg if idcg > 0 else 0.0

    redundancy_ratio = dup_count / len(selected_chunks) if selected_chunks else 0.0
    unique_parent_count = len(seen_parents)
    avg_chunks_per_parent = len(selected_chunks) / unique_parent_count if unique_parent_count > 0 else 0.0

    return {
        "ev_precision": ev_precision,
        "pri_precision": pri_precision,
        "mrr": mrr,
        "ndcg10": ndcg10,
        "redundancy_ratio": redundancy_ratio,
        "unique_parent_count": unique_parent_count,
        "avg_chunks_per_parent": avg_chunks_per_parent
    }


def main():
    print("==========================================================================", flush=True)
    print("      EXECUTING PHASE 6 — CANDIDATE DEPTH VS PARENT-AWARE SELECTION      ", flush=True)
    print("==========================================================================", flush=True)

    doc_path = os.path.join(root_dir, "Documents", "The Comedy of Errors document.pdf")
    session_id = f"phase6_exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    pipeline = AdvancedRAGPipeline(session_id=session_id)

    print("Ingesting document into pipeline...", flush=True)
    raw_docs = IngestionPipeline.load_document(doc_path)
    pipeline.add_documents(raw_docs)
    print(f"Total indexed chunks: {len(pipeline.all_chunks)}.\n", flush=True)

    candidate_depths = [20, 50, 100]
    parent_policies = [
        {"id": "P0", "name": "No Dedup", "max_per_parent": None},
        {"id": "P1", "name": "Max 1 / Parent", "max_per_parent": 1},
        {"id": "P2", "name": "Max 2 / Parent", "max_per_parent": 2},
        {"id": "P3", "name": "Max 3 / Parent", "max_per_parent": 3}
    ]

    # Matrix storage: (N, policy_id) -> list of case results
    matrix_results = {}
    for N in candidate_depths:
        for pol in parent_policies:
            matrix_results[(N, pol["id"])] = []

    # Pre-retrieve vector and BM25 candidates for all 6 questions at N=100
    pre_retrieved = {}
    for case in BENCHMARK_SUITE:
        q_id = case["q_id"]
        q_text = case["question"]
        q_emb = np.array(pipeline.embeddings.embed_query(q_text))

        search_filter = Filter(must=[FieldCondition(key="metadata.session_id", match=MatchValue(value=pipeline.session_id))])
        try:
            vec_100_res = pipeline.vector_store.similarity_search_with_score_by_vector(q_emb.tolist(), k=100, filter=search_filter)
        except Exception:
            vec_100_res = pipeline.vector_store.similarity_search_with_score_by_vector(q_emb.tolist(), k=100)
            vec_100_res = [(doc, score) for doc, score in vec_100_res if doc.metadata.get("session_id") == pipeline.session_id]

        semantic_100 = [format_chunk_item(doc, pipeline.all_chunks) for doc, _ in vec_100_res]

        bm25_raw = pipeline.bm25_retriever.invoke(q_text) if pipeline.bm25_retriever else []
        bm25_100 = [format_chunk_item(doc, pipeline.all_chunks) for doc in bm25_raw if doc.metadata.get("session_id") == pipeline.session_id][:100]

        pre_retrieved[q_id] = {
            "semantic": semantic_100,
            "bm25": bm25_100
        }

    # Execute 12-configuration matrix
    counterfactual_trace = []
    per_question_matrix = []

    for N in candidate_depths:
        for pol in parent_policies:
            pol_id = pol["id"]
            max_p = pol["max_per_parent"]

            print(f"--- Running Config: N={N:3d} | Policy: {pol_id:2s} ({pol['name']}) ---", flush=True)

            for case in BENCHMARK_SUITE:
                q_id = case["q_id"]
                req_units = case["required_evidence"]
                gold_map = case["gold_map"]
                claims = case["claims"]

                sem_cand = pre_retrieved[q_id]["semantic"][:N]
                bm25_cand = pre_retrieved[q_id]["bm25"][:N]

                # RRF Fusion over N candidates
                rrf_ranked = execute_rrf_fusion(sem_cand, bm25_cand, w_vec=0.8, w_bm25=0.2, rrf_k=60)

                # Parent-Aware Selection
                selected_top10 = apply_parent_aware_selection(rrf_ranked, max_per_parent=max_p, top_k=10)
                selected_cids = [item["chunk_id"] for item in selected_top10]

                # Required Evidence Recall
                retrieved_req_units = [cid for cid in req_units if cid in selected_cids]
                ev_recall = len(retrieved_req_units) / len(req_units) if req_units else 0.0

                # Claim Availability
                supported_claims = 0
                missing_claims = []
                for claim in claims:
                    if all(cid in selected_cids for cid in claim["required_chunks"]):
                        supported_claims += 1
                    else:
                        missing_claims.append(claim["claim_id"])
                claim_avail = supported_claims / len(claims) if claims else 0.0

                # Evidence Completeness
                ev_completeness = 1.0 if len(retrieved_req_units) == len(req_units) else 0.0

                # Secondary metrics
                sec_metrics = calculate_secondary_metrics(selected_top10, gold_map, top_k=10)

                case_res = {
                    "q_id": q_id,
                    "N": N,
                    "policy": pol_id,
                    "policy_name": pol["name"],
                    "total_required_evidence": len(req_units),
                    "retrieved_evidence_count": len(retrieved_req_units),
                    "retrieved_evidence_units": retrieved_req_units,
                    "ev_recall": ev_recall,
                    "total_claims": len(claims),
                    "supported_claims": supported_claims,
                    "claim_avail": claim_avail,
                    "missing_claims": missing_claims,
                    "ev_completeness": ev_completeness,
                    "selected_chunk_ids": selected_cids,
                    "secondary": sec_metrics
                }

                matrix_results[(N, pol_id)].append(case_res)

                per_question_matrix.append(case_res)

                # Counterfactual tracing for gold chunks
                for req_cid in req_units:
                    orig_rrf_item = next((item for item in rrf_ranked if item["chunk_id"] == req_cid), None)
                    orig_rrf_rank = orig_rrf_item["rrf_rank"] if orig_rrf_item else None
                    admitted = req_cid in selected_cids

                    counterfactual_trace.append({
                        "q_id": q_id,
                        "candidate_N": N,
                        "policy": pol_id,
                        "gold_chunk": req_cid,
                        "original_rrf_rank": orig_rrf_rank or "—",
                        "admitted_top10": "YES" if admitted else "NO",
                        "reason_excluded": "Retained in Top-10" if admitted else ("Absent from candidate pool N" if orig_rrf_rank is None else f"Original rank {orig_rrf_rank} > 10" if pol_id == "P0" else f"Exceeded parent limit or rank {orig_rrf_rank}")
                    })

    # Build 12-Configuration Master Aggregate Table
    master_table = []
    for N in candidate_depths:
        for pol in parent_policies:
            pol_id = pol["id"]
            recs = matrix_results[(N, pol_id)]

            tot_req_units = sum(r["total_required_evidence"] for r in recs)
            tot_ret_units = sum(r["retrieved_evidence_count"] for r in recs)
            agg_recall = tot_ret_units / tot_req_units

            tot_claims = sum(r["total_claims"] for r in recs)
            tot_supp_claims = sum(r["supported_claims"] for r in recs)
            agg_claim_avail = tot_supp_claims / tot_claims

            agg_ev_completeness = float(np.mean([r["ev_completeness"] for r in recs]))
            agg_ev_precision = float(np.mean([r["secondary"]["ev_precision"] for r in recs]))
            agg_pri_precision = float(np.mean([r["secondary"]["pri_precision"] for r in recs]))
            agg_mrr = float(np.mean([r["secondary"]["mrr"] for r in recs]))
            agg_ndcg = float(np.mean([r["secondary"]["ndcg10"] for r in recs]))
            agg_redundancy = float(np.mean([r["secondary"]["redundancy_ratio"] for r in recs]))
            agg_unique_parents = float(np.mean([r["secondary"]["unique_parent_count"] for r in recs]))
            agg_avg_chunks_per_parent = float(np.mean([r["secondary"]["avg_chunks_per_parent"] for r in recs]))

            master_table.append({
                "candidate_N": N,
                "policy_id": pol_id,
                "policy_name": pol["name"],
                "recall_at_10": agg_recall,
                "evidence_precision": agg_ev_precision,
                "primary_precision": agg_pri_precision,
                "claim_availability": agg_claim_avail,
                "evidence_completeness": agg_ev_completeness,
                "ndcg10": agg_ndcg,
                "mrr": agg_mrr,
                "redundancy_ratio": agg_redundancy,
                "unique_parents": agg_unique_parents,
                "avg_chunks_per_parent": agg_avg_chunks_per_parent
            })

    print("\n=========================================================================", flush=True)
    print("                12-CONFIGURATION MASTER AGGREGATE TABLE                   ", flush=True)
    print("=========================================================================", flush=True)
    print(f"{'N':>4s} | {'Policy':15s} | {'Recall@10':9s} | {'ClaimAvail':10s} | {'EvComp':8s} | {'nDCG':6s} | {'MRR':5s} | {'Redun':7s} | {'Parents':7s}", flush=True)
    print("-" * 88, flush=True)
    for row in master_table:
        print(f"{row['candidate_N']:4d} | {row['policy_name']:15s} | {row['recall_at_10']:9.1%} | {row['claim_availability']:10.1%} | {row['evidence_completeness']:8.1%} | {row['ndcg10']:6.3f} | {row['mrr']:5.3f} | {row['redundancy_ratio']:7.1%} | {row['unique_parents']:7.1f}", flush=True)

    # Save complete Phase 6 output payload
    out_payload = {
        "timestamp": datetime.now().isoformat(),
        "master_table": master_table,
        "counterfactual_trace": counterfactual_trace,
        "per_question_matrix": per_question_matrix
    }

    out_file = os.path.join(current_dir, "results", "phase6_experiment_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2, ensure_ascii=False)

    print(f"\nPhase 6 experiment complete. Saved results to:\n{out_file}\n", flush=True)


if __name__ == "__main__":
    main()
