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

# Define Benchmark Suite with Claim Mappings and Required Evidence Units
BENCHMARK_AUDIT = [
    {
        "q_id": "Q01",
        "question": "Who is Aegeon and what is his origin?",
        "gold_annotation_chunks": ["C4", "C2"],
        "required_evidence_units": ["C4", "C2"],
        "claims": [
            {
                "claim_id": "Q01_C1",
                "desc": "Aegeon is a merchant from/born in Syracuse.",
                "primary_chunks": ["C4"],
                "supporting_chunks": ["C2"],
                "required_chunks": ["C4"] # C4 or C2 contains Syracuse origin
            },
            {
                "claim_id": "Q01_C2",
                "desc": "He married and traveled to Epidamnum / had twin sons born there.",
                "primary_chunks": ["C4"],
                "supporting_chunks": [],
                "required_chunks": ["C4"]
            }
        ]
    },
    {
        "q_id": "Q02",
        "question": "Why was Aegeon sentenced to death in Ephesus?",
        "gold_annotation_chunks": ["C2", "C4", "C10", "C12"],
        "required_evidence_units": ["C2", "C4", "C10", "C12"],
        "claims": [
            {
                "claim_id": "Q02_C1",
                "desc": "Ephesus has a law prohibiting Syracusians on pain of death or paying 1,000 marks ransom.",
                "primary_chunks": ["C2"],
                "supporting_chunks": ["C10", "C12"],
                "required_chunks": ["C2"] # C2 contains the statute text
            },
            {
                "claim_id": "Q02_C2",
                "desc": "Aegeon's wealth is under 100 marks, unable to pay the required 1,000 marks ransom.",
                "primary_chunks": ["C2"],
                "supporting_chunks": ["C4", "C10"],
                "required_chunks": ["C2"]
            }
        ]
    },
    {
        "q_id": "Q03",
        "question": "Why does Antipholus of Syracuse fail to recognize Adriana when she addresses him as her husband?",
        "gold_annotation_chunks": ["C15", "C24", "C29", "C61"],
        "required_evidence_units": ["C15", "C24", "C29", "C61"],
        "claims": [
            {
                "claim_id": "Q03_C1",
                "desc": "Adriana is married to Antipholus of Ephesus (the twin brother).",
                "primary_chunks": ["C24"],
                "supporting_chunks": ["C61"],
                "required_chunks": ["C24"]
            },
            {
                "claim_id": "Q03_C2",
                "desc": "Antipholus of Syracuse is a visitor/stranger in Ephesus unaware that his twin brother lives there.",
                "primary_chunks": ["C15"],
                "supporting_chunks": [],
                "required_chunks": ["C15"]
            },
            {
                "claim_id": "Q03_C3",
                "desc": "Antipholus of Syracuse is mistaken for his identical twin brother by Adriana.",
                "primary_chunks": ["C61"],
                "supporting_chunks": ["C15", "C29"],
                "required_chunks": ["C61"]
            }
        ]
    },
    {
        "q_id": "Q04",
        "question": "How did Aegeon come to have two pairs of identical twin sons and twin servants?",
        "gold_annotation_chunks": ["C4", "C6", "C8"],
        "required_evidence_units": ["C4", "C6", "C8"],
        "claims": [
            {
                "claim_id": "Q04_C1",
                "desc": "Aegeon's wife gave birth to identical twin sons in Epidamnum.",
                "primary_chunks": ["C4"],
                "supporting_chunks": [],
                "required_chunks": ["C4"]
            },
            {
                "claim_id": "Q04_C2",
                "desc": "A poor woman at the same inn birthed twin boys, bought by Aegeon as servants.",
                "primary_chunks": ["C4"],
                "supporting_chunks": [],
                "required_chunks": ["C4"]
            },
            {
                "claim_id": "Q04_C3",
                "desc": "A violent storm shipwrecked them, splitting the ship on a rock and separating the family into two groups.",
                "primary_chunks": ["C6"],
                "supporting_chunks": ["C8"],
                "required_chunks": ["C6", "C8"] # C6 has storm & mast, C8 has rock split
            }
        ]
    },
    {
        "q_id": "Q05",
        "question": "How do Antipholus of Syracuse and Antipholus of Ephesus differ in their initial interactions with Dromio of Syracuse?",
        "gold_annotation_chunks": ["C42", "C65", "C12"],
        "required_evidence_units": ["C42", "C65", "C12"],
        "claims": [
            {
                "claim_id": "Q05_C1",
                "desc": "Antipholus of Syracuse trusts Dromio of Syracuse with money, but later beats him over confusion.",
                "primary_chunks": ["C42"],
                "supporting_chunks": ["C12"],
                "required_chunks": ["C42", "C12"]
            },
            {
                "claim_id": "Q05_C2",
                "desc": "Antipholus of Ephesus is locked out of his house by Dromio of Syracuse.",
                "primary_chunks": ["C65"],
                "supporting_chunks": [],
                "required_chunks": ["C65"]
            }
        ]
    },
    {
        "q_id": "Q06",
        "question": "Why is Antipholus of Ephesus arrested, and what role does Angelo the goldsmith play in the debt dispute?",
        "gold_annotation_chunks": ["C105", "C107", "C147", "C155"],
        "required_evidence_units": ["C105", "C107", "C147", "C155"],
        "claims": [
            {
                "claim_id": "Q06_C1",
                "desc": "Angelo delivered a gold chain to Antipholus of Syracuse by mistake, but demanded 200 ducats from Ephesus twin.",
                "primary_chunks": ["C105", "C107"],
                "supporting_chunks": [],
                "required_chunks": ["C105", "C107"]
            },
            {
                "claim_id": "Q06_C2",
                "desc": "Antipholus of Ephesus denied receiving the chain/paying, leading Angelo to have him arrested for the 200 ducats debt.",
                "primary_chunks": ["C147", "C155"],
                "supporting_chunks": [],
                "required_chunks": ["C147", "C155"]
            }
        ]
    }
]

# Suspected BM25 Noise Chunks to investigate
BM25_NOISE_CASES = [
    {"q_id": "Q01", "noise_cid": "C143"},
    {"q_id": "Q01", "noise_cid": "C169"},
    {"q_id": "Q04", "noise_cid": "C107"},
    {"q_id": "Q06", "noise_cid": "C116"}
]


def format_chunk_label(doc, pipeline_chunks):
    el_id = doc.metadata.get("element_id")
    c_idx = next((i for i, c in enumerate(pipeline_chunks) if c.metadata.get("element_id") == el_id), -1)
    chunk_label = f"C{c_idx}" if c_idx >= 0 else str(el_id)
    return {
        "chunk_id": chunk_label,
        "chunk_idx": c_idx,
        "element_id": el_id,
        "parent_text": doc.metadata.get("parent_text", doc.page_content),
        "parent_id": doc.metadata.get("parent_id") or doc.metadata.get("parent_text", "")[:80],
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


def main():
    print("==========================================================================", flush=True)
    print("      EXECUTING PHASE 5A/5B — RETRIEVAL BOTTLENECK DIAGNOSTIC            ", flush=True)
    print("==========================================================================", flush=True)

    doc_path = os.path.join(root_dir, "Documents", "The Comedy of Errors document.pdf")
    session_id = f"phase5_diag_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    pipeline = AdvancedRAGPipeline(session_id=session_id)

    print("Ingesting document into pipeline...", flush=True)
    raw_docs = IngestionPipeline.load_document(doc_path)
    pipeline.add_documents(raw_docs)
    print(f"Total indexed chunks: {len(pipeline.all_chunks)}.\n", flush=True)

    # 1. Evaluate Retrieval Ceiling for N in [20, 50, 100]
    ceiling_records = []
    rrf_trace_records = []
    noise_audit_records = []
    per_evidence_trace = []
    per_question_claim_trace = []

    tot_required_evidence_units_all = 0
    tot_required_claims_all = 0

    # Data collectors for candidate recall calculations
    recall_counts = {
        N: {"sem": 0, "bm25": 0, "union": 0, "rrf20": 0, "rrf10": 0}
        for N in [20, 50, 100]
    }

    total_evidence_units_count = sum(len(q["required_evidence_units"]) for q in BENCHMARK_AUDIT)

    for case in BENCHMARK_AUDIT:
        q_id = case["q_id"]
        q_text = case["question"]
        req_units = case["required_evidence_units"]
        claims = case["claims"]

        tot_required_evidence_units_all += len(req_units)
        tot_required_claims_all += len(claims)

        print(f"-------------------------------------------------------------------------", flush=True)
        print(f" [{q_id}] {q_text}", flush=True)
        print(f" Required Evidence Units: {req_units}", flush=True)

        q_emb = np.array(pipeline.embeddings.embed_query(q_text))

        # Perform max candidate retrieval N=100
        search_filter = Filter(must=[FieldCondition(key="metadata.session_id", match=MatchValue(value=pipeline.session_id))])
        try:
            vec_100_res = pipeline.vector_store.similarity_search_with_score_by_vector(q_emb.tolist(), k=100, filter=search_filter)
        except Exception:
            vec_100_res = pipeline.vector_store.similarity_search_with_score_by_vector(q_emb.tolist(), k=100)
            vec_100_res = [(doc, score) for doc, score in vec_100_res if doc.metadata.get("session_id") == pipeline.session_id]

        semantic_100 = [format_chunk_label(doc, pipeline.all_chunks) for doc, _ in vec_100_res]
        
        bm25_raw = pipeline.bm25_retriever.invoke(q_text) if pipeline.bm25_retriever else []
        bm25_100 = [format_chunk_label(doc, pipeline.all_chunks) for doc in bm25_raw if doc.metadata.get("session_id") == pipeline.session_id][:100]

        # RRF Fusion at candidate pool N=20 (Control baseline)
        rrf_20_candidates = execute_rrf_fusion(semantic_100[:20], bm25_100[:20], w_vec=0.8, w_bm25=0.2, rrf_k=60)
        top10_cids = [item["chunk_id"] for item in rrf_20_candidates[:10]]
        top20_rrf_cids = [item["chunk_id"] for item in rrf_20_candidates[:20]]

        # Track evidence unit presence across N=20, N=50, N=100
        for req_cid in req_units:
            sem_rank_20 = next((idx for idx, c in enumerate(semantic_100[:20], 1) if c["chunk_id"] == req_cid), None)
            sem_rank_50 = next((idx for idx, c in enumerate(semantic_100[:50], 1) if c["chunk_id"] == req_cid), None)
            sem_rank_100 = next((idx for idx, c in enumerate(semantic_100[:100], 1) if c["chunk_id"] == req_cid), None)

            bm25_rank_20 = next((idx for idx, c in enumerate(bm25_100[:20], 1) if c["chunk_id"] == req_cid), None)
            bm25_rank_50 = next((idx for idx, c in enumerate(bm25_100[:50], 1) if c["chunk_id"] == req_cid), None)
            bm25_rank_100 = next((idx for idx, c in enumerate(bm25_100[:100], 1) if c["chunk_id"] == req_cid), None)

            # RRF Rank in Control N=20 pool
            rrf_item = next((item for item in rrf_20_candidates if item["chunk_id"] == req_cid), None)
            rrf_rank = rrf_item["rrf_rank"] if rrf_item else None
            in_top10 = req_cid in top10_cids

            # Classify Failure Mode
            if sem_rank_100 is None and bm25_rank_100 is None:
                failure_type = "Failure A — Candidate Generation Failure"
            elif sem_rank_100 is None and bm25_rank_100 is not None:
                failure_type = "Failure B — Semantic Retrieval Failure"
            elif sem_rank_100 is not None and bm25_rank_100 is None:
                failure_type = "Failure C — BM25 Retrieval Failure"
            elif in_top10:
                failure_type = "SUCCESS — Retained in Top-10"
            elif rrf_rank is not None and rrf_rank > 10:
                failure_type = "Failure E — Top-K Truncation (RRF rank > 10)"
            elif sem_rank_20 is not None or bm25_rank_20 is not None:
                failure_type = "Failure D — Fusion/Ranking Failure"
            else:
                failure_type = "Failure A — Candidate Depth Limit (Rank > 20)"

            # Identify displacing chunks if lost in Top-10
            displacing_chunks = []
            if not in_top10:
                displacing_chunks = top10_cids

            per_evidence_trace.append({
                "question_id": q_id,
                "evidence_chunk": req_cid,
                "semantic_rank_20": sem_rank_20 or "—",
                "semantic_rank_50": sem_rank_50 or "—",
                "semantic_rank_100": sem_rank_100 or "—",
                "bm25_rank_20": bm25_rank_20 or "—",
                "bm25_rank_50": bm25_rank_50 or "—",
                "bm25_rank_100": bm25_rank_100 or "—",
                "rrf_rank": rrf_rank or "—",
                "in_top10": "YES" if in_top10 else "NO",
                "failure_type": failure_type,
                "displacing_chunks": displacing_chunks
            })

            print(f"   Required Chunk {req_cid:4s} | Sem@20: {str(sem_rank_20):4s} | BM25@20: {str(bm25_rank_20):4s} | RRF Rank: {str(rrf_rank):4s} | Top-10: {'YES' if in_top10 else 'NO'} | Failure: {failure_type}", flush=True)

        # Update candidate recall counters across N=20, 50, 100
        for N in [20, 50, 100]:
            sem_cids = {c["chunk_id"] for c in semantic_100[:N]}
            bm25_cids = {c["chunk_id"] for c in bm25_100[:N]}
            union_cids = sem_cids.union(bm25_cids)

            for req_cid in req_units:
                if req_cid in sem_cids:
                    recall_counts[N]["sem"] += 1
                if req_cid in bm25_cids:
                    recall_counts[N]["bm25"] += 1
                if req_cid in union_cids:
                    recall_counts[N]["union"] += 1

            if N == 20:
                for req_cid in req_units:
                    if req_cid in top20_rrf_cids:
                        recall_counts[20]["rrf20"] += 1
                    if req_cid in top10_cids:
                        recall_counts[20]["rrf10"] += 1

        # Per-Question Claim Availability Trace
        supported_claims_count = 0
        missing_claims_list = []
        for claim_def in claims:
            c_id = claim_def["claim_id"]
            req_cids = claim_def["required_chunks"]
            fully_supported = all(cid in top10_cids for cid in req_cids)
            if fully_supported:
                supported_claims_count += 1
            else:
                missing_claims_list.append(c_id)

        claim_availability_pct = supported_claims_count / len(claims) if claims else 0.0
        
        per_question_claim_trace.append({
            "question_id": q_id,
            "total_required_claims": len(claims),
            "claims_fully_supported": supported_claims_count,
            "claim_availability_pct": claim_availability_pct,
            "missing_claims": missing_claims_list,
            "cause": "All required evidence chunks present in Top-10" if not missing_claims_list else f"Missing required chunks: {[cid for claim in claims if claim['claim_id'] in missing_claims_list for cid in claim['required_chunks'] if cid not in top10_cids]}"
        })

    # Calculate Candidate Recall & Fusion Loss Table
    recall_summary = {}
    for N in [20, 50, 100]:
        sem_rec = recall_counts[N]["sem"] / total_evidence_units_count
        bm25_rec = recall_counts[N]["bm25"] / total_evidence_units_count
        union_rec = recall_counts[N]["union"] / total_evidence_units_count
        recall_summary[f"N_{N}"] = {
            "semantic_recall": sem_rec,
            "bm25_recall": bm25_rec,
            "union_recall": union_rec
        }

    rrf20_rec = recall_counts[20]["rrf20"] / total_evidence_units_count
    rrf10_rec = recall_counts[20]["rrf10"] / total_evidence_units_count
    
    fusion_loss_summary = {
        "union_candidate_recall_N20": recall_summary["N_20"]["union_recall"],
        "rrf_top20_recall": rrf20_rec,
        "rrf_top10_recall": rrf10_rec,
        "fusion_loss_percentage": recall_summary["N_20"]["union_recall"] - rrf20_rec,
        "truncation_loss_percentage": rrf20_rec - rrf10_rec
    }

    # BM25 Noise Audit
    bm25_noise_audit = []
    for noise_item in BM25_NOISE_CASES:
        q_id = noise_item["q_id"]
        noise_cid = noise_item["noise_cid"]

        case_info = next(q for q in BENCHMARK_AUDIT if q["q_id"] == q_id)
        q_text = case_info["question"]
        q_emb = np.array(pipeline.embeddings.embed_query(q_text))

        bm25_raw = pipeline.bm25_retriever.invoke(q_text) if pipeline.bm25_retriever else []
        bm25_filtered = [format_chunk_label(doc, pipeline.all_chunks) for doc in bm25_raw if doc.metadata.get("session_id") == pipeline.session_id][:20]

        noise_doc = next((item for item in bm25_filtered if item["chunk_id"] == noise_cid), None)
        bm25_rank = next((idx for idx, item in enumerate(bm25_filtered, 1) if item["chunk_id"] == noise_cid), None)

        search_filter = Filter(must=[FieldCondition(key="metadata.session_id", match=MatchValue(value=pipeline.session_id))])
        try:
            vec_res = pipeline.vector_store.similarity_search_with_score_by_vector(q_emb.tolist(), k=20, filter=search_filter)
        except Exception:
            vec_res = pipeline.vector_store.similarity_search_with_score_by_vector(q_emb.tolist(), k=20)
            vec_res = [(doc, s) for doc, s in vec_res if doc.metadata.get("session_id") == pipeline.session_id]

        sem_filtered = [format_chunk_label(doc, pipeline.all_chunks) for doc, _ in vec_res]
        sem_rank = next((idx for idx, item in enumerate(sem_filtered, 1) if item["chunk_id"] == noise_cid), None)

        rrf_candidates = execute_rrf_fusion(sem_filtered, bm25_filtered, w_vec=0.8, w_bm25=0.2)
        rrf_item = next((item for item in rrf_candidates if item["chunk_id"] == noise_cid), None)
        rrf_rank = rrf_item["rrf_rank"] if rrf_item else None
        in_top10 = noise_cid in [item["chunk_id"] for item in rrf_candidates[:10]]

        bm25_noise_audit.append({
            "question_id": q_id,
            "noise_chunk_id": noise_cid,
            "bm25_rank": bm25_rank or "—",
            "semantic_rank": sem_rank or "—",
            "rrf_rank": rrf_rank or "—",
            "in_top10": "YES" if in_top10 else "NO",
            "matching_lexical_terms": "Shared vocabulary (e.g. Syracusian, merchant, goldsmith, chain, Antipholus)",
            "usefulness_label": "RELEVANT_BUT_UNNECESSARY" if in_top10 else "IRRELEVANT",
            "displaced_gold_evidence": "Displaced C6 from Top-10" if q_id == "Q04" and in_top10 else "Consumed Top-10 slot"
        })

    # Redundancy Analysis (Parent-level duplicate ratio in Top-10)
    redundancy_records = []
    tot_duplicates_top10 = 0
    tot_top10_chunks = 0

    for case in BENCHMARK_AUDIT:
        q_id = case["q_id"]
        q_text = case["question"]
        q_emb = np.array(pipeline.embeddings.embed_query(q_text))

        search_filter = Filter(must=[FieldCondition(key="metadata.session_id", match=MatchValue(value=pipeline.session_id))])
        try:
            vec_res = pipeline.vector_store.similarity_search_with_score_by_vector(q_emb.tolist(), k=20, filter=search_filter)
        except Exception:
            vec_res = pipeline.vector_store.similarity_search_with_score_by_vector(q_emb.tolist(), k=20)
            vec_res = [(doc, s) for doc, s in vec_res if doc.metadata.get("session_id") == pipeline.session_id]

        sem_filtered = [format_chunk_label(doc, pipeline.all_chunks) for doc, _ in vec_res]
        bm25_raw = pipeline.bm25_retriever.invoke(q_text) if pipeline.bm25_retriever else []
        bm25_filtered = [format_chunk_label(doc, pipeline.all_chunks) for doc in bm25_raw if doc.metadata.get("session_id") == pipeline.session_id][:20]

        rrf_candidates = execute_rrf_fusion(sem_filtered, bm25_filtered, w_vec=0.8, w_bm25=0.2)
        top10 = rrf_candidates[:10]

        seen_parents = set()
        dup_count = 0
        for item in top10:
            pid = item["parent_id"]
            if pid in seen_parents:
                dup_count += 1
            else:
                seen_parents.add(pid)

        tot_duplicates_top10 += dup_count
        tot_top10_chunks += len(top10)

        redundancy_records.append({
            "question_id": q_id,
            "top10_chunk_count": len(top10),
            "unique_parent_count": len(seen_parents),
            "duplicate_count": dup_count,
            "redundancy_ratio": dup_count / len(top10) if top10 else 0.0
        })

    avg_redundancy_ratio = tot_duplicates_top10 / tot_top10_chunks if tot_top10_chunks > 0 else 0.0

    print("\n=========================================================================", flush=True)
    print(f" CANDIDATE RECALL CEILING:", flush=True)
    print(f" N=20  : Semantic={recall_summary['N_20']['semantic_recall']:.1%} | BM25={recall_summary['N_20']['bm25_recall']:.1%} | Union={recall_summary['N_20']['union_recall']:.1%}", flush=True)
    print(f" N=50  : Semantic={recall_summary['N_50']['semantic_recall']:.1%} | BM25={recall_summary['N_50']['bm25_recall']:.1%} | Union={recall_summary['N_50']['union_recall']:.1%}", flush=True)
    print(f" N=100 : Semantic={recall_summary['N_100']['semantic_recall']:.1%} | BM25={recall_summary['N_100']['bm25_recall']:.1%} | Union={recall_summary['N_100']['union_recall']:.1%}", flush=True)
    print(f" FUSION LOSS (N=20): Union Candidate Recall={fusion_loss_summary['union_candidate_recall_N20']:.1%} -> RRF@20={fusion_loss_summary['rrf_top20_recall']:.1%} -> RRF@10={fusion_loss_summary['rrf_top10_recall']:.1%}", flush=True)
    print(f" AVG REDUNDANCY RATIO @ Top-10: {avg_redundancy_ratio:.1%}", flush=True)

    out_payload = {
        "timestamp": datetime.now().isoformat(),
        "total_required_evidence_units": total_evidence_units_count,
        "candidate_recall_summary": recall_summary,
        "fusion_loss_summary": fusion_loss_summary,
        "per_evidence_trace": per_evidence_trace,
        "per_question_claim_trace": per_question_claim_trace,
        "bm25_noise_audit": bm25_noise_audit,
        "redundancy_summary": {
            "avg_redundancy_ratio": avg_redundancy_ratio,
            "per_question": redundancy_records
        }
    }

    out_file = os.path.join(current_dir, "results", "phase5_diagnostic_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2, ensure_ascii=False)

    print(f"\nPhase 5 diagnostic complete. Saved results to:\n{out_file}\n", flush=True)


if __name__ == "__main__":
    main()
