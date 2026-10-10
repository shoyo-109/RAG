import os
import sys
import json
import math
import time
import importlib.util
from datetime import datetime
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import CrossEncoder

# Load environment
load_dotenv()
api_key = os.getenv("NVIDIA_API_KEY") or os.getenv("NEMOTRON_API_KEY")
if not api_key:
    raise ValueError("NVIDIA_API_KEY or NEMOTRON_API_KEY not found in environment variables.")

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key
)

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
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

UNREACHABLE_DIAGNOSTIC_CHUNKS = ["C15", "C29", "C61", "C42", "C147"]

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

def get_q2_validated_multiquery(q_id):
    """Returns exact validated Q2 queries from Phase 7A."""
    q2_validated = {
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
    return q2_validated[q_id]

def combine_multiquery_semantic(vec_results_list, rrf_k=60):
    scores = {}
    doc_map = {}
    query_rank_map = {}

    for q_idx, res_list in enumerate(vec_results_list):
        for rank_idx, item in enumerate(res_list, 1):
            cid = item["chunk_id"]
            if cid not in doc_map:
                doc_map[cid] = item
                query_rank_map[cid] = {}
            query_rank_map[cid][q_idx] = rank_idx
            scores[cid] = scores.get(cid, 0.0) + (1.0 / (rrf_k + rank_idx))

    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    combined_ranked = []
    for rank_idx, (cid, score) in enumerate(sorted_docs, 1):
        doc = doc_map[cid]
        combined_ranked.append({
            "chunk_id": cid,
            "chunk_idx": doc["chunk_idx"],
            "element_id": doc["element_id"],
            "parent_id": doc["parent_id"],
            "parent_text": doc["parent_text"],
            "breadcrumb": doc["breadcrumb"],
            "text": doc["text"],
            "multi_semantic_rank": rank_idx,
            "subquery_ranks": query_rank_map[cid],
            "multi_semantic_score": score
        })
    return combined_ranked

def calculate_secondary_metrics(selected_chunks, gold_map, top_k=10):
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

    ev_precision = useful_count / top_k if top_k > 0 else 0.0
    pri_precision = primary_count / top_k if top_k > 0 else 0.0
    mrr = 1.0 / first_useful_rank if first_useful_rank > 0 else 0.0

    dcg = sum((2**rel_map.get(c, 0) - 1) / math.log2(i + 2) for i, c in enumerate(classifications[:top_k]))
    ideal_rels = sorted([3 if r == "PRIMARY" else 2 for r in gold_map.values()], reverse=True)[:top_k]
    idcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(ideal_rels))
    ndcg10 = dcg / idcg if idcg > 0 else 0.0

    return {
        "ev_precision": ev_precision,
        "pri_precision": pri_precision,
        "mrr": mrr,
        "ndcg10": ndcg10,
        "redundancy": dup_count
    }

# -----------------------------------------------------------------------------
# Expansion Functions
# -----------------------------------------------------------------------------

def expand_parent_candidates(base_candidates, all_chunk_items):
    """
    Identifies parent IDs represented in base_candidates, then retrieves all sibling chunks
    sharing those parent IDs from all_chunk_items.
    """
    base_cids = {c["chunk_id"] for c in base_candidates}
    parent_ids_in_base = {c["parent_id"] for c in base_candidates if c.get("parent_id")}

    expanded_map = {}
    provenance_list = []

    # First add base candidates
    for c in base_candidates:
        cid = c["chunk_id"]
        expanded_map[cid] = dict(c)
        provenance_list.append({
            "chunk_id": cid,
            "source": "base",
            "source_chunk_id": cid,
            "parent_id": c.get("parent_id"),
            "distance_from_seed": 0
        })

    # Find sibling chunks for parents
    new_added_count = 0
    for chunk_item in all_chunk_items:
        cid = chunk_item["chunk_id"]
        pid = chunk_item["parent_id"]
        if pid in parent_ids_in_base and cid not in expanded_map:
            # Find seed chunk in base that shared this parent
            seed_doc = next((b for b in base_candidates if b["parent_id"] == pid), base_candidates[0])
            doc_copy = dict(chunk_item)
            expanded_map[cid] = doc_copy
            provenance_list.append({
                "chunk_id": cid,
                "source": "parent",
                "source_chunk_id": seed_doc["chunk_id"],
                "parent_id": pid,
                "distance_from_seed": 0
            })
            new_added_count += 1

    return list(expanded_map.values()), provenance_list

def expand_neighbor_candidates(base_candidates, all_chunk_items, window=1):
    """
    For every candidate in base_candidates (at chunk_idx), adds neighbor chunks within ±window.
    """
    base_cids = {c["chunk_id"] for c in base_candidates}
    total_chunks = len(all_chunk_items)
    
    # Map c_idx to item
    index_to_item = {c["chunk_idx"]: c for c in all_chunk_items if c["chunk_idx"] >= 0}

    expanded_map = {}
    provenance_list = []

    # First add base candidates
    for c in base_candidates:
        cid = c["chunk_id"]
        expanded_map[cid] = dict(c)
        provenance_list.append({
            "chunk_id": cid,
            "source": "base",
            "source_chunk_id": cid,
            "parent_id": c.get("parent_id"),
            "distance_from_seed": 0
        })

    for c in base_candidates:
        seed_idx = c["chunk_idx"]
        seed_cid = c["chunk_id"]
        if seed_idx < 0:
            continue

        for offset in range(-window, window + 1):
            if offset == 0:
                continue
            target_idx = seed_idx + offset
            if 0 <= target_idx < total_chunks and target_idx in index_to_item:
                target_item = index_to_item[target_idx]
                target_cid = target_item["chunk_id"]
                if target_cid not in expanded_map:
                    expanded_map[target_cid] = dict(target_item)
                    provenance_list.append({
                        "chunk_id": target_cid,
                        "source": f"neighbor_±{window}",
                        "source_chunk_id": seed_cid,
                        "parent_id": target_item.get("parent_id"),
                        "distance_from_seed": offset
                    })

    return list(expanded_map.values()), provenance_list

def expand_parent_and_neighbor_candidates(base_candidates, all_chunk_items, window=1):
    """Combines Parent Expansion and Neighbor Expansion."""
    p_expanded, p_prov = expand_parent_candidates(base_candidates, all_chunk_items)
    n_expanded, n_prov = expand_neighbor_candidates(base_candidates, all_chunk_items, window=window)

    combined_map = {}
    prov_map = {}

    for c, p in zip(p_expanded, p_prov):
        cid = c["chunk_id"]
        combined_map[cid] = c
        prov_map[cid] = p

    for c, p in zip(n_expanded, n_prov):
        cid = c["chunk_id"]
        if cid not in combined_map:
            combined_map[cid] = c
            prov_map[cid] = p
        elif prov_map[cid]["source"] == "base":
            pass
        else:
            prov_map[cid]["source"] += f"+neighbor_±{window}"

    return list(combined_map.values()), list(prov_map.values())

def main():
    print("==========================================================================", flush=True)
    print("   EXECUTING PHASE 7C — TARGETED CANDIDATE GENERATION EXPANSION         ", flush=True)
    print("==========================================================================", flush=True)

    # Initialize Cross-Encoder model
    ce_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    print(f"Loading Cross-Encoder model: '{ce_model_name}'...", flush=True)
    t_ce_load_start = time.time()
    ce_model = CrossEncoder(ce_model_name)
    ce_load_time_s = time.time() - t_ce_load_start
    ce_device = str(ce_model.model.device)
    print(f"Cross-Encoder loaded successfully on device: {ce_device} ({ce_load_time_s:.2f}s)\n", flush=True)

    doc_path = os.path.join(root_dir, "Documents", "The Comedy of Errors document.pdf")
    session_id = f"phase7c_exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    pipeline = AdvancedRAGPipeline(session_id=session_id)

    print("Ingesting document into pipeline...", flush=True)
    raw_docs = IngestionPipeline.load_document(doc_path)
    pipeline.add_documents(raw_docs)
    print(f"Total indexed chunks: {len(pipeline.all_chunks)}.\n", flush=True)

    # Format all indexed chunks
    all_chunk_items = [format_chunk_item(doc, pipeline.all_chunks) for doc in pipeline.all_chunks]

    # Pre-retrieve base vector and BM25 candidates up to N=100
    search_filter = Filter(must=[FieldCondition(key="metadata.session_id", match=MatchValue(value=pipeline.session_id))])
    
    retrieved_cache = {}
    q2_gen_latencies = [1200.0] * len(BENCHMARK_SUITE)
    candidate_retrieval_latencies = []

    for case in BENCHMARK_SUITE:
        q_id = case["q_id"]
        q_orig = case["question"]
        q2_sub = get_q2_validated_multiquery(q_id)

        t_ret_start = time.time()

        # Vector Search for Q0 (k=100)
        q0_emb = np.array(pipeline.embeddings.embed_query(q_orig))
        try:
            res_q0 = pipeline.vector_store.similarity_search_with_score_by_vector(q0_emb.tolist(), k=100, filter=search_filter)
        except Exception:
            res_q0 = pipeline.vector_store.similarity_search_with_score_by_vector(q0_emb.tolist(), k=100)
            res_q0 = [(doc, s) for doc, s in res_q0 if doc.metadata.get("session_id") == pipeline.session_id]
        sem_q0_100 = [format_chunk_item(doc, pipeline.all_chunks) for doc, _ in res_q0]

        # Vector Search for Q2 sub-queries (k=100 each)
        sem_q2_sub_100 = []
        for sq in q2_sub:
            sq_emb = np.array(pipeline.embeddings.embed_query(sq))
            try:
                res_sq = pipeline.vector_store.similarity_search_with_score_by_vector(sq_emb.tolist(), k=100, filter=search_filter)
            except Exception:
                res_sq = pipeline.vector_store.similarity_search_with_score_by_vector(sq_emb.tolist(), k=100)
                res_sq = [(doc, s) for doc, s in res_sq if doc.metadata.get("session_id") == pipeline.session_id]
            sem_q2_sub_100.append([format_chunk_item(doc, pipeline.all_chunks) for doc, _ in res_sq])

        # BM25 Search on original query (k=100)
        bm25_raw = pipeline.bm25_retriever.invoke(q_orig) if pipeline.bm25_retriever else []
        bm25_100 = [format_chunk_item(doc, pipeline.all_chunks) for doc in bm25_raw if doc.metadata.get("session_id") == pipeline.session_id][:100]

        ret_lat_ms = (time.time() - t_ret_start) * 1000
        candidate_retrieval_latencies.append(ret_lat_ms)

        retrieved_cache[q_id] = {
            "sem_q0_100": sem_q0_100,
            "sem_q2_sub_100": sem_q2_sub_100,
            "bm25_100": bm25_100
        }

    # Evaluate Candidate Expansion Conditions across N=20 (Primary) and N=50 (Secondary)
    depths_to_test = [20, 50]
    master_experiments = {}

    for N in depths_to_test:
        print(f"\n=========================================================================", flush=True)
        print(f"                EVALUATING CANDIDATE POOL DEPTH N = {N}                 ", flush=True)
        print(f"=========================================================================", flush=True)

        conditions = [
            {"id": "Q2_CE_Control", "name": "Q2 + CE Control", "type": "control"},
            {"id": "Q2_Parent_CE", "name": "Q2 + Parent Exp + CE", "type": "parent"},
            {"id": "Q2_Neighbor_1_CE", "name": "Q2 + Neighbor ±1 + CE", "type": "neighbor_1"},
            {"id": "Q2_Neighbor_2_CE", "name": "Q2 + Neighbor ±2 + CE", "type": "neighbor_2"},
            {"id": "Q2_Combined_CE", "name": "Q2 + Parent + Neighbor ±1 + CE", "type": "combined"}
        ]

        cond_results = {c["id"]: [] for c in conditions}
        unreachable_tracking = {c["id"]: [] for c in conditions}
        per_question_analysis = {c["id"]: [] for c in conditions}
        provenance_store = {c["id"]: [] for c in conditions}

        for case_idx, case in enumerate(BENCHMARK_SUITE):
            q_id = case["q_id"]
            q_orig = case["question"]
            req_units = case["required_evidence"]
            gold_map = case["gold_map"]
            claims = case["claims"]

            cache = retrieved_cache[q_id]
            q2_gen_lat = q2_gen_latencies[case_idx]
            cand_ret_lat = candidate_retrieval_latencies[case_idx]

            # Construct Base Q2 Candidate Union at depth N
            sub_N_lists = [sub[:N] for sub in cache["sem_q2_sub_100"]]
            bm25_cand_N = cache["bm25_100"][:N]

            union_doc_map = {}
            for sub_list in sub_N_lists:
                for item in sub_list:
                    union_doc_map[item["chunk_id"]] = item
            for item in bm25_cand_N:
                union_doc_map[item["chunk_id"]] = item

            base_candidates = list(union_doc_map.values())
            base_cids = [item["chunk_id"] for item in base_candidates]
            base_cand_retrieved = [cid for cid in req_units if cid in base_cids]
            base_cand_recall = len(base_cand_retrieved) / len(req_units)

            # Evaluate each expansion condition
            for cond in conditions:
                c_id = cond["id"]
                c_type = cond["type"]

                t_exp_start = time.time()
                if c_type == "control":
                    expanded_candidates = list(base_candidates)
                    prov_list = [{
                        "chunk_id": item["chunk_id"],
                        "source": "base",
                        "source_chunk_id": item["chunk_id"],
                        "parent_id": item.get("parent_id"),
                        "distance_from_seed": 0
                    } for item in base_candidates]
                elif c_type == "parent":
                    expanded_candidates, prov_list = expand_parent_candidates(base_candidates, all_chunk_items)
                elif c_type == "neighbor_1":
                    expanded_candidates, prov_list = expand_neighbor_candidates(base_candidates, all_chunk_items, window=1)
                elif c_type == "neighbor_2":
                    expanded_candidates, prov_list = expand_neighbor_candidates(base_candidates, all_chunk_items, window=2)
                elif c_type == "combined":
                    expanded_candidates, prov_list = expand_parent_and_neighbor_candidates(base_candidates, all_chunk_items, window=1)

                exp_time_ms = (time.time() - t_exp_start) * 1000

                exp_cids = [item["chunk_id"] for item in expanded_candidates]
                exp_cand_retrieved = [cid for cid in req_units if cid in exp_cids]
                exp_cand_recall = len(exp_cand_retrieved) / len(req_units)

                new_cids = [cid for cid in exp_cids if cid not in base_cids]
                new_gold_recovered = [cid for cid in req_units if cid in exp_cids and cid not in base_cids]

                # Cross-Encoder Reranking over expanded candidates
                ce_pairs = [(q_orig, doc["text"]) for doc in expanded_candidates]
                
                t_ce_start = time.time()
                ce_scores = ce_model.predict(ce_pairs, batch_size=32, show_progress_bar=False)
                ce_lat_ms = (time.time() - t_ce_start) * 1000

                ce_ranked_docs = []
                for doc, score in zip(expanded_candidates, ce_scores):
                    doc_copy = dict(doc)
                    doc_copy["ce_score"] = float(score)
                    ce_ranked_docs.append(doc_copy)

                ce_ranked_docs = sorted(ce_ranked_docs, key=lambda x: x["ce_score"], reverse=True)
                for ce_rank, doc in enumerate(ce_ranked_docs, 1):
                    doc["ce_rank"] = ce_rank

                top10_docs = ce_ranked_docs[:10]
                top10_cids = [item["chunk_id"] for item in top10_docs]

                top10_retrieved = [cid for cid in req_units if cid in top10_cids]
                final_recall_10 = len(top10_retrieved) / len(req_units)

                sec = calculate_secondary_metrics(top10_docs, gold_map, top_k=10)
                supported_claims = sum(1 for claim in claims if all(cid in top10_cids for cid in claim["required_chunks"]))
                claim_avail = supported_claims / len(claims)

                tot_lat_ms = q2_gen_lat + cand_ret_lat + exp_time_ms + ce_lat_ms

                cond_results[c_id].append({
                    "q_id": q_id,
                    "base_cand_count": len(base_candidates),
                    "expanded_cand_count": len(expanded_candidates),
                    "new_cands_count": len(new_cids),
                    "base_cand_recall": base_cand_recall,
                    "exp_cand_recall": exp_cand_recall,
                    "new_gold_recovered_count": len(new_gold_recovered),
                    "new_gold_recovered": new_gold_recovered,
                    "final_recall_10": final_recall_10,
                    "claim_avail": claim_avail,
                    "secondary": sec,
                    "ce_pairs_count": len(ce_pairs),
                    "ce_latency_ms": ce_lat_ms,
                    "tot_latency_ms": tot_lat_ms
                })

                # Track Unreachable Diagnostic Chunks (C15, C29, C61, C42, C147)
                for unreach_cid in UNREACHABLE_DIAGNOSTIC_CHUNKS:
                    if unreach_cid in req_units:
                        in_base = unreach_cid in base_cids
                        in_exp = unreach_cid in exp_cids
                        ce_doc = next((d for d in ce_ranked_docs if d["chunk_id"] == unreach_cid), None)
                        ce_r = ce_doc["ce_rank"] if ce_doc else "—"
                        ce_s = round(ce_doc["ce_score"], 4) if ce_doc else "—"
                        in_t10 = unreach_cid in top10_cids

                        # Identify provenance source for this chunk if added
                        prov_info = next((p for p in prov_list if p["chunk_id"] == unreach_cid), None)

                        unreachable_tracking[c_id].append({
                            "q_id": q_id,
                            "N": N,
                            "unreachable_chunk": unreach_cid,
                            "in_base_candidate_pool": in_base,
                            "in_expanded_candidate_pool": in_exp,
                            "recovered_by_expansion": not in_base and in_exp,
                            "expansion_provenance": prov_info,
                            "ce_score": ce_s,
                            "ce_rank": ce_r,
                            "in_top10": "YES" if in_t10 else "NO"
                        })

                # Per-Question Record
                per_question_analysis[c_id].append({
                    "q_id": q_id,
                    "N": N,
                    "base_recall": base_cand_recall,
                    "exp_recall": exp_cand_recall,
                    "ce_top10_recall": final_recall_10,
                    "evidence_recovered": new_gold_recovered,
                    "expansion_mechanism": cond["name"],
                    "top10_cids": top10_cids
                })

                provenance_store[c_id].extend(prov_list)

        # Build Comparison Table for Depth N
        tot_req_evidence = sum(len(c["required_evidence"]) for c in BENCHMARK_SUITE)
        comparison_table = []

        for cond in conditions:
            c_id = cond["id"]
            res_list = cond_results[c_id]

            agg_base_cand_recall = sum(r["base_cand_recall"] * len(BENCHMARK_SUITE[idx]["required_evidence"]) for idx, r in enumerate(res_list)) / tot_req_evidence
            agg_exp_cand_recall = sum(r["exp_cand_recall"] * len(BENCHMARK_SUITE[idx]["required_evidence"]) for idx, r in enumerate(res_list)) / tot_req_evidence
            
            agg_base_count = sum(r["base_cand_count"] for r in res_list)
            agg_exp_count = sum(r["expanded_cand_count"] for r in res_list)
            cand_exp_factor = agg_exp_count / agg_base_count if agg_base_count > 0 else 1.0

            agg_final_recall = sum(r["final_recall_10"] * len(BENCHMARK_SUITE[idx]["required_evidence"]) for idx, r in enumerate(res_list)) / tot_req_evidence
            
            ev_prec = float(np.mean([r["secondary"]["ev_precision"] for r in res_list]))
            pri_prec = float(np.mean([r["secondary"]["pri_precision"] for r in res_list]))
            mrr = float(np.mean([r["secondary"]["mrr"] for r in res_list]))
            ndcg10 = float(np.mean([r["secondary"]["ndcg10"] for r in res_list]))

            completeness = sum(1 for r in res_list if r["final_recall_10"] == 1.0) / len(BENCHMARK_SUITE)
            
            tot_new_gold = sum(r["new_gold_recovered_count"] for r in res_list)
            tot_new_cands = sum(r["new_cands_count"] for r in res_list)

            exp_efficiency = tot_new_gold / tot_new_cands if tot_new_cands > 0 else 0.0
            candidate_conversion = agg_final_recall / agg_exp_cand_recall if agg_exp_cand_recall > 0 else 0.0

            comparison_table.append({
                "condition_id": c_id,
                "condition_name": cond["name"],
                "base_candidate_recall": agg_base_cand_recall,
                "expanded_candidate_recall": agg_exp_cand_recall,
                "candidate_expansion_factor": round(cand_exp_factor, 2),
                "final_recall_at_10": agg_final_recall,
                "evidence_precision": ev_prec,
                "primary_precision": pri_prec,
                "mrr": mrr,
                "ndcg10": ndcg10,
                "evidence_completeness": completeness,
                "required_evidence_completeness": agg_final_recall,
                "new_gold_evidence_recovered_count": tot_new_gold,
                "total_new_candidates_added": tot_new_cands,
                "expansion_recovery_efficiency": exp_efficiency,
                "candidate_to_top10_conversion": candidate_conversion,
                "mean_ce_pairs": float(np.mean([r["ce_pairs_count"] for r in res_list])),
                "mean_ce_latency_ms": float(np.mean([r["ce_latency_ms"] for r in res_list])),
                "mean_total_latency_ms": float(np.mean([r["tot_latency_ms"] for r in res_list]))
            })

        master_experiments[f"N_{N}"] = {
            "comparison_table": comparison_table,
            "unreachable_tracking": unreachable_tracking,
            "per_question_analysis": per_question_analysis
        }

        # Print Comparison Table for Depth N
        print(f"\nPhase 7C Master Comparison Table (N={N}):", flush=True)
        print("-" * 150, flush=True)
        print(f"{'Condition':30s} | {'BaseCandRec':11s} | {'ExpCandRec':10s} | {'ExpFactor':9s} | {'Rec@10':7s} | {'EvPrec':7s} | {'MRR':5s} | {'nDCG@10':7s} | {'Completeness':12s} | {'NewGold':7s}", flush=True)
        print("-" * 150, flush=True)
        for row in comparison_table:
            print(f"{row['condition_name']:30s} | {row['base_candidate_recall']:11.1%} | {row['expanded_candidate_recall']:10.1%} | {row['candidate_expansion_factor']:8.2f}x | {row['final_recall_at_10']:7.1%} | {row['evidence_precision']:7.3f} | {row['mrr']:5.3f} | {row['ndcg10']:7.3f} | {row['evidence_completeness']:12.1%} | {row['new_gold_evidence_recovered_count']:7d}", flush=True)

    # Output full experiment JSON payload
    out_payload = {
        "timestamp": datetime.now().isoformat(),
        "model_metadata": {
            "cross_encoder_model": ce_model_name,
            "device": ce_device,
            "vector_embedding_model": "all-MiniLM-L6-v2",
            "unreachable_diagnostic_chunks": UNREACHABLE_DIAGNOSTIC_CHUNKS,
            "batch_size": 32
        },
        "experiments": master_experiments
    }

    out_file = os.path.join(current_dir, "results", "phase7c_diagnostic_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2, ensure_ascii=False)

    print(f"\n=========================================================================", flush=True)
    print(f"Phase 7C diagnostic execution complete. Saved results to:\n{out_file}", flush=True)
    print(f"=========================================================================\n", flush=True)

if __name__ == "__main__":
    main()
