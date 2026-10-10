import os
import sys
import json
import math
import importlib.util
from datetime import datetime
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

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

def generate_q1_reformulation(original_question):
    """Q1: Single retrieval-oriented query reformulation."""
    prompt = f"""You are an expert search and retrieval assistant. Given a user question about a play or historical text, generate a single retrieval-oriented query reformulation.

Rules:
- Preserve the user's core information need.
- Expose key entities, actions, and events using clear document-retrieval terminology.
- Do NOT invent facts or add information not present in the question.
- Do NOT answer the question.
- Output ONLY the single reformulated query string on one line, without commentary, quote marks, or prefixes.

Original Question: {original_question}
Reformulated Retrieval Query:"""

    response = client.chat.completions.create(
        model="nvidia/nemotron-3-ultra-550b-a55b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        top_p=0.95,
        max_tokens=256
    )
    content = response.choices[0].message.content.strip()
    # Clean up quotes if present
    if content.startswith('"') and content.endswith('"'):
        content = content[1:-1]
    return content

def generate_q2_multiquery(original_question):
    """Q2: Multi-query expansion generating exactly 3 sub-queries."""
    prompt = f"""You are an expert search and retrieval assistant. Given a user question about a play or historical text, generate exactly 3 distinct search sub-queries to retrieve relevant evidence from the document.

Rules:
- Each query should target a different aspect of the evidence needed (e.g. Query A: entities/character background; Query B: key rules, laws, or relationships; Query C: specific actions, events, or consequences).
- Do NOT invent facts or assume details not in the question.
- Do NOT answer the question.
- Format your output strictly as a JSON array of 3 strings: ["query_1", "query_2", "query_3"] without any markdown syntax or conversational text.

Original Question: {original_question}
Sub-queries JSON array:"""

    response = client.chat.completions.create(
        model="nvidia/nemotron-3-ultra-550b-a55b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        top_p=0.95,
        max_tokens=512
    )
    content = response.choices[0].message.content.strip()
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()
    
    try:
        sub_queries = json.loads(content)
        if isinstance(sub_queries, list) and len(sub_queries) == 3:
            return sub_queries
    except Exception:
        pass
    
    # Fallback splitting by line
    lines = [l.strip() for l in content.split("\n") if l.strip()]
    cleaned = []
    for l in lines:
        if l.startswith(("- ", "1. ", "2. ", "3. ", '"')):
            l = l.lstrip("- 123. \"").rstrip('"')
        cleaned.append(l)
    if len(cleaned) >= 3:
        return cleaned[:3]
    return [original_question, original_question, original_question]

def execute_rrf_fusion(vector_results, bm25_results, w_vec=0.8, w_bm25=0.2, rrf_k=60):
    """Executes standard RRF fusion given vector and BM25 candidate lists."""
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

def combine_multiquery_semantic(vec_results_list, rrf_k=60):
    """
    Combines semantic candidate lists from multiple sub-queries (Q_A, Q_B, Q_C).
    Ranks candidates by reciprocal rank sum across sub-queries.
    """
    scores = {}
    doc_map = {}
    query_rank_map = {} # cid -> {q_idx: rank}

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
    """Calculates Evidence Precision, Primary Precision, MRR, nDCG."""
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

    return {
        "ev_precision": ev_precision,
        "pri_precision": pri_precision,
        "mrr": mrr,
        "ndcg10": ndcg10
    }

def main():
    print("==========================================================================", flush=True)
    print("       EXECUTING PHASE 7A — QUERY EXPANSION & MULTI-QUERY DIAGNOSTIC      ", flush=True)
    print("==========================================================================", flush=True)

    doc_path = os.path.join(root_dir, "Documents", "The Comedy of Errors document.pdf")
    session_id = f"phase7a_exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    pipeline = AdvancedRAGPipeline(session_id=session_id)

    print("Ingesting document into pipeline...", flush=True)
    raw_docs = IngestionPipeline.load_document(doc_path)
    pipeline.add_documents(raw_docs)
    print(f"Total indexed chunks: {len(pipeline.all_chunks)}.\n", flush=True)

    # Generate Expansion Queries for all benchmark questions
    print("Generating Q1 (Reformulated) and Q2 (Multi-Query) for benchmark suite...", flush=True)
    query_definitions = []
    for case in BENCHMARK_SUITE:
        q_id = case["q_id"]
        q_orig = case["question"]
        print(f"\nProcessing {q_id}: '{q_orig}'", flush=True)
        
        q1_ref = generate_q1_reformulation(q_orig)
        print(f"  Q1 Reformulated: '{q1_ref}'", flush=True)
        
        q2_sub = generate_q2_multiquery(q_orig)
        print(f"  Q2 Multi-queries:", flush=True)
        for idx, sq in enumerate(q2_sub):
            print(f"    Sub-query {chr(65+idx)}: '{sq}'", flush=True)
            
        query_definitions.append({
            "q_id": q_id,
            "original_query": q_orig,
            "q1_reformulated": q1_ref,
            "q2_multiqueries": q2_sub
        })

    # Retrieve vector candidates up to N=100 for all queries
    search_filter = Filter(must=[FieldCondition(key="metadata.session_id", match=MatchValue(value=pipeline.session_id))])
    
    retrieved_cache = {}
    for case in BENCHMARK_SUITE:
        q_id = case["q_id"]
        q_def = next(d for d in query_definitions if d["q_id"] == q_id)
        
        # Q0 Vector Search (k=100)
        q0_emb = np.array(pipeline.embeddings.embed_query(q_def["original_query"]))
        try:
            res_q0 = pipeline.vector_store.similarity_search_with_score_by_vector(q0_emb.tolist(), k=100, filter=search_filter)
        except Exception:
            res_q0 = pipeline.vector_store.similarity_search_with_score_by_vector(q0_emb.tolist(), k=100)
            res_q0 = [(doc, s) for doc, s in res_q0 if doc.metadata.get("session_id") == pipeline.session_id]
        sem_q0_100 = [format_chunk_item(doc, pipeline.all_chunks) for doc, _ in res_q0]

        # Q1 Vector Search (k=100)
        q1_emb = np.array(pipeline.embeddings.embed_query(q_def["q1_reformulated"]))
        try:
            res_q1 = pipeline.vector_store.similarity_search_with_score_by_vector(q1_emb.tolist(), k=100, filter=search_filter)
        except Exception:
            res_q1 = pipeline.vector_store.similarity_search_with_score_by_vector(q1_emb.tolist(), k=100)
            res_q1 = [(doc, s) for doc, s in res_q1 if doc.metadata.get("session_id") == pipeline.session_id]
        sem_q1_100 = [format_chunk_item(doc, pipeline.all_chunks) for doc, _ in res_q1]

        # Q2 Vector Search (k=100 for each of the 3 sub-queries)
        sem_q2_sub_100 = []
        for sq in q_def["q2_multiqueries"]:
            sq_emb = np.array(pipeline.embeddings.embed_query(sq))
            try:
                res_sq = pipeline.vector_store.similarity_search_with_score_by_vector(sq_emb.tolist(), k=100, filter=search_filter)
            except Exception:
                res_sq = pipeline.vector_store.similarity_search_with_score_by_vector(sq_emb.tolist(), k=100)
                res_sq = [(doc, s) for doc, s in res_sq if doc.metadata.get("session_id") == pipeline.session_id]
            sem_q2_sub_100.append([format_chunk_item(doc, pipeline.all_chunks) for doc, _ in res_sq])

        # BM25 Search on original query (k=100)
        bm25_raw = pipeline.bm25_retriever.invoke(q_def["original_query"]) if pipeline.bm25_retriever else []
        bm25_100 = [format_chunk_item(doc, pipeline.all_chunks) for doc in bm25_raw if doc.metadata.get("session_id") == pipeline.session_id][:100]

        retrieved_cache[q_id] = {
            "sem_q0_100": sem_q0_100,
            "sem_q1_100": sem_q1_100,
            "sem_q2_sub_100": sem_q2_sub_100,
            "bm25_100": bm25_100
        }

    # Execute 3x3 Matrix (Strategy x Candidate N)
    candidate_depths = [20, 50, 100]
    strategies = [
        {"id": "Q0", "name": "Original"},
        {"id": "Q1", "name": "Reformulated"},
        {"id": "Q2", "name": "Multi-query"}
    ]

    master_results = []
    per_question_details = []
    missing_chunk_tracing = []

    for strat in strategies:
        s_id = strat["id"]
        for N in candidate_depths:
            case_results = []
            
            for case in BENCHMARK_SUITE:
                q_id = case["q_id"]
                req_units = case["required_evidence"]
                gold_map = case["gold_map"]
                claims = case["claims"]

                cache = retrieved_cache[q_id]
                bm25_cand = cache["bm25_100"][:N]

                # Determine semantic candidates for strategy at depth N
                if s_id == "Q0":
                    sem_cand = cache["sem_q0_100"][:N]
                    sem_union_cids = [item["chunk_id"] for item in sem_cand]
                elif s_id == "Q1":
                    sem_cand = cache["sem_q1_100"][:N]
                    sem_union_cids = [item["chunk_id"] for item in sem_cand]
                elif s_id == "Q2":
                    sub_N_lists = [sub[:N] for sub in cache["sem_q2_sub_100"]]
                    # Combined semantic list ranked by Reciprocal Rank Sum
                    combined_sem_ranked = combine_multiquery_semantic(sub_N_lists, rrf_k=60)
                    sem_cand = combined_sem_ranked[:N] # Top-N of multi-query semantic candidate pool
                    # Candidate Union includes ALL unique candidates across sub-queries at depth N
                    sem_union_cids = [item["chunk_id"] for item in combined_sem_ranked]

                sem_cids = [item["chunk_id"] for item in sem_cand]
                bm25_cids = [item["chunk_id"] for item in bm25_cand]

                # Semantic Recall
                sem_retrieved_units = [cid for cid in req_units if cid in sem_cids]
                sem_recall = len(sem_retrieved_units) / len(req_units) if req_units else 0.0

                # Candidate Union Recall (for Q2)
                union_retrieved_units = [cid for cid in req_units if cid in sem_union_cids]
                union_recall = len(union_retrieved_units) / len(req_units) if req_units else 0.0

                # Hybrid Candidate Pool (Semantic top-N + BM25 top-N)
                hybrid_cand_cids = list(dict.fromkeys(sem_cids + bm25_cids))
                hybrid_retrieved_units = [cid for cid in req_units if cid in hybrid_cand_cids]
                hybrid_cand_recall = len(hybrid_retrieved_units) / len(req_units) if req_units else 0.0

                # RRF Ranking (80/20) over semantic & BM25 top-N
                rrf_ranked = execute_rrf_fusion(sem_cand, bm25_cand, w_vec=0.8, w_bm25=0.2, rrf_k=60)
                selected_top10 = rrf_ranked[:10]
                top10_cids = [item["chunk_id"] for item in selected_top10]

                # Final Top-10 Evidence Recall
                top10_retrieved_units = [cid for cid in req_units if cid in top10_cids]
                final_recall = len(top10_retrieved_units) / len(req_units) if req_units else 0.0

                # Claim Availability
                supported_claims = 0
                missing_claims = []
                for claim in claims:
                    if all(cid in top10_cids for cid in claim["required_chunks"]):
                        supported_claims += 1
                    else:
                        missing_claims.append(claim["claim_id"])
                claim_avail = supported_claims / len(claims) if claims else 0.0

                # Secondary metrics
                sec = calculate_secondary_metrics(selected_top10, gold_map, top_k=10)

                q_res = {
                    "q_id": q_id,
                    "strategy": s_id,
                    "N": N,
                    "req_units": req_units,
                    "sem_retrieved": sem_retrieved_units,
                    "sem_recall": sem_recall,
                    "union_recall": union_recall,
                    "hybrid_cand_recall": hybrid_cand_recall,
                    "final_recall": final_recall,
                    "claim_avail": claim_avail,
                    "missing_claims": missing_claims,
                    "top10_cids": top10_cids,
                    "secondary": sec
                }
                case_results.append(q_res)
                per_question_details.append(q_res)

                # Trace missing gold chunks
                for req_cid in req_units:
                    q0_rank = next((idx+1 for idx, item in enumerate(cache["sem_q0_100"]) if item["chunk_id"] == req_cid), None)
                    q1_rank = next((idx+1 for idx, item in enumerate(cache["sem_q1_100"]) if item["chunk_id"] == req_cid), None)
                    
                    q2_sub_ranks = []
                    for s_idx, s_list in enumerate(cache["sem_q2_sub_100"]):
                        r = next((idx+1 for idx, item in enumerate(s_list) if item["chunk_id"] == req_cid), None)
                        q2_sub_ranks.append(r)

                    final_rrf_item = next((item for item in rrf_ranked if item["chunk_id"] == req_cid), None)
                    final_rrf_rank = final_rrf_item["rrf_rank"] if final_rrf_item else None

                    missing_chunk_tracing.append({
                        "q_id": q_id,
                        "gold_chunk": req_cid,
                        "strategy": s_id,
                        "N": N,
                        "q0_sem_rank": q0_rank or "—",
                        "q1_sem_rank": q1_rank or "—",
                        "q2_sub_ranks": [r or "—" for r in q2_sub_ranks],
                        "in_sem_candidate_pool": req_cid in sem_cids,
                        "final_rrf_rank": final_rrf_rank or "—",
                        "in_final_top10": req_cid in top10_cids
                    })

            # Aggregate for Master Table
            tot_req = sum(r["req_units"].__len__() for r in case_results)
            tot_sem_ret = sum(len(r["sem_retrieved"]) for r in case_results)
            agg_sem_recall = tot_sem_ret / tot_req

            tot_union_ret = sum(int(r["union_recall"] * len(r["req_units"])) for r in case_results)
            agg_union_recall = tot_union_ret / tot_req

            tot_hyb_ret = sum(int(r["hybrid_cand_recall"] * len(r["req_units"])) for r in case_results)
            agg_hyb_recall = tot_hyb_ret / tot_req

            tot_final_ret = sum(int(r["final_recall"] * len(r["req_units"])) for r in case_results)
            agg_final_recall = tot_final_ret / tot_req

            tot_claims = sum(len(c["claims"]) for c in BENCHMARK_SUITE)
            tot_supp_claims = sum(int(r["claim_avail"] * len(next(c["claims"] for c in BENCHMARK_SUITE if c["q_id"] == r["q_id"]))) for r in case_results)
            agg_claim_avail = tot_supp_claims / tot_claims

            agg_ndcg = float(np.mean([r["secondary"]["ndcg10"] for r in case_results]))
            agg_mrr = float(np.mean([r["secondary"]["mrr"] for r in case_results]))

            master_results.append({
                "strategy_id": s_id,
                "strategy_name": strat["name"],
                "candidate_N": N,
                "sem_candidate_recall": agg_sem_recall,
                "union_candidate_recall": agg_union_recall,
                "hybrid_candidate_recall": agg_hyb_recall,
                "final_recall_at_10": agg_final_recall,
                "claim_availability": agg_claim_avail,
                "ndcg10": agg_ndcg,
                "mrr": agg_mrr
            })

    # Calculate Query Diversity (Jaccard similarity between Q_A, Q_B, Q_C)
    query_diversity_stats = []
    for case in BENCHMARK_SUITE:
        q_id = case["q_id"]
        cache = retrieved_cache[q_id]
        
        for N in [20, 50, 100]:
            sub_cids = [[item["chunk_id"] for item in s[:N]] for s in cache["sem_q2_sub_100"]]
            set_a, set_b, set_c = set(sub_cids[0]), set(sub_cids[1]), set(sub_cids[2])

            j_ab = len(set_a & set_b) / len(set_a | set_b) if (set_a | set_b) else 0.0
            j_ac = len(set_a & set_c) / len(set_a | set_c) if (set_a | set_c) else 0.0
            j_bc = len(set_b & set_c) / len(set_b | set_c) if (set_b | set_c) else 0.0

            unique_a = len(set_a - set_b - set_c)
            unique_b = len(set_b - set_a - set_c)
            unique_c = len(set_c - set_a - set_b)

            query_diversity_stats.append({
                "q_id": q_id,
                "candidate_N": N,
                "jaccard_ab": j_ab,
                "jaccard_ac": j_ac,
                "jaccard_bc": j_bc,
                "avg_jaccard": (j_ab + j_ac + j_bc) / 3.0,
                "unique_a": unique_a,
                "unique_b": unique_b,
                "unique_c": unique_c,
                "total_union_size": len(set_a | set_b | set_c)
            })

    # Print Master Table
    print("\n=========================================================================", flush=True)
    print("                PHASE 7A — 3x3 MASTER RESULTS TABLE                       ", flush=True)
    print("=========================================================================", flush=True)
    print(f"{'Strategy':13s} | {'N':>4s} | {'SemCandRec':10s} | {'HybCandRec':10s} | {'FinalRec@10':11s} | {'ClaimAvail':10s} | {'nDCG@10':7s} | {'MRR':5s}", flush=True)
    print("-" * 85, flush=True)
    for row in master_results:
        print(f"{row['strategy_name']:13s} | {row['candidate_N']:4d} | {row['sem_candidate_recall']:10.1%} | {row['hybrid_candidate_recall']:10.1%} | {row['final_recall_at_10']:11.1%} | {row['claim_availability']:10.1%} | {row['ndcg10']:7.3f} | {row['mrr']:5.3f}", flush=True)

    # Save output payload
    out_payload = {
        "timestamp": datetime.now().isoformat(),
        "query_definitions": query_definitions,
        "master_results": master_results,
        "query_diversity_stats": query_diversity_stats,
        "missing_chunk_tracing": missing_chunk_tracing,
        "per_question_details": per_question_details
    }

    out_file = os.path.join(current_dir, "results", "phase7a_diagnostic_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2, ensure_ascii=False)

    print(f"\nPhase 7A diagnostic complete. Saved results to:\n{out_file}\n", flush=True)

if __name__ == "__main__":
    main()
