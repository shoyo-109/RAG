import os
import sys
import json
import math
import time
import re
import importlib.util
from datetime import datetime
import numpy as np
from dotenv import load_dotenv
import tiktoken
from sentence_transformers import CrossEncoder

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

# Tiktoken encoder for exact token counts
enc = tiktoken.get_encoding("cl100k_base")

# Benchmark Questions and Gold Mappings
BENCHMARK_SUITE = [
    {
        "q_id": "Q01",
        "question": "Who is Aegeon and what is his origin?",
        "gold_map": {"C4": "PRIMARY", "C2": "SUPPORTING"}
    },
    {
        "q_id": "Q02",
        "question": "Why was Aegeon sentenced to death in Ephesus?",
        "gold_map": {"C2": "PRIMARY", "C4": "SUPPORTING", "C10": "SUPPORTING", "C12": "SUPPORTING"}
    },
    {
        "q_id": "Q03",
        "question": "Why does Antipholus of Syracuse fail to recognize Adriana when she addresses him as her husband?",
        "gold_map": {"C15": "PRIMARY", "C24": "PRIMARY", "C29": "SUPPORTING", "C61": "SUPPORTING"}
    },
    {
        "q_id": "Q04",
        "question": "How did Aegeon come to have two pairs of identical twin sons and twin servants?",
        "gold_map": {"C4": "PRIMARY", "C6": "SUPPORTING", "C8": "SUPPORTING"}
    },
    {
        "q_id": "Q05",
        "question": "How do Antipholus of Syracuse and Antipholus of Ephesus differ in their initial interactions with Dromio of Syracuse?",
        "gold_map": {"C42": "PRIMARY", "C65": "PRIMARY", "C12": "SUPPORTING"}
    },
    {
        "q_id": "Q06",
        "question": "Why is Antipholus of Ephesus arrested, and what role does Angelo the goldsmith play in the debt dispute?",
        "gold_map": {"C105": "PRIMARY", "C107": "PRIMARY", "C147": "SUPPORTING", "C155": "SUPPORTING"}
    }
]

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

TARGET_CHUNKS = ["C15", "C29", "C42", "C61", "C147"]

def format_chunk_item(doc, pipeline_chunks):
    el_id = doc.metadata.get("element_id")
    c_idx = next((i for i, c in enumerate(pipeline_chunks) if c.metadata.get("element_id") == el_id), -1)
    chunk_label = f"C{c_idx}" if c_idx >= 0 else str(el_id)
    return {
        "chunk_id": chunk_label,
        "chunk_idx": c_idx,
        "element_id": el_id,
        "page": doc.metadata.get("page"),
        "breadcrumb_path": doc.metadata.get("breadcrumb_path", ""),
        "parent_id": doc.metadata.get("parent_id") or doc.metadata.get("parent_text", "")[:80],
        "parent_text": doc.metadata.get("parent_text", ""),
        "text": doc.page_content
    }

def main():
    print("==========================================================================", flush=True)
    print("      EXECUTING PHASE 9A — FAILED EVIDENCE AUTOPSY DIAGNOSTIC SCRIPT      ", flush=True)
    print("==========================================================================", flush=True)

    # Initialize Cross-Encoder model
    ce_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    print(f"Loading Cross-Encoder model: '{ce_model_name}'...", flush=True)
    ce_model = CrossEncoder(ce_model_name)

    doc_path = os.path.join(root_dir, "Documents", "The Comedy of Errors document.pdf")
    session_id = f"phase9a_autopsy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    pipeline = AdvancedRAGPipeline(session_id=session_id)

    print("Ingesting document into pipeline...", flush=True)
    raw_docs = IngestionPipeline.load_document(doc_path)
    pipeline.add_documents(raw_docs)

    chunk_count = len(pipeline.all_chunks)
    print(f"Total indexed chunks: {chunk_count}.", flush=True)
    if chunk_count != 203:
        raise ValueError(f"CRITICAL DRIFT ERROR: Chunk count is {chunk_count}, expected 203!")

    # Format all 203 chunks into list
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
    chunk_by_id = {c["chunk_id"]: c for c in formatted_chunks}

    # Embed all 203 chunks once for exact dense ranking across the entire corpus
    print("Embedding all 203 chunks for exact dense ranking analysis...", flush=True)
    all_texts = [c["text"] for c in formatted_chunks]
    doc_embeddings = np.array(pipeline.embeddings.embed_documents(all_texts)) # shape (203, 384)
    # Normalize doc embeddings
    doc_norms = np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
    doc_norms[doc_norms == 0] = 1.0
    doc_embeddings_norm = doc_embeddings / doc_norms

    # BM25 setup for exact scoring
    if pipeline.bm25_retriever:
        pipeline.bm25_retriever.k = 203

    search_filter = Filter(must=[FieldCondition(key="metadata.session_id", match=MatchValue(value=pipeline.session_id))])

    # Run complete rank analysis across all 203 chunks for target chunks
    autopsy_results = {}

    for target_cid in TARGET_CHUNKS:
        target_chunk = chunk_by_id[target_cid]
        c_idx = target_chunk["chunk_idx"]
        
        # Local context
        prev_chunk = chunk_by_id[f"C{c_idx - 1}"] if c_idx > 0 else None
        next_chunk = chunk_by_id[f"C{c_idx + 1}"] if c_idx < 202 else None

        # Find target question
        q_info = None
        for b_case in BENCHMARK_SUITE:
            if target_cid in b_case["gold_map"]:
                q_info = b_case
                break
        
        q_id = q_info["q_id"]
        q_text = q_info["question"]
        gold_role = q_info["gold_map"][target_cid]
        subqueries = FROZEN_Q2_QUERIES[q_id]

        # 1. Exact Semantic Dense Rank across ALL 203 chunks
        q0_emb = np.array(pipeline.embeddings.embed_query(q_text))
        q0_norm = np.linalg.norm(q0_emb)
        q0_emb_norm = q0_emb / q0_norm if q0_norm > 0 else q0_emb
        dense_sims = np.dot(doc_embeddings_norm, q0_emb_norm) # shape (203,)
        
        # Rank chunks descending by similarity
        dense_ranked_indices = np.argsort(-dense_sims)
        dense_ranked_cids = [f"C{i}" for i in dense_ranked_indices]
        q0_dense_rank = dense_ranked_cids.index(target_cid) + 1
        q0_dense_sim = float(dense_sims[c_idx])

        # Subquery Exact Dense Ranks
        subquery_ranks = {}
        for sq_idx, sq in enumerate(subqueries, 1):
            sq_emb = np.array(pipeline.embeddings.embed_query(sq))
            sq_norm = np.linalg.norm(sq_emb)
            sq_emb_norm = sq_emb / sq_norm if sq_norm > 0 else sq_emb
            sq_sims = np.dot(doc_embeddings_norm, sq_emb_norm)
            sq_ranked_indices = np.argsort(-sq_sims)
            sq_ranked_cids = [f"C{i}" for i in sq_ranked_indices]
            sq_rank = sq_ranked_cids.index(target_cid) + 1
            sq_sim = float(sq_sims[c_idx])
            subquery_ranks[f"subquery_{sq_idx}"] = {
                "query": sq,
                "rank": sq_rank,
                "cosine_sim": round(sq_sim, 4)
            }

        # 2. Exact BM25 Rank across ALL 203 chunks
        if pipeline.bm25_retriever:
            bm25_docs = pipeline.bm25_retriever.invoke(q_text)
            bm25_cids = []
            for doc in bm25_docs:
                el_id = doc.metadata.get("element_id")
                c_item = next((c for c in formatted_chunks if c["element_id"] == el_id), None)
                if c_item and c_item["chunk_id"] not in bm25_cids:
                    bm25_cids.append(c_item["chunk_id"])
            bm25_rank = bm25_cids.index(target_cid) + 1 if target_cid in bm25_cids else "N/A (> len)"
            bm25_score = f"Rank {bm25_rank}"
        else:
            bm25_rank = "N/A"
            bm25_score = "N/A"

        # 3. Frozen Q2 Candidate Union Presence & Ranks at N=20, N=50, N=100
        depth_analysis = {}
        for N in [20, 50, 100]:
            # Construct Q2 candidate union at depth N
            q2_cand_map = {}
            for sq_idx, sq in enumerate(subqueries, 1):
                sq_emb = np.array(pipeline.embeddings.embed_query(sq))
                try:
                    res_sq_N = pipeline.vector_store.similarity_search_with_score_by_vector(sq_emb.tolist(), k=N, filter=search_filter)
                except Exception:
                    res_sq_N = pipeline.vector_store.similarity_search_with_score_by_vector(sq_emb.tolist(), k=N)
                    res_sq_N = [(doc, s) for doc, s in res_sq_N if doc.metadata.get("session_id") == pipeline.session_id]
                for doc, s in res_sq_N:
                    c_item = format_chunk_item(doc, pipeline.all_chunks)
                    q2_cand_map[c_item["chunk_id"]] = c_item
            
            # Add BM25 top N
            bm25_N = bm25_cids[:N]
            for cid in bm25_N:
                q2_cand_map[cid] = chunk_by_id[cid]

            q2_cand_list = list(q2_cand_map.values())
            q2_cand_ids = list(q2_cand_map.keys())

            present_in_cand = target_cid in q2_cand_ids

            # Rerank candidates with Cross-Encoder if present, or rerank all candidates + target chunk
            ce_score = "N/A"
            ce_rank = "N/A"
            if present_in_cand:
                pairs = [(q_text, doc["text"]) for doc in q2_cand_list]
                scores = ce_model.predict(pairs, batch_size=32, show_progress_bar=False)
                ranked_pairs = sorted(zip(q2_cand_ids, scores), key=lambda x: x[1], reverse=True)
                for rank_idx, (cid, score) in enumerate(ranked_pairs, 1):
                    if cid == target_cid:
                        ce_rank = rank_idx
                        ce_score = float(score)
                        break
            else:
                # Target chunk is NOT in candidate pool, compute CE score directly vs question
                direct_score = float(ce_model.predict([(q_text, target_chunk["text"])])[0])
                ce_score = f"{direct_score:.4f} (Direct - Not in Pool)"

            depth_analysis[f"N_{N}"] = {
                "in_candidate_pool": present_in_cand,
                "candidate_pool_size": len(q2_cand_list),
                "ce_rank": ce_rank,
                "ce_score": ce_score
            }

        autopsy_results[target_cid] = {
            "chunk_id": target_cid,
            "element_id": target_chunk["element_id"],
            "character_count": len(target_chunk["text"]),
            "token_count": len(enc.encode(target_chunk["text"])),
            "word_count": len(target_chunk["text"].split()),
            "raw_text": target_chunk["text"],
            "local_context": {
                "previous_chunk": {
                    "chunk_id": prev_chunk["chunk_id"] if prev_chunk else "N/A",
                    "text": prev_chunk["text"] if prev_chunk else "N/A"
                },
                "target_chunk": {
                    "chunk_id": target_cid,
                    "text": target_chunk["text"]
                },
                "next_chunk": {
                    "chunk_id": next_chunk["chunk_id"] if next_chunk else "N/A",
                    "text": next_chunk["text"] if next_chunk else "N/A"
                }
            },
            "structural_info": {
                "page": target_chunk["page"],
                "breadcrumb_path": target_chunk["breadcrumb_path"],
                "parent_id": target_chunk["parent_id"],
                "parent_text_snippet": target_chunk["parent_text"][:120] if target_chunk["parent_text"] else "N/A"
            },
            "gold_relationship": {
                "question_id": q_id,
                "question_text": q_text,
                "gold_role": gold_role
            },
            "retrieval_trace": {
                "q0_dense_rank_all203": q0_dense_rank,
                "subquery_dense_ranks": subquery_ranks,
                "bm25_rank_all203": bm25_rank,
                "depth_analysis": depth_analysis
            }
        }

    # Save to JSON artifact
    os.makedirs(os.path.join(root_dir, "evaluation", "results"), exist_ok=True)
    json_path = os.path.join(root_dir, "evaluation", "results", "phase9a_autopsy_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(autopsy_results, f, indent=2)

    print(f"\nAutopsy raw results successfully saved to: {json_path}", flush=True)

    # Print Summary Table
    print("\nTarget Chunks Retrieval Summary:")
    print("=" * 120)
    print(f"{'Chunk':6s} | {'Q_ID':4s} | {'Role':10s} | {'Char/Tok':10s} | {'DenseQ0':8s} | {'BM25':6s} | {'In N20':7s} | {'In N50':7s} | {'CE Rank (N50)':13s} | {'CE Score':12s}")
    print("=" * 120)
    for cid in TARGET_CHUNKS:
        res = autopsy_results[cid]
        q_id = res["gold_relationship"]["question_id"]
        role = res["gold_relationship"]["gold_role"]
        chartok = f"{res['character_count']}/{res['token_count']}"
        dq0 = str(res["retrieval_trace"]["q0_dense_rank_all203"])
        bm25 = str(res["retrieval_trace"]["bm25_rank_all203"])
        in20 = "YES" if res["retrieval_trace"]["depth_analysis"]["N_20"]["in_candidate_pool"] else "NO"
        in50 = "YES" if res["retrieval_trace"]["depth_analysis"]["N_50"]["in_candidate_pool"] else "NO"
        ce_r = str(res["retrieval_trace"]["depth_analysis"]["N_50"]["ce_rank"])
        ce_s = str(res["retrieval_trace"]["depth_analysis"]["N_50"]["ce_score"])
        print(f"{cid:6s} | {q_id:4s} | {role:10s} | {chartok:10s} | {dq0:8s} | {bm25:6s} | {in20:7s} | {in50:7s} | {ce_r:13s} | {ce_s:12s}")

if __name__ == "__main__":
    main()
