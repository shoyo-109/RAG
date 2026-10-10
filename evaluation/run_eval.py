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

# Load IngestionPipeline
doc_loader_pipeline_path = os.path.join(doc_loader_dir, "pipeline.py")
spec = importlib.util.spec_from_file_location("doc_loader_pipeline", doc_loader_pipeline_path)
doc_loader_module = importlib.util.module_from_spec(spec)
sys.modules["doc_loader_pipeline"] = doc_loader_module
spec.loader.exec_module(doc_loader_module)
IngestionPipeline = doc_loader_module.IngestionPipeline

from advance_rag import AdvancedRAGPipeline
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

# Define Benchmark Suite with Manually Verified Gold Annotations & Categories
BENCHMARK_CASES = [
    {
        "question_id": "Q01",
        "question": "Who is Aegeon and what is his origin?",
        "question_type": "direct",
        "gold_evidence": [
            {"chunk_id": "C4", "role": "primary", "desc": "Born in Syracuse, married, merchant who traveled to Epidamnum"},
            {"chunk_id": "C2", "role": "supporting", "desc": "Duke Solinus addresses him as Syracusian merchant"}
        ]
    },
    {
        "question_id": "Q02",
        "question": "Why was Aegeon sentenced to death in Ephesus?",
        "question_type": "causal",
        "gold_evidence": [
            {"chunk_id": "C2", "role": "primary", "desc": "Statute prohibiting Syracusians in Ephesus on pain of death/1000 marks ransom, Aegeon wealth under 100 marks"},
            {"chunk_id": "C4", "role": "supporting", "desc": "Aegeon origin in Syracuse"},
            {"chunk_id": "C10", "role": "supporting", "desc": "Duke Solinus grants 1 day to beg/borrow 1000 marks before execution"},
            {"chunk_id": "C12", "role": "supporting", "desc": "First Merchant confirms Syracusian merchant apprehended and unable to buy out life"}
        ]
    },
    {
        "question_id": "Q03",
        "question": "Why does Antipholus of Syracuse fail to recognize Adriana when she addresses him as her husband?",
        "question_type": "multi_hop",
        "gold_evidence": [
            {"chunk_id": "C15", "role": "primary", "desc": "Antipholus of Syracuse in Ephesus seeking twin brother, unaware twin lives there"},
            {"chunk_id": "C24", "role": "primary", "desc": "Adriana is married to Antipholus of Ephesus"},
            {"chunk_id": "C29", "role": "supporting", "desc": "Dromio of Ephesus mistakes Antipholus of Syracuse for his master"},
            {"chunk_id": "C61", "role": "supporting", "desc": "Adriana confronts Antipholus of Syracuse mistaking him for her husband"}
        ]
    },
    {
        "question_id": "Q04",
        "question": "How did Aegeon come to have two pairs of identical twin sons and twin servants?",
        "question_type": "relationship",
        "gold_evidence": [
            {"chunk_id": "C4", "role": "primary", "desc": "Wife gave birth to identical twin sons in Epidamnum, poor woman birthed twin boys bought by Aegeon"},
            {"chunk_id": "C6", "role": "supporting", "desc": "Storm separated family into two groups on spare mast"},
            {"chunk_id": "C8", "role": "supporting", "desc": "Ship split on rock; one set saved by Corinth fishermen, other by another ship"}
        ]
    },
    {
        "question_id": "Q05",
        "question": "How do Antipholus of Syracuse and Antipholus of Ephesus differ in their initial interactions with Dromio of Syracuse?",
        "question_type": "comparative",
        "gold_evidence": [
            {"chunk_id": "C12", "role": "supporting", "desc": "Antipholus of Syracuse trusts Dromio of Syracuse with money"},
            {"chunk_id": "C42", "role": "primary", "desc": "Antipholus of Syracuse beats Dromio of Syracuse over money confusion"},
            {"chunk_id": "C65", "role": "primary", "desc": "Antipholus of Ephesus locked out by Dromio of Syracuse inside house"}
        ]
    },
    {
        "question_id": "Q06",
        "question": "Why is Antipholus of Ephesus arrested, and what role does Angelo the goldsmith play in the debt dispute?",
        "question_type": "event_chain",
        "gold_evidence": [
            {"chunk_id": "C105", "role": "primary", "desc": "Angelo owes 200 ducats to Second Merchant and needs payment for chain"},
            {"chunk_id": "C107", "role": "primary", "desc": "Angelo gave chain to Antipholus of Syracuse by mistake, demands payment from Ephesus twin"},
            {"chunk_id": "C147", "role": "supporting", "desc": "Antipholus of Ephesus denies getting chain, Angelo has him arrested"},
            {"chunk_id": "C155", "role": "supporting", "desc": "Officer holds Antipholus of Ephesus under arrest for 200 ducats debt"}
        ]
    }
]


def classify_retrieved_chunk(chunk_id: str, doc_text: str, gold_ids: set, gold_map: dict, seen_parent_texts: set, parent_text: str) -> str:
    """Classifies a retrieved chunk as PRIMARY, SUPPORTING, RELEVANT_BUT_UNNECESSARY, IRRELEVANT, or DUPLICATE."""
    # Check duplicate parent text
    if parent_text in seen_parent_texts:
        return "DUPLICATE"

    # Check direct match with gold evidence
    if chunk_id in gold_ids:
        role = gold_map[chunk_id]["role"].upper()
        return role  # PRIMARY or SUPPORTING

    # Check if text contains core keywords for the query domain
    t_lower = doc_text.lower()
    if any(kw in t_lower for kw in ["aegeon", "syracuse", "ephesus", "antipholus", "dromio", "adriana", "goldsmith", "angelo", "chain", "ducats", "twin", "abbess"]):
        return "RELEVANT_BUT_UNNECESSARY"

    return "IRRELEVANT"


def calculate_ndcg(retrieved_classifications: list, gold_count: int, k: int = 10) -> float:
    """Calculates nDCG@K."""
    rel_map = {
        "PRIMARY": 3,
        "SUPPORTING": 2,
        "RELEVANT_BUT_UNNECESSARY": 1,
        "IRRELEVANT": 0,
        "DUPLICATE": 0
    }
    
    dcg = 0.0
    for idx, cls_val in enumerate(retrieved_classifications[:k]):
        rel = rel_map.get(cls_val, 0)
        dcg += (2**rel - 1) / math.log2(idx + 2)

    # Ideal DCG: top items are all PRIMARY (3) then SUPPORTING (2)
    ideal_rels = sorted([3] * gold_count + [2] * max(0, k - gold_count), reverse=True)[:k]
    idcg = 0.0
    for idx, rel in enumerate(ideal_rels):
        idcg += (2**rel - 1) / math.log2(idx + 2)

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval_set(retrieved_chunks_info: list, gold_evidence: list, k_list=[1, 3, 5, 10]):
    """Computes all precision, recall, ranking, redundancy, and completeness metrics for a single retrieval list."""
    gold_ids = {g["chunk_id"] for g in gold_evidence}
    gold_map = {g["chunk_id"]: g for g in gold_evidence}
    gold_count = len(gold_ids)

    seen_parents = set()
    retrieved_classifications = []
    useful_ranks = []
    
    unique_chunk_ids = set()
    unique_parent_ids = set()
    unique_scenes = set()
    duplicate_count = 0

    for idx, item in enumerate(retrieved_chunks_info, 1):
        cid = item["chunk_id"]
        unique_chunk_ids.add(cid)
        
        parent_text = item.get("parent_text", item["text"])
        unique_parent_ids.add(parent_text[:100])
        
        scene = item.get("breadcrumb_path", "General")
        unique_scenes.add(scene)

        classification = classify_retrieved_chunk(cid, item["text"], gold_ids, gold_map, seen_parents, parent_text)
        
        if classification == "DUPLICATE":
            duplicate_count += 1
        else:
            seen_parents.add(parent_text)

        retrieved_classifications.append(classification)

        if classification in ["PRIMARY", "SUPPORTING"]:
            useful_ranks.append(idx)

    # Compute Recall@K & Precision@K
    retrieved_cids = [item["chunk_id"] for item in retrieved_chunks_info]
    
    recalls = {}
    precisions = {}
    for k in k_list:
        ret_k_cids = set(retrieved_cids[:k])
        hits = len(gold_ids.intersection(ret_k_cids))
        recalls[f"recall_at_{k}"] = hits / gold_count if gold_count > 0 else 0.0
        
        # Useful precision @ K
        ret_k_cls = retrieved_classifications[:k]
        useful_count = sum(1 for c in ret_k_cls if c in ["PRIMARY", "SUPPORTING"])
        precisions[f"precision_at_{k}"] = useful_count / k

    # Breakdown of Precision @ 10
    cls_10 = retrieved_classifications[:10]
    total_10 = len(cls_10) if cls_10 else 1
    primary_pct = sum(1 for c in cls_10 if c == "PRIMARY") / total_10
    supporting_pct = sum(1 for c in cls_10 if c == "SUPPORTING") / total_10
    rel_unnecessary_pct = sum(1 for c in cls_10 if c == "RELEVANT_BUT_UNNECESSARY") / total_10
    irrelevant_pct = sum(1 for c in cls_10 if c == "IRRELEVANT") / total_10
    duplicate_pct = sum(1 for c in cls_10 if c == "DUPLICATE") / total_10

    # MRR
    mrr = 1.0 / useful_ranks[0] if useful_ranks else 0.0

    # nDCG@10
    ndcg_10 = calculate_ndcg(retrieved_classifications, gold_count, k=10)

    # Evidence Completeness
    retrieved_all_set = set(retrieved_cids[:10])
    at_least_one = len(gold_ids.intersection(retrieved_all_set)) > 0
    all_required = gold_ids.issubset(retrieved_all_set)
    
    # Logically sufficient: all PRIMARY gold evidence retrieved
    primary_gold_ids = {g["chunk_id"] for g in gold_evidence if g["role"].lower() == "primary"}
    logically_sufficient = primary_gold_ids.issubset(retrieved_all_set) if primary_gold_ids else all_required

    # Redundancy
    redundancy_ratio = duplicate_count / len(retrieved_chunks_info) if retrieved_chunks_info else 0.0

    return {
        "recalls": recalls,
        "precisions": precisions,
        "primary_precision_at_10": primary_pct,
        "supporting_precision_at_10": supporting_pct,
        "relevant_unnecessary_pct_at_10": rel_unnecessary_pct,
        "irrelevant_pct_at_10": irrelevant_pct,
        "duplicate_pct_at_10": duplicate_pct,
        "mrr": mrr,
        "ndcg_at_10": ndcg_10,
        "evidence_completeness": {
            "at_least_one_relevant": at_least_one,
            "all_required_evidence": all_required,
            "logically_sufficient": logically_sufficient
        },
        "redundancy": {
            "total_retrieved": len(retrieved_chunks_info),
            "unique_chunk_ids": len(unique_chunk_ids),
            "unique_parent_ids": len(unique_parent_ids),
            "unique_scene_ids": len(unique_scenes),
            "duplicate_count": duplicate_count,
            "redundancy_ratio": redundancy_ratio
        },
        "classifications": retrieved_classifications
    }


def main():
    print("=========================================================================")
    print("      EXECUTING RAG RETRIEVAL EVALUATION (BASELINE DIAGNOSIS)           ")
    print("=========================================================================")
    
    doc_path = os.path.join(root_dir, "Documents", "The Comedy of Errors document.pdf")
    if not os.path.exists(doc_path):
        print(f"Error: Document not found at {doc_path}")
        sys.exit(1)

    session_id = f"eval_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    pipeline = AdvancedRAGPipeline(session_id=session_id)
    
    print("Ingesting PDF document...")
    raw_docs = IngestionPipeline.load_document(doc_path)
    pipeline.add_documents(raw_docs)
    print(f"Total indexed chunks: {len(pipeline.all_chunks)} in session '{session_id}'.\n")

    # Map chunk index to element_id for easy lookup
    chunk_by_cid = {}
    for idx, c in enumerate(pipeline.all_chunks):
        cid = f"C{idx}"
        chunk_by_cid[cid] = c

    full_evaluation_results = []
    
    # Storage for aggregate calculations: method -> metric_name -> list of values
    methods = ["semantic", "bm25", "rrf"]
    agg_metrics = {m: {
        "recall_1": [], "recall_3": [], "recall_5": [], "recall_10": [],
        "precision_1": [], "precision_3": [], "precision_5": [], "precision_10": [],
        "mrr": [], "ndcg_10": [], "completeness": [], "redundancy": []
    } for m in methods}

    category_metrics = {}

    for case in BENCHMARK_CASES:
        q_id = case["question_id"]
        q_text = case["question"]
        q_type = case["question_type"]
        gold_evidence = case["gold_evidence"]

        print(f"[{q_id}] ({q_type.upper()}) {q_text}")
        query_emb = np.array(pipeline.embeddings.embed_query(q_text))

        # 1. Semantic Search
        search_filter = Filter(must=[FieldCondition(key="metadata.session_id", match=MatchValue(value=session_id))])
        try:
            vec_res = pipeline.vector_store.similarity_search_with_score_by_vector(query_emb.tolist(), k=10, filter=search_filter)
        except Exception:
            vec_res = pipeline.vector_store.similarity_search_with_score_by_vector(query_emb.tolist(), k=10)
            vec_res = [(doc, s) for doc, s in vec_res if doc.metadata.get("session_id") == session_id]

        semantic_chunks = []
        for doc, score in vec_res:
            cid = doc.metadata.get("element_id")
            # Find index
            c_idx = next((i for i, c in enumerate(pipeline.all_chunks) if c.metadata.get("element_id") == cid), -1)
            c_label = f"C{c_idx}" if c_idx >= 0 else str(cid)
            semantic_chunks.append({
                "chunk_id": c_label,
                "element_id": cid,
                "text": doc.page_content,
                "parent_text": doc.metadata.get("parent_text", doc.page_content),
                "breadcrumb_path": doc.metadata.get("breadcrumb_path", ""),
                "score": float(score)
            })

        # 2. BM25 Search
        bm25_raw = pipeline.bm25_retriever.invoke(q_text) if pipeline.bm25_retriever else []
        bm25_filtered = [doc for doc in bm25_raw if doc.metadata.get("session_id") == session_id][:10]
        
        bm25_chunks = []
        for rank_idx, doc in enumerate(bm25_filtered, 1):
            cid = doc.metadata.get("element_id")
            c_idx = next((i for i, c in enumerate(pipeline.all_chunks) if c.metadata.get("element_id") == cid), -1)
            c_label = f"C{c_idx}" if c_idx >= 0 else str(cid)
            bm25_chunks.append({
                "chunk_id": c_label,
                "element_id": cid,
                "text": doc.page_content,
                "parent_text": doc.metadata.get("parent_text", doc.page_content),
                "breadcrumb_path": doc.metadata.get("breadcrumb_path", ""),
                "bm25_rank": rank_idx
            })

        # 3. RRF Hybrid Search
        rrf_docs, max_rrf_score, top_vec_score = pipeline.custom_hybrid_search(q_text, query_emb, top_k=10, rrf_k=60)
        rrf_chunks = []
        for rank_idx, doc in enumerate(rrf_docs, 1):
            cid = doc.metadata.get("element_id")
            c_idx = next((i for i, c in enumerate(pipeline.all_chunks) if c.metadata.get("element_id") == cid), -1)
            c_label = f"C{c_idx}" if c_idx >= 0 else str(cid)
            rrf_chunks.append({
                "chunk_id": c_label,
                "element_id": cid,
                "text": doc.page_content,
                "parent_text": doc.metadata.get("parent_text", doc.page_content),
                "breadcrumb_path": doc.metadata.get("breadcrumb_path", ""),
                "rrf_rank": rank_idx
            })

        # Evaluate each search list
        sem_eval = evaluate_retrieval_set(semantic_chunks, gold_evidence)
        bm25_eval = evaluate_retrieval_set(bm25_chunks, gold_evidence)
        rrf_eval = evaluate_retrieval_set(rrf_chunks, gold_evidence)

        # Record metrics for aggregation
        eval_map = {"semantic": sem_eval, "bm25": bm25_eval, "rrf": rrf_eval}
        for m in methods:
            ev = eval_map[m]
            agg_metrics[m]["recall_1"].append(ev["recalls"]["recall_at_1"])
            agg_metrics[m]["recall_3"].append(ev["recalls"]["recall_at_3"])
            agg_metrics[m]["recall_5"].append(ev["recalls"]["recall_at_5"])
            agg_metrics[m]["recall_10"].append(ev["recalls"]["recall_at_10"])
            
            agg_metrics[m]["precision_1"].append(ev["precisions"]["precision_at_1"])
            agg_metrics[m]["precision_3"].append(ev["precisions"]["precision_at_3"])
            agg_metrics[m]["precision_5"].append(ev["precisions"]["precision_at_5"])
            agg_metrics[m]["precision_10"].append(ev["precisions"]["precision_at_10"])
            
            agg_metrics[m]["mrr"].append(ev["mrr"])
            agg_metrics[m]["ndcg_10"].append(ev["ndcg_at_10"])
            agg_metrics[m]["completeness"].append(1.0 if ev["evidence_completeness"]["logically_sufficient"] else 0.0)
            agg_metrics[m]["redundancy"].append(ev["redundancy"]["redundancy_ratio"])

        # Category breakdown tracking (for RRF)
        if q_type not in category_metrics:
            category_metrics[q_type] = {"recall": [], "precision": [], "completeness": [], "redundancy": []}
        category_metrics[q_type]["recall"].append(rrf_eval["recalls"]["recall_at_10"])
        category_metrics[q_type]["precision"].append(rrf_eval["precisions"]["precision_at_10"])
        category_metrics[q_type]["completeness"].append(1.0 if rrf_eval["evidence_completeness"]["logically_sufficient"] else 0.0)
        category_metrics[q_type]["redundancy"].append(rrf_eval["redundancy"]["redundancy_ratio"])

        # Diagnosis string
        gold_cids = [g["chunk_id"] for g in gold_evidence]
        ret_rrf_cids = [item["chunk_id"] for item in rrf_chunks]
        missing_gold = set(gold_cids) - set(ret_rrf_cids)
        
        if not missing_gold and rrf_eval["redundancy"]["redundancy_ratio"] < 0.3:
            diag = "Optimal retrieval: Complete evidence chain retrieved with low redundancy."
        elif not missing_gold:
            diag = f"Relevant evidence retrieved completely, but high redundancy ({rrf_eval['redundancy']['redundancy_ratio']:.0%}) consumes Top-K budget."
        elif len(missing_gold) < len(gold_cids):
            diag = f"Partial evidence retrieved ({len(gold_cids)-len(missing_gold)}/{len(gold_cids)} chunks). Missing critical context: {list(missing_gold)}."
        else:
            diag = "Retrieval failure: None of the required gold evidence chunks were retrieved in Top-10."

        case_record = {
            "question_id": q_id,
            "question": q_text,
            "question_type": q_type,
            "gold_evidence": gold_evidence,
            "retrieval": {
                "semantic_results": [item["chunk_id"] for item in semantic_chunks],
                "bm25_results": [item["chunk_id"] for item in bm25_chunks],
                "rrf_results": [item["chunk_id"] for item in rrf_chunks]
            },
            "evaluation": {
                "semantic": sem_eval,
                "bm25": bm25_eval,
                "rrf": rrf_eval
            },
            "diagnosis": diag
        }
        full_evaluation_results.append(case_record)

        print(f"   RRF Recall@10: {rrf_eval['recalls']['recall_at_10']:.0%} | Precision@10: {rrf_eval['precisions']['precision_at_10']:.0%} | Sufficient: {rrf_eval['evidence_completeness']['logically_sufficient']} | Redundancy: {rrf_eval['redundancy']['redundancy_ratio']:.0%}")
        print(f"   Diagnosis: {diag}\n")

    # Compute Averages for Aggregate Table
    summary_table = {}
    for m in methods:
        summary_table[m] = {metric_k: float(np.mean(vals)) for metric_k, vals in agg_metrics[m].items()}

    cat_summary = {}
    for q_t, val_dict in category_metrics.items():
        cat_summary[q_t] = {
            "recall": float(np.mean(val_dict["recall"])),
            "precision": float(np.mean(val_dict["precision"])),
            "completeness": float(np.mean(val_dict["completeness"])),
            "redundancy": float(np.mean(val_dict["redundancy"]))
        }

    output_payload = {
        "run_id": session_id,
        "corpus": "The Comedy of Errors document.pdf",
        "total_chunks_indexed": len(pipeline.all_chunks),
        "cases": full_evaluation_results,
        "aggregate_summary": summary_table,
        "category_summary": cat_summary
    }

    output_file = os.path.join(current_dir, "results", f"baseline_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)

    print(f"Evaluation complete. Results saved to:\n{output_file}\n")


if __name__ == "__main__":
    main()
