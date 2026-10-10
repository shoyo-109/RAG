import json
import math
import numpy as np

# Load raw audit dump
with open('evaluation/audit/raw_retrieval_dumps.json', 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

# Corrected Gold Evidence Definitions
GOLD_ANNOTATIONS = {
    "Q01": {
        "gold": {"C4": "PRIMARY", "C2": "SUPPORTING"},
        "desc": "Who is Aegeon and what is his origin?"
    },
    "Q02": {
        "gold": {"C2": "PRIMARY", "C4": "SUPPORTING", "C10": "SUPPORTING", "C12": "SUPPORTING"},
        "desc": "Why was Aegeon sentenced to death in Ephesus?"
    },
    "Q03": {
        "gold": {"C15": "PRIMARY", "C24": "PRIMARY", "C29": "SUPPORTING", "C50": "SUPPORTING"},
        "desc": "Why does Antipholus of Syracuse fail to recognize Adriana when she addresses him as her husband?"
    },
    "Q04": {
        "gold": {"C4": "PRIMARY", "C6": "SUPPORTING", "C8": "SUPPORTING"},
        "desc": "How did Aegeon come to have two pairs of identical twin sons and twin servants?"
    },
    "Q05": {
        "gold": {"C12": "SUPPORTING", "C42": "PRIMARY", "C70": "PRIMARY", "C65": "PRIMARY"},
        "desc": "How do Antipholus of Syracuse and Antipholus of Ephesus differ in their initial interactions with Dromio of Syracuse?"
    },
    "Q06": {
        "gold": {"C105": "PRIMARY", "C110": "PRIMARY", "C107": "PRIMARY", "C147": "SUPPORTING", "C155": "SUPPORTING"},
        "desc": "Why is Antipholus of Ephesus arrested, and what role does Angelo the goldsmith play in the debt dispute?"
    }
}

def audit_question(q_id, method):
    gold_map = GOLD_ANNOTATIONS[q_id]["gold"]
    retrieved_items = raw_data[q_id][method]

    seen_parents = set()
    relevance_scores = []
    classifications = []
    gold_found = set()
    first_useful_rank = 0

    duplicate_count = 0

    for idx, item in enumerate(retrieved_items, 1):
        cid = item["chunk_id"]
        parent_key = item.get("parent_id") or item.get("parent_text")[:80]

        is_dup = parent_key in seen_parents
        if is_dup:
            duplicate_count += 1
            rel = 0
            cls_val = "DUPLICATE"
        else:
            seen_parents.add(parent_key)
            if cid in gold_map:
                gold_found.add(cid)
                role = gold_map[cid]
                if role == "PRIMARY":
                    rel = 3
                    cls_val = "PRIMARY"
                else:
                    rel = 2
                    cls_val = "SUPPORTING"
                if first_useful_rank == 0:
                    first_useful_rank = idx
            else:
                # Check keyword relevance
                text_lower = item["text"].lower()
                if any(kw in text_lower for kw in ["aegeon", "syracuse", "ephesus", "antipholus", "dromio", "adriana", "goldsmith", "angelo", "chain", "ducats", "twin"]):
                    rel = 1
                    cls_val = "RELEVANT_BUT_UNNECESSARY"
                else:
                    rel = 0
                    cls_val = "IRRELEVANT"

        relevance_scores.append(rel)
        classifications.append(cls_val)

    total_gold = len(gold_map)
    cids_retrieved = [item["chunk_id"] for item in retrieved_items]

    # Metrics computation
    hit_10 = 1.0 if len(gold_found) > 0 else 0.0

    recalls = {}
    ev_precisions = {}
    pri_precisions = {}

    for k in [1, 3, 5, 10]:
        cids_k = set(cids_retrieved[:k])
        hits_k = len(set(gold_map.keys()).intersection(cids_k))
        recalls[f"recall_{k}"] = hits_k / total_gold

        # Precision@K
        cls_k = classifications[:k]
        ev_useful = sum(1 for c in cls_k if c in ["PRIMARY", "SUPPORTING"])
        pri_useful = sum(1 for c in cls_k if c == "PRIMARY")

        ev_precisions[f"evidence_precision_{k}"] = ev_useful / k
        pri_precisions[f"primary_precision_{k}"] = pri_useful / k

    mrr = 1.0 / first_useful_rank if first_useful_rank > 0 else 0.0

    # DCG & IDCG
    dcg_10 = sum((2**rel - 1) / math.log2(rank + 2) for rank, rel in enumerate(relevance_scores[:10]))
    
    # Ideal relevance vector
    ideal_rels = []
    for cid, role in gold_map.items():
        ideal_rels.append(3 if role == "PRIMARY" else 2)
    ideal_rels.sort(reverse=True)
    ideal_rels = ideal_rels[:10]
    idcg_10 = sum((2**rel - 1) / math.log2(rank + 2) for rank, rel in enumerate(ideal_rels))
    ndcg_10 = dcg_10 / idcg_10 if idcg_10 > 0 else 0.0

    # Completeness check
    primary_golds = {cid for cid, role in gold_map.items() if role == "PRIMARY"}
    completeness = 1.0 if primary_golds.issubset(gold_found) else 0.0

    redundancy_ratio = duplicate_count / len(retrieved_items) if retrieved_items else 0.0

    return {
        "hit_10": hit_10,
        "recalls": recalls,
        "evidence_precisions": ev_precisions,
        "primary_precisions": pri_precisions,
        "mrr": mrr,
        "ndcg_10": ndcg_10,
        "completeness": completeness,
        "redundancy_ratio": redundancy_ratio,
        "gold_found": list(gold_found),
        "total_gold": total_gold,
        "classifications": classifications
    }

methods = ["semantic", "bm25", "rrf"]
audit_results = {m: {} for m in methods}

for m in methods:
    print(f"\n==========================================================")
    print(f" AUDIT METRICS FOR: {m.upper()}")
    print(f"==========================================================")
    
    macro_acc = {
        "hit_10": [], "recall_1": [], "recall_3": [], "recall_5": [], "recall_10": [],
        "ev_prec_1": [], "ev_prec_3": [], "ev_prec_5": [], "ev_prec_10": [],
        "pri_prec_1": [], "pri_prec_3": [], "pri_prec_5": [], "pri_prec_10": [],
        "mrr": [], "ndcg_10": [], "completeness": [], "redundancy": []
    }

    for q_id in sorted(GOLD_ANNOTATIONS.keys()):
        res = audit_question(q_id, m)
        audit_results[m][q_id] = res

        rec = res["recalls"]
        ev_p = res["evidence_precisions"]
        pri_p = res["primary_precisions"]

        macro_acc["hit_10"].append(res["hit_10"])
        macro_acc["recall_1"].append(rec["recall_1"])
        macro_acc["recall_3"].append(rec["recall_3"])
        macro_acc["recall_5"].append(rec["recall_5"])
        macro_acc["recall_10"].append(rec["recall_10"])

        macro_acc["ev_prec_1"].append(ev_p["evidence_precision_1"])
        macro_acc["ev_prec_3"].append(ev_p["evidence_precision_3"])
        macro_acc["ev_prec_5"].append(ev_p["evidence_precision_5"])
        macro_acc["ev_prec_10"].append(ev_p["evidence_precision_10"])

        macro_acc["pri_prec_1"].append(pri_p["primary_precision_1"])
        macro_acc["pri_prec_3"].append(pri_p["primary_precision_3"])
        macro_acc["pri_prec_5"].append(pri_p["primary_precision_5"])
        macro_acc["pri_prec_10"].append(pri_p["primary_precision_10"])

        macro_acc["mrr"].append(res["mrr"])
        macro_acc["ndcg_10"].append(res["ndcg_10"])
        macro_acc["completeness"].append(res["completeness"])
        macro_acc["redundancy"].append(res["redundancy_ratio"])

        print(f"[{q_id}] Hit: {res['hit_10']} | Recall@10: {rec['recall_10']:.1%} | EvPrec@5: {ev_p['evidence_precision_5']:.1%} | PriPrec@5: {pri_p['primary_precision_5']:.1%} | MRR: {res['mrr']:.3f} | Comp: {res['completeness']} | Redun: {res['redundancy_ratio']:.1%}")

    print(f"\n--- {m.upper()} MACRO AVERAGES ---")
    for k, vals in macro_acc.items():
        print(f"  {k:15s}: {np.mean(vals):.4f}")

with open('evaluation/audit/audited_metrics_summary.json', 'w', encoding='utf-8') as f:
    json.dump(audit_results, f, indent=2, ensure_ascii=False)
