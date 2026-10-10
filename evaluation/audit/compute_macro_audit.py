import json
import math
import numpy as np

with open('evaluation/audit/raw_retrieval_dumps.json', 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

GOLD = {
    "Q01": {"C4": "PRIMARY", "C2": "SUPPORTING"},
    "Q02": {"C2": "PRIMARY", "C4": "SUPPORTING", "C10": "SUPPORTING", "C12": "SUPPORTING"},
    "Q03": {"C15": "PRIMARY", "C24": "PRIMARY", "C29": "SUPPORTING", "C50": "SUPPORTING"},
    "Q04": {"C4": "PRIMARY", "C6": "SUPPORTING", "C8": "SUPPORTING"},
    "Q05": {"C42": "PRIMARY", "C70": "PRIMARY", "C12": "SUPPORTING"},
    "Q06": {"C105": "PRIMARY", "C110": "PRIMARY", "C147": "SUPPORTING", "C155": "SUPPORTING"}
}

methods = ["semantic", "bm25", "rrf"]

per_question_rows = {m: [] for m in methods}
macro_results = {m: {} for m in methods}

for m in methods:
    acc = {
        "hit_10": [], "recall_1": [], "recall_3": [], "recall_5": [], "recall_10": [],
        "ev_p_1": [], "ev_p_3": [], "ev_p_5": [], "ev_p_10": [],
        "pri_p_1": [], "pri_p_3": [], "pri_p_5": [], "pri_p_10": [],
        "mrr": [], "ndcg_10": [], "completeness": [], "redundancy": []
    }

    for q_id, gold_map in GOLD.items():
        items = raw_data[q_id][m][:10]
        cids = [x["chunk_id"] for x in items]
        
        seen_parents = set()
        classifications = []
        first_useful = 0
        gold_hits = set()
        dup_count = 0

        for idx, item in enumerate(items, 1):
            cid = item["chunk_id"]
            parent_key = item.get("parent_id") or item.get("parent_text")[:80]

            if parent_key in seen_parents:
                dup_count += 1
                classifications.append("DUPLICATE")
            else:
                seen_parents.add(parent_key)
                if cid in gold_map:
                    gold_hits.add(cid)
                    role = gold_map[cid]
                    classifications.append(role)
                    if first_useful == 0:
                        first_useful = idx
                else:
                    t_lower = item["text"].lower()
                    if any(kw in t_lower for kw in ["aegeon", "syracuse", "ephesus", "antipholus", "dromio", "adriana", "goldsmith", "angelo", "chain", "ducats", "twin"]):
                        classifications.append("RELEVANT_BUT_UNNECESSARY")
                    else:
                        classifications.append("IRRELEVANT")

        hit_10 = 1.0 if len(gold_hits) > 0 else 0.0

        # Recalls
        r1 = len(gold_hits.intersection(set(cids[:1]))) / len(gold_map)
        r3 = len(gold_hits.intersection(set(cids[:3]))) / len(gold_map)
        r5 = len(gold_hits.intersection(set(cids[:5]))) / len(gold_map)
        r10 = len(gold_hits.intersection(set(cids[:10]))) / len(gold_map)

        # Evidence Precision
        ev1 = sum(1 for c in classifications[:1] if c in ["PRIMARY", "SUPPORTING"]) / 1
        ev3 = sum(1 for c in classifications[:3] if c in ["PRIMARY", "SUPPORTING"]) / 3
        ev5 = sum(1 for c in classifications[:5] if c in ["PRIMARY", "SUPPORTING"]) / 5
        ev10 = sum(1 for c in classifications[:10] if c in ["PRIMARY", "SUPPORTING"]) / 10

        # Primary Precision
        p1 = sum(1 for c in classifications[:1] if c == "PRIMARY") / 1
        p3 = sum(1 for c in classifications[:3] if c == "PRIMARY") / 3
        p5 = sum(1 for c in classifications[:5] if c == "PRIMARY") / 5
        p10 = sum(1 for c in classifications[:10] if c == "PRIMARY") / 10

        mrr = 1.0 / first_useful if first_useful > 0 else 0.0

        # nDCG@10
        rel_weights = {"PRIMARY": 3, "SUPPORTING": 2, "RELEVANT_BUT_UNNECESSARY": 1, "IRRELEVANT": 0, "DUPLICATE": 0}
        dcg = sum((2**rel_weights[c] - 1) / math.log2(i + 2) for i, c in enumerate(classifications[:10]))
        
        ideal_rels = sorted([3 if r == "PRIMARY" else 2 for r in gold_map.values()], reverse=True)[:10]
        idcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(ideal_rels))
        ndcg10 = dcg / idcg if idcg > 0 else 0.0

        # Completeness
        primary_golds = {cid for cid, r in gold_map.items() if r == "PRIMARY"}
        comp = 1.0 if primary_golds.issubset(gold_hits) else 0.0

        redundancy = dup_count / len(items) if items else 0.0

        acc["hit_10"].append(hit_10)
        acc["recall_1"].append(r1)
        acc["recall_3"].append(r3)
        acc["recall_5"].append(r5)
        acc["recall_10"].append(r10)

        acc["ev_p_1"].append(ev1)
        acc["ev_p_3"].append(ev3)
        acc["ev_p_5"].append(ev5)
        acc["ev_p_10"].append(ev10)

        acc["pri_p_1"].append(p1)
        acc["pri_p_3"].append(p3)
        acc["pri_p_5"].append(p5)
        acc["pri_p_10"].append(p10)

        acc["mrr"].append(mrr)
        acc["ndcg_10"].append(ndcg10)
        acc["completeness"].append(comp)
        acc["redundancy"].append(redundancy)

        per_question_rows[m].append({
            "q_id": q_id, "hit_10": hit_10, "r5": r5, "r10": r10,
            "ev5": ev5, "ev10": ev10, "pri10": p10, "mrr": mrr,
            "comp": comp, "red": redundancy
        })

    macro_results[m] = {k: float(np.mean(vals)) for k, vals in acc.items()}

print("=== FINAL AUDITED MACRO SUMMARY ===")
print(json.dumps(macro_results, indent=2))
