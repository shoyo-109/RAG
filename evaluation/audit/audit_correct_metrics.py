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

audited_final_metrics = {m: {} for m in methods}

for m in methods:
    acc = {
        "hit_10": [], "recall_1": [], "recall_3": [], "recall_5": [], "recall_10": [],
        "ev_p_1": [], "ev_p_3": [], "ev_p_5": [], "ev_p_10": [],
        "pri_p_1": [], "pri_p_3": [], "pri_p_5": [], "pri_p_10": [],
        "mrr": [], "ndcg_10": [], "completeness": [], "redundancy": []
    }

    print(f"\n==========================================================================")
    print(f" AUDITING RETRIEVAL METHOD: {m.upper()}")
    print(f"==========================================================================")

    for q_id, gold_map in GOLD.items():
        items = raw_data[q_id][m][:10]
        cids = [x["chunk_id"] for x in items]
        
        seen_chunks = set()
        seen_parents = set()
        classifications = []
        first_useful_rank = 0
        gold_hits_set = set()
        dup_count = 0

        for idx, item in enumerate(items, 1):
            cid = item["chunk_id"]
            parent_key = item.get("parent_id") or item.get("parent_text")[:80]

            # Rule 1: Is this chunk a Gold Evidence chunk?
            if cid in gold_map:
                role = gold_map[cid]
                classifications.append(role)
                gold_hits_set.add(cid)
                seen_parents.add(parent_key)
                seen_chunks.add(cid)
                if first_useful_rank == 0:
                    first_useful_rank = idx
            # Rule 2: Is this chunk a duplicate of a previously seen chunk or parent section?
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

        hit_10 = 1.0 if len(gold_hits_set) > 0 else 0.0

        r1 = len(gold_hits_set.intersection(set(cids[:1]))) / len(gold_map)
        r3 = len(gold_hits_set.intersection(set(cids[:3]))) / len(gold_map)
        r5 = len(gold_hits_set.intersection(set(cids[:5]))) / len(gold_map)
        r10 = len(gold_hits_set.intersection(set(cids[:10]))) / len(gold_map)

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
        comp = 1.0 if primary_golds.issubset(gold_hits_set) else 0.0

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

        print(f"[{q_id}] Hit@10: {hit_10} | Rec@10: {r10:.1%} | EvPrec@5: {ev5:.1%} | PriPrec@5: {p5:.1%} | MRR: {mrr:.3f} | nDCG@10: {ndcg10:.3f} | Comp: {comp} | Redun: {redundancy:.1%}")

    audited_final_metrics[m] = {k: float(np.mean(vals)) for k, vals in acc.items()}

print("\n==========================================================================")
print(" FINAL AUDITED MACRO SUMMARY (CORRECTED EVALUATION)")
print("==========================================================================")
print(json.dumps(audited_final_metrics, indent=2))
