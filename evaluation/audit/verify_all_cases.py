import json

with open('evaluation/audit/raw_retrieval_dumps.json', 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

GOLD = {
    "Q01": {"C4": "PRIMARY", "C2": "SUPPORTING"},
    "Q02": {"C2": "PRIMARY", "C4": "SUPPORTING", "C10": "SUPPORTING", "C12": "SUPPORTING"},
    "Q03": {"C15": "PRIMARY", "C24": "PRIMARY", "C29": "SUPPORTING", "C50": "SUPPORTING"},
    "Q04": {"C4": "PRIMARY", "C6": "SUPPORTING", "C8": "SUPPORTING"},
    "Q05": {"C42": "PRIMARY", "C65": "PRIMARY", "C70": "PRIMARY", "C12": "SUPPORTING"},
    "Q06": {"C105": "PRIMARY", "C107": "PRIMARY", "C110": "PRIMARY", "C147": "SUPPORTING", "C155": "SUPPORTING"}
}

methods = ["semantic", "bm25", "rrf"]

for q_id, gold_dict in GOLD.items():
    print(f"\n==========================================================================")
    print(f" {q_id}: {raw_data[q_id]['question']}")
    print(f" Gold Set: {gold_dict}")
    print(f"==========================================================================")

    for m in methods:
        ret = raw_data[q_id][m]
        ret_cids = [x['chunk_id'] for x in ret[:10]]
        
        gold_found = {}
        for rank, cid in enumerate(ret_cids, 1):
            if cid in gold_dict:
                gold_found[cid] = (gold_dict[cid], rank)
                
        print(f"  [{m.upper():8s}] Retrieved Top-10: {ret_cids}")
        print(f"            Gold Hits: {gold_found}")
