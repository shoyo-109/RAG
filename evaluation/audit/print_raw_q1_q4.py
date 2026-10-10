import json

with open('evaluation/audit/raw_retrieval_dumps.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

for q_id in ['Q01', 'Q02', 'Q03', 'Q04']:
    q_data = data[q_id]
    print(f"\n=================================================================")
    print(f" {q_id}: {q_data['question']}")
    print(f"=================================================================")
    
    print("\n--- SEMANTIC SEARCH TOP 10 ---")
    for item in q_data['semantic']:
        print(f"  Rank {item['rank']}: [{item['chunk_id']}] (score: {item['score']:.4f}) Header: {item['breadcrumb'][:35]} | Snippet: {item['text'][:80]}")

    print("\n--- BM25 SEARCH TOP 10 ---")
    for item in q_data['bm25']:
        print(f"  Rank {item['rank']}: [{item['chunk_id']}] Header: {item['breadcrumb'][:35]} | Snippet: {item['text'][:80]}")

    print("\n--- RRF HYBRID SEARCH TOP 10 ---")
    for item in q_data['rrf']:
        print(f"  Rank {item['rank']}: [{item['chunk_id']}] Header: {item['breadcrumb'][:35]} | Snippet: {item['text'][:80]}")
