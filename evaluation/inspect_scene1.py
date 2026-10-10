import json

with open('evaluation/chunks_dump.json', 'r', encoding='utf-8') as f:
    chunks = json.load(f)

print("=== Chunks C0 to C15 ===")
for c in chunks[:16]:
    print(f"\n--- [{c['chunk_id']}] (Index: {c['index']}, Page: {c['page']}) Header: {c['breadcrumb_path']} ---")
    print(c['text'])
