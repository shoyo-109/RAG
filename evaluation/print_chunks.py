import json
import sys

with open('evaluation/chunks_dump.json', 'r', encoding='utf-8') as f:
    chunks = json.load(f)

chunk_ids = sys.argv[1:] if len(sys.argv) > 1 else ['C2', 'C3', 'C4']

for c in chunks:
    if c['chunk_id'] in chunk_ids:
        print(f"\n==========================================")
        print(f"[{c['chunk_id']}] Index: {c['index']} | Page: {c['page']} | Header: {c['breadcrumb_path']}")
        print(f"==========================================")
        print(c['text'])
