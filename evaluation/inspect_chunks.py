import json
import sys

with open('evaluation/chunks_dump.json', 'r', encoding='utf-8') as f:
    chunks = json.load(f)

print(f"Total loaded chunks: {len(chunks)}")

def search(name, keywords):
    print(f"\n==========================================")
    print(f" SEARCH: {name} (Keywords: {keywords})")
    print(f"==========================================")
    matches = []
    for c in chunks:
        text = c['text']
        t_lower = text.lower()
        if all(kw.lower() in t_lower for kw in keywords):
            matches.append(c)
            text_clean = text[:200].replace('\n', ' ')
            print(f"   Text: {text_clean}...\n")
    return matches

search("Q01 - Direct: Aegeon identity", ["aegeon"])
search("Q02 - Causal: Aegeon death sentence & Ephesus law", ["syracuse"])
search("Q02 - Causal: Egeon condemned / ransom", ["ransom"])
search("Q02 - Causal: Egeon law Ephesus", ["law", "ephesus"])
search("Q03 - Multi-hop: Adriana husband mistaken", ["adriana", "husband"])
search("Q03 - Multi-hop: Syracuse Antipholus dinner", ["dinner", "adriana"])
search("Q04 - Relationship: Twins birth / two sons", ["two sons"])
search("Q04 - Relationship: Dromio birth / twin servants", ["bought"])
search("Q05 - Comparative: Antipholus Syracuse vs Ephesus Dromio", ["beat", "dromio"])
search("Q06 - Event-chain: Angelo goldsmith debt arrest", ["goldsmith"])
search("Q06 - Event-chain: Chain ducats arrest", ["ducats", "arrest"])
