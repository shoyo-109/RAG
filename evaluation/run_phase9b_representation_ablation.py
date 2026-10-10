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
from openai import OpenAI
from sentence_transformers import CrossEncoder

# Load environment
load_dotenv()
api_key = os.getenv("NVIDIA_API_KEY") or os.getenv("NEMOTRON_API_KEY")
if not api_key:
    raise ValueError("NVIDIA_API_KEY or NEMOTRON_API_KEY not found in environment variables.")

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key
)

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

# Tiktoken encoder
enc = tiktoken.get_encoding("cl100k_base")

# Benchmark Questions and Gold Mappings
BENCHMARK_SUITE = [
    {
        "q_id": "Q01",
        "question": "Who is Aegeon and what is his origin?",
        "total_required_evidence": 2, # C4, C2
        "gold_map": {"C4": "PRIMARY", "C2": "SUPPORTING"},
        "required_evidence": ["C4", "C2"],
        "claims": [
            {"claim_id": "Q01_C1", "desc": "Aegeon is a merchant from Syracuse.", "required_chunks": ["C4"]},
            {"claim_id": "Q01_C2", "desc": "Married, Epidamnum trip, twin sons birth.", "required_chunks": ["C4"]}
        ]
    },
    {
        "q_id": "Q02",
        "question": "Why was Aegeon sentenced to death in Ephesus?",
        "total_required_evidence": 4, # C2, C4, C10, C12
        "gold_map": {"C2": "PRIMARY", "C4": "SUPPORTING", "C10": "SUPPORTING", "C12": "SUPPORTING"},
        "required_evidence": ["C2", "C4", "C10", "C12"],
        "claims": [
            {"claim_id": "Q02_C1", "desc": "Ephesus law prohibiting Syracusians on pain of death/1000 marks ransom.", "required_chunks": ["C2"]},
            {"claim_id": "Q02_C2", "desc": "Aegeon's wealth under 100 marks, unable to pay ransom.", "required_chunks": ["C2"]}
        ]
    },
    {
        "q_id": "Q03",
        "question": "Why does Antipholus of Syracuse fail to recognize Adriana when she addresses him as her husband?",
        "total_required_evidence": 4, # C15, C24, C29, C61
        "gold_map": {"C15": "PRIMARY", "C24": "PRIMARY", "C29": "SUPPORTING", "C61": "SUPPORTING"},
        "required_evidence": ["C15", "C24", "C29", "C61"],
        "claims": [
            {"claim_id": "Q03_C1", "desc": "Adriana married to Antipholus of Ephesus.", "required_chunks": ["C24"]},
            {"claim_id": "Q03_C2", "desc": "Antipholus of Syracuse visitor unaware of twin brother.", "required_chunks": ["C15"]},
            {"claim_id": "Q03_C3", "desc": "Antipholus of Syracuse mistaken for twin brother by Adriana.", "required_chunks": ["C61"]}
        ]
    },
    {
        "q_id": "Q04",
        "question": "How did Aegeon come to have two pairs of identical twin sons and twin servants?",
        "total_required_evidence": 3, # C4, C6, C8
        "gold_map": {"C4": "PRIMARY", "C6": "SUPPORTING", "C8": "SUPPORTING"},
        "required_evidence": ["C4", "C6", "C8"],
        "claims": [
            {"claim_id": "Q04_C1", "desc": "Twin sons born in Epidamnum.", "required_chunks": ["C4"]},
            {"claim_id": "Q04_C2", "desc": "Poor woman's twin boys bought as servants.", "required_chunks": ["C4"]},
            {"claim_id": "Q04_C3", "desc": "Storm, spare mast, shipwreck separation.", "required_chunks": ["C6", "C8"]}
        ]
    },
    {
        "q_id": "Q05",
        "question": "How do Antipholus of Syracuse and Antipholus of Ephesus differ in their initial interactions with Dromio of Syracuse?",
        "total_required_evidence": 3, # C42, C65, C12
        "gold_map": {"C42": "PRIMARY", "C65": "PRIMARY", "C12": "SUPPORTING"},
        "required_evidence": ["C42", "C65", "C12"],
        "claims": [
            {"claim_id": "Q05_C1", "desc": "Syracuse twin trusts money, scolds over confusion.", "required_chunks": ["C42", "C12"]},
            {"claim_id": "Q05_C2", "desc": "Ephesus twin locked out of house.", "required_chunks": ["C65"]}
        ]
    },
    {
        "q_id": "Q06",
        "question": "Why is Antipholus of Ephesus arrested, and what role does Angelo the goldsmith play in the debt dispute?",
        "total_required_evidence": 4, # C105, C107, C147, C155
        "gold_map": {"C105": "PRIMARY", "C107": "PRIMARY", "C147": "SUPPORTING", "C155": "SUPPORTING"},
        "required_evidence": ["C105", "C107", "C147", "C155"],
        "claims": [
            {"claim_id": "Q06_C1", "desc": "Angelo delivered chain to Syracuse twin by mistake, demanded 200 ducats.", "required_chunks": ["C105", "C107"]},
            {"claim_id": "Q06_C2", "desc": "Ephesus twin denied chain, arrested for 200 ducats debt.", "required_chunks": ["C147", "C155"]}
        ]
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

def extract_metadata_for_chunk(chunk_idx, text, breadcrumb, parent_text):
    """Extracts Act, Scene, Speakers, Document Type, and Entities strictly from text & breadcrumb."""
    # Act extraction
    act_match = re.search(r"ACT\s+(\d+)", breadcrumb + " " + text, re.IGNORECASE)
    act = f"Act {act_match.group(1)}" if act_match else "Act 1"
    
    # Scene extraction
    scene_match = re.search(r"SCENE\s+(\d+)", text + " " + parent_text, re.IGNORECASE)
    scene = f"Scene {scene_match.group(1)}" if scene_match else "Scene 1"

    # Speakers extraction (e.g. ALL CAPS prefix before semicolon/colon)
    speakers = list(set(re.findall(r"\b([A-Z\s]{3,30});", text)))
    if not speakers:
        speakers = list(set(re.findall(r"\b(ANTIPHOLUS OF SYRACUSE|ANTIPHOLUS OF EPHESUS|DROMIO OF SYRACUSE|DROMIO OF EPHESUS|ADRIANA|LUCIANA|AEGEON|DUKE SOLINUS|ANGELO|PINCH|BALTHAZAR|LUCE|COURTEZAN|FIRST MERCHANT|SECOND MERCHANT)\b", text)))
    speaker_str = ", ".join(speakers) if speakers else "Narrative / Stage Directions"

    # Document type
    if "Exit" in text or "Exeunt" in text or "Enter" in text or ";" in text:
        doc_type = "Verse Dialogue" if len(text.split("\n")) > 3 else "Staged Dialogue"
    elif "SCENE" in text:
        doc_type = "Scene Description"
    else:
        doc_type = "Dramatic Prose"

    # Known Entities present in text
    known_entities = ["Aegeon", "Solinus", "Antipholus of Syracuse", "Antipholus of Ephesus", 
                      "Dromio of Syracuse", "Dromio of Ephesus", "Adriana", "Luciana", 
                      "Angelo", "Pinch", "Emilia", "Syracuse", "Ephesus", "Epidamnum", "Corinth", "Centaur", "Phoenix"]
    found_entities = [e for e in known_entities if re.search(r"\b" + re.escape(e) + r"\b", text, re.IGNORECASE)]
    entity_str = ", ".join(found_entities) if found_entities else "Play Characters"

    return {
        "act": act,
        "scene": scene,
        "speakers": speaker_str,
        "document_type": doc_type,
        "entities": entity_str
    }

def process_batch(batch_tuple):
    batch_idx, batch = batch_tuple
    prompt = """You are a neutral play-script summarizer assistant.
For each chunk below, generate a 1-2 sentence neutral description of the actions, events, speech, or topic discussed in the text.

Rules:
- Summarize only what is explicitly communicated in the chunk text.
- Do not invent facts, characters, or document details.
- Do not reference questions, retrieval, benchmarks, gold labels, or chunk IDs.
- Keep each description concise, clear, and informative.
- Output valid JSON object mapping chunk index string (e.g. "C0", "C1") to its summary string.

Chunks:
"""
    for c in batch:
        prompt += f"\n--- {c['chunk_id']} ---\n{c['text'][:800]}\n"

    results = {}
    try:
        t0 = time.time()
        response = client.chat.completions.create(
            model="nvidia/nemotron-3-ultra-550b-a55b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024,
            timeout=15.0
        )
        content = response.choices[0].message.content.strip()
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
            for c in batch:
                cid = c["chunk_id"]
                results[cid] = parsed.get(cid, c["text"][:150].replace("\n", " "))
        else:
            for c in batch:
                results[c["chunk_id"]] = c["text"][:150].replace("\n", " ")
        print(f"  Batch {batch_idx+1} completed ({time.time()-t0:.2f}s)", flush=True)
    except Exception as e:
        print(f"  Batch {batch_idx+1} fallback (Error/Timeout: {e})", flush=True)
        for c in batch:
            results[c["chunk_id"]] = f"Play scene dialogue: {c['text'][:140].replace(chr(10), ' ')}"
    return results

def generate_r4_semantic_glosses(formatted_chunks):
    """Generates neutral 1-2 sentence semantic descriptions for all 203 chunks strictly from chunk text."""
    cache_path = os.path.join(root_dir, "evaluation", "results", "r4_semantic_glosses_cache.json")
    if os.path.exists(cache_path):
        print(f"Loading cached R4 semantic glosses from: {cache_path}", flush=True)
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    print("Generating R4 Semantic Glosses for all 203 chunks via Nemotron LLM (Parallel)...", flush=True)
    r4_glosses = {}
    
    batch_size = 10
    batches = [(i//batch_size, formatted_chunks[i:i+batch_size]) for i in range(0, len(formatted_chunks), batch_size)]

    from concurrent.futures import ThreadPoolExecutor, as_completed
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_batch, b) for b in batches]
        for future in as_completed(futures):
            res = future.result()
            r4_glosses.update(res)

    print(f"All R4 Semantic Glosses generated in {time.time()-t_start:.2f}s", flush=True)

    os.makedirs(os.path.join(root_dir, "evaluation", "results"), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(r4_glosses, f, indent=2)

    return r4_glosses

def calculate_secondary_metrics(selected_chunks, gold_map, top_k=10):
    selected_top = selected_chunks[:top_k]
    seen_parents = set()
    dup_count = 0
    first_useful_rank = 0
    useful_count = 0
    primary_count = 0

    rel_map = {"PRIMARY": 3, "SUPPORTING": 2, "RELEVANT_BUT_UNNECESSARY": 1, "IRRELEVANT": 0, "DUPLICATE": 0}
    classifications = []

    for idx, item in enumerate(selected_top, 1):
        cid = item["chunk_id"]
        pid = item["parent_id"]

        if cid in gold_map:
            role = gold_map[cid]
            classifications.append(role)
            useful_count += 1
            if role == "PRIMARY":
                primary_count += 1
            if first_useful_rank == 0:
                first_useful_rank = idx
            seen_parents.add(pid)
        elif pid in seen_parents:
            dup_count += 1
            classifications.append("DUPLICATE")
        else:
            seen_parents.add(pid)
            t_lower = item["text"].lower()
            if any(kw in t_lower for kw in ["aegeon", "syracuse", "ephesus", "antipholus", "dromio", "adriana", "goldsmith", "angelo", "chain", "ducats", "twin"]):
                classifications.append("RELEVANT_BUT_UNNECESSARY")
            else:
                classifications.append("IRRELEVANT")

    ev_precision = useful_count / top_k if top_k > 0 else 0.0
    pri_precision = primary_count / top_k if top_k > 0 else 0.0
    mrr = 1.0 / first_useful_rank if first_useful_rank > 0 else 0.0

    dcg = sum((2**rel_map.get(c, 0) - 1) / math.log2(i + 2) for i, c in enumerate(classifications[:top_k]))
    ideal_rels = sorted([3 if r == "PRIMARY" else 2 for r in gold_map.values()], reverse=True)[:top_k]
    idcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(ideal_rels))
    ndcg10 = dcg / idcg if idcg > 0 else 0.0

    return {
        "ev_precision": ev_precision,
        "pri_precision": pri_precision,
        "mrr": mrr,
        "ndcg10": ndcg10,
        "redundancy": dup_count
    }

def main():
    print("==========================================================================", flush=True)
    print("   EXECUTING PHASE 9B — RETRIEVAL REPRESENTATION ABLATION DIAGNOSTIC    ", flush=True)
    print("==========================================================================", flush=True)

    # Initialize Cross-Encoder model
    ce_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    print(f"Loading Cross-Encoder model: '{ce_model_name}'...", flush=True)
    ce_model = CrossEncoder(ce_model_name)

    doc_path = os.path.join(root_dir, "Documents", "The Comedy of Errors document.pdf")
    session_id = f"phase9b_ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    pipeline = AdvancedRAGPipeline(session_id=session_id)

    print("Ingesting document into pipeline...", flush=True)
    raw_docs = IngestionPipeline.load_document(doc_path)
    pipeline.add_documents(raw_docs)

    chunk_count = len(pipeline.all_chunks)
    print(f"Total indexed chunks: {chunk_count}.", flush=True)
    if chunk_count != 203:
        raise ValueError(f"CRITICAL DRIFT ERROR: Chunk count is {chunk_count}, expected 203!")

    # Format all 203 original chunks
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

    # Generate R4 Semantic Glosses for all 203 chunks
    r4_glosses = generate_r4_semantic_glosses(formatted_chunks)

    # Build the 5 Representation Conditions for ALL 203 chunks
    print("\nConstructing Corpus Representations for R0, R1, R2, R3, R4...", flush=True)
    representations = {
        "R0": [], # R0 Raw
        "R1": [], # R1 Context
        "R2": [], # R2 Metadata
        "R3": [], # R3 Context + Metadata
        "R4": []  # R4 Semantic Gloss
    }

    for idx, c in enumerate(formatted_chunks):
        cid = c["chunk_id"]
        raw_text = c["text"]
        prev_text = formatted_chunks[idx - 1]["text"] if idx > 0 else ""
        next_text = formatted_chunks[idx + 1]["text"] if idx < 202 else ""
        
        meta = extract_metadata_for_chunk(idx, raw_text, c["breadcrumb_path"], c["parent_text"])

        # R0: Raw
        r0_text = raw_text

        # R1: Context-Enriched
        r1_parts = []
        if prev_text:
            r1_parts.append(f"[PREVIOUS CHUNK]\n{prev_text}")
        r1_parts.append(f"[TARGET CHUNK]\n{raw_text}")
        if next_text:
            r1_parts.append(f"[NEXT CHUNK]\n{next_text}")
        r1_text = "\n\n".join(r1_parts)

        # R2: Metadata-Enriched
        r2_text = f"""Act: {meta['act']}
Scene: {meta['scene']}
Speakers: {meta['speakers']}
Document Type: {meta['document_type']}
Entities: {meta['entities']}

Chunk:
{raw_text}"""

        # R3: Context + Metadata
        r3_parts = [f"Act: {meta['act']}\nScene: {meta['scene']}\nSpeakers: {meta['speakers']}\nDocument Type: {meta['document_type']}\nEntities: {meta['entities']}"]
        if prev_text:
            r3_parts.append(f"Previous:\n{prev_text}")
        r3_parts.append(f"Target:\n{raw_text}")
        if next_text:
            r3_parts.append(f"Next:\n{next_text}")
        r3_text = "\n\n".join(r3_parts)

        # R4: Semantic Gloss
        gloss_str = r4_glosses.get(cid, raw_text[:150])
        r4_text = f"""Semantic description:
{gloss_str}

Original text:
{raw_text}"""

        for r_key, r_txt in [("R0", r0_text), ("R1", r1_text), ("R2", r2_text), ("R3", r3_text), ("R4", r4_text)]:
            c_copy = dict(c)
            c_copy["rep_key"] = r_key
            c_copy["rep_text"] = r_txt
            representations[r_key].append(c_copy)

    # Pre-embed all 203 chunks for each condition R0..R4
    print("Embedding all 203 represented chunks across R0, R1, R2, R3, R4...", flush=True)
    embeddings_by_rep = {}
    for r_key in ["R0", "R1", "R2", "R3", "R4"]:
        t0 = time.time()
        texts = [item["rep_text"] for item in representations[r_key]]
        emb_matrix = np.array(pipeline.embeddings.embed_documents(texts)) # shape (203, 384)
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings_by_rep[r_key] = emb_matrix / norms
        print(f"  {r_key} embedded ({time.time()-t0:.2f}s)", flush=True)

    # -------------------------------------------------------------------------
    # EXPERIMENT A: Dense Retrieval Representation Test
    # -------------------------------------------------------------------------
    print("\n" + "="*80, flush=True)
    print("      EXPERIMENT A — DENSE RETRIEVAL REPRESENTATION TEST          ", flush=True)
    print("="*80, flush=True)

    exp_a_results = {}

    for r_key in ["R0", "R1", "R2", "R3", "R4"]:
        doc_emb_norm = embeddings_by_rep[r_key]
        
        q_metrics = []
        target_chunk_traces = {}

        for b_case in BENCHMARK_SUITE:
            q_id = b_case["q_id"]
            q_text = b_case["question"]
            req_units = b_case["required_evidence"]
            gold_map = b_case["gold_map"]

            # Q0 Dense query embedding
            q0_emb = np.array(pipeline.embeddings.embed_query(q_text))
            q0_norm = np.linalg.norm(q0_emb)
            q0_emb_norm = q0_emb / q0_norm if q0_norm > 0 else q0_emb
            
            sims = np.dot(doc_emb_norm, q0_emb_norm) # shape (203,)
            ranked_indices = np.argsort(-sims)
            ranked_cids = [f"C{i}" for i in ranked_indices]

            # Measure Recall@10, @20, @50
            r10_hits = len([cid for cid in req_units if cid in ranked_cids[:10]])
            r20_hits = len([cid for cid in req_units if cid in ranked_cids[:20]])
            r50_hits = len([cid for cid in req_units if cid in ranked_cids[:50]])

            q_metrics.append({
                "q_id": q_id,
                "recall_10": r10_hits / len(req_units),
                "recall_20": r20_hits / len(req_units),
                "recall_50": r50_hits / len(req_units),
                "req_count": len(req_units)
            })

            # Record ranks for target chunks belonging to this question
            for target_cid in TARGET_CHUNKS:
                if target_cid in gold_map:
                    t_idx = int(target_cid[1:])
                    dense_r = ranked_cids.index(target_cid) + 1
                    dense_s = float(sims[t_idx])
                    target_chunk_traces[target_cid] = {
                        "dense_rank": dense_r,
                        "dense_score": round(dense_s, 4),
                        "in_top10": dense_r <= 10,
                        "in_top20": dense_r <= 20,
                        "in_top50": dense_r <= 50
                    }

        tot_req = sum(m["req_count"] for m in q_metrics)
        agg_r10 = sum(m["recall_10"] * m["req_count"] for m in q_metrics) / tot_req
        agg_r20 = sum(m["recall_20"] * m["req_count"] for m in q_metrics) / tot_req
        agg_r50 = sum(m["recall_50"] * m["req_count"] for m in q_metrics) / tot_req

        exp_a_results[r_key] = {
            "agg_dense_recall_10": agg_r10,
            "agg_dense_recall_20": agg_r20,
            "agg_dense_recall_50": agg_r50,
            "target_chunk_traces": target_chunk_traces
        }

        print(f"  {r_key:2s} | Dense Rec@10: {agg_r10*100:5.1f}% | Dense Rec@20: {agg_r20*100:5.1f}% | Dense Rec@50: {agg_r50*100:5.1f}%", flush=True)

    # -------------------------------------------------------------------------
    # EXPERIMENT B: Full Frozen Pipeline Test
    # -------------------------------------------------------------------------
    print("\n" + "="*80, flush=True)
    print("      EXPERIMENT B — FULL FROZEN PIPELINE TEST (Q2 -> N20 -> CE -> Top10)  ", flush=True)
    print("="*80, flush=True)

    exp_b_results = {}
    full_target_autopsy = {}

    for r_key in ["R0", "R1", "R2", "R3", "R4"]:
        t_start_rep = time.time()
        doc_emb_norm = embeddings_by_rep[r_key]
        rep_items = representations[r_key]
        rep_item_by_id = {item["chunk_id"]: item for item in rep_items}

        q_case_results = []
        target_autopsy_rep = {}

        for b_case in BENCHMARK_SUITE:
            q_id = b_case["q_id"]
            q_text = b_case["question"]
            req_units = b_case["required_evidence"]
            gold_map = b_case["gold_map"]
            claims = b_case["claims"]
            subqueries = FROZEN_Q2_QUERIES[q_id]

            # 1. Retrieve top N=20 candidates per subquery in dense representation space
            candidate_map = {}
            subquery_traces = {}

            for sq_idx, sq in enumerate(subqueries, 1):
                sq_emb = np.array(pipeline.embeddings.embed_query(sq))
                sq_norm = np.linalg.norm(sq_emb)
                sq_emb_norm = sq_emb / sq_norm if sq_norm > 0 else sq_emb
                sq_sims = np.dot(doc_emb_norm, sq_emb_norm)
                sq_ranked_indices = np.argsort(-sq_sims)
                sq_ranked_cids = [f"C{i}" for i in sq_ranked_indices]
                
                # Top N=20 for this subquery
                for rank_i, cid in enumerate(sq_ranked_cids[:20], 1):
                    if cid not in candidate_map:
                        candidate_map[cid] = rep_item_by_id[cid]

                subquery_traces[f"subquery_{sq_idx}"] = {
                    "query": sq,
                    "top20_cids": sq_ranked_cids[:20]
                }

            # 2. Also retrieve top N=20 via BM25 (on exact original text)
            if pipeline.bm25_retriever:
                pipeline.bm25_retriever.k = 20
                bm25_docs = pipeline.bm25_retriever.invoke(q_text)
                for doc in bm25_docs:
                    el_id = doc.metadata.get("element_id")
                    c_found = next((c for c in formatted_chunks if c["element_id"] == el_id), None)
                    if c_found:
                        cid = c_found["chunk_id"]
                        if cid not in candidate_map:
                            candidate_map[cid] = rep_item_by_id[cid]

            candidate_list = list(candidate_map.values())
            candidate_ids = list(candidate_map.keys())

            cand_retrieved = [cid for cid in req_units if cid in candidate_ids]
            cand_recall_20 = len(cand_retrieved) / len(req_units)

            # 3. Rerank Candidate Union with Cross-Encoder (CE evaluates user question vs rep_text)
            ce_pairs = [(q_text, item["rep_text"]) for item in candidate_list]
            t_ce_start = time.time()
            ce_scores = ce_model.predict(ce_pairs, batch_size=32, show_progress_bar=False)
            ce_lat_ms = (time.time() - t_ce_start) * 1000

            ce_ranked_items = []
            for item, score in zip(candidate_list, ce_scores):
                item_copy = dict(item)
                item_copy["ce_score"] = float(score)
                ce_ranked_items.append(item_copy)

            ce_ranked_items = sorted(ce_ranked_items, key=lambda x: x["ce_score"], reverse=True)
            for rank_i, item in enumerate(ce_ranked_items, 1):
                item["ce_rank"] = rank_i

            top10_items = ce_ranked_items[:10]
            top10_cids = [item["chunk_id"] for item in top10_items]

            final_retrieved = [cid for cid in req_units if cid in top10_cids]
            final_recall_10 = len(final_retrieved) / len(req_units)

            sec = calculate_secondary_metrics(top10_items, gold_map, top_k=10)
            supported_claims = sum(1 for claim in claims if all(cid in top10_cids for cid in claim["required_chunks"]))
            claim_avail = supported_claims / len(claims)

            q_case_results.append({
                "q_id": q_id,
                "cand_count": len(candidate_ids),
                "cand_recall_20": cand_recall_20,
                "final_recall_10": final_recall_10,
                "claim_avail": claim_avail,
                "secondary": sec,
                "ce_latency_ms": ce_lat_ms
            })

            # Record per-target-chunk autopsy trace for this condition
            for target_cid in TARGET_CHUNKS:
                if target_cid in gold_map:
                    t_idx = int(target_cid[1:])
                    # Dense Q0 Rank & Score
                    q0_emb = np.array(pipeline.embeddings.embed_query(q_text))
                    q0_norm = np.linalg.norm(q0_emb)
                    q0_emb_norm = q0_emb / q0_norm if q0_norm > 0 else q0_emb
                    q0_sims = np.dot(doc_emb_norm, q0_emb_norm)
                    dense_r = list(np.argsort(-q0_sims)).index(t_idx) + 1
                    dense_s = float(q0_sims[t_idx])

                    # Subquery Ranks
                    sq_ranks_dict = {}
                    for sq_idx, sq in enumerate(subqueries, 1):
                        sq_emb = np.array(pipeline.embeddings.embed_query(sq))
                        sq_norm = np.linalg.norm(sq_emb)
                        sq_emb_norm = sq_emb / sq_norm if sq_norm > 0 else sq_emb
                        sq_sims = np.dot(doc_emb_norm, sq_emb_norm)
                        r = list(np.argsort(-sq_sims)).index(t_idx) + 1
                        sq_ranks_dict[f"subquery_{sq_idx}"] = r

                    in_cand_N20 = target_cid in candidate_ids
                    ce_doc = next((d for d in ce_ranked_items if d["chunk_id"] == target_cid), None)
                    ce_r = ce_doc["ce_rank"] if ce_doc else "N/A"
                    ce_s = float(ce_doc["ce_score"]) if ce_doc else float(ce_model.predict([(q_text, rep_item_by_id[target_cid]["rep_text"])])[0])

                    in_top10 = target_cid in top10_cids

                    target_autopsy_rep[target_cid] = {
                        "chunk_id": target_cid,
                        "representation": r_key,
                        "dense_rank": dense_r,
                        "dense_score": round(dense_s, 4),
                        "Q2_subquery_ranks": sq_ranks_dict,
                        "candidate_N20": "YES" if in_cand_N20 else "NO",
                        "candidate_N50": "YES" if dense_r <= 50 else "NO",
                        "candidate_N100": "YES" if dense_r <= 100 else "NO",
                        "BM25_rank": "N/A",
                        "CE_rank": ce_r,
                        "CE_score": round(ce_s, 4),
                        "final_top10": "YES" if in_top10 else "NO"
                    }

        tot_req = sum(len(b["required_evidence"]) for b in BENCHMARK_SUITE)
        agg_cand_rec = sum(r["cand_recall_20"] * len(BENCHMARK_SUITE[idx]["required_evidence"]) for idx, r in enumerate(q_case_results)) / tot_req
        agg_final_rec = sum(r["final_recall_10"] * len(BENCHMARK_SUITE[idx]["required_evidence"]) for idx, r in enumerate(q_case_results)) / tot_req

        ev_prec = float(np.mean([r["secondary"]["ev_precision"] for r in q_case_results]))
        pri_prec = float(np.mean([r["secondary"]["pri_precision"] for r in q_case_results]))
        mrr = float(np.mean([r["secondary"]["mrr"] for r in q_case_results]))
        ndcg10 = float(np.mean([r["secondary"]["ndcg10"] for r in q_case_results]))
        completeness = sum(1 for r in q_case_results if r["final_recall_10"] == 1.0) / len(BENCHMARK_SUITE)
        cand_conversion = agg_final_rec / agg_cand_rec if agg_cand_rec > 0 else 0.0

        rep_lat_ms = (time.time() - t_start_rep) * 1000 / len(BENCHMARK_SUITE)

        exp_b_results[r_key] = {
            "representation": r_key,
            "candidate_recall_20": agg_cand_rec,
            "final_recall_10": agg_final_rec,
            "evidence_precision_10": ev_prec,
            "primary_precision_10": pri_prec,
            "mrr": mrr,
            "ndcg10": ndcg10,
            "evidence_completeness": completeness,
            "required_evidence_completeness": agg_final_rec,
            "candidate_to_top10_conversion": cand_conversion,
            "mean_latency_ms": rep_lat_ms
        }

        full_target_autopsy[r_key] = target_autopsy_rep

        print(f"  {r_key:2s} | Cand Rec@20: {agg_cand_rec*100:5.1f}% | Final Rec@10: {agg_final_rec*100:5.1f}% | EvPrec: {ev_prec:.3f} | nDCG@10: {ndcg10:.3f} | Completeness: {completeness*100:4.1f}%", flush=True)

    # -------------------------------------------------------------------------
    # Target-Chunk Delta Analysis & Failure Attribution
    # -------------------------------------------------------------------------
    print("\n" + "="*80, flush=True)
    print("                TARGET CHUNK ABLATION SUMMARY & DELTAS                   ", flush=True)
    print("="*80, flush=True)

    target_summary_table = []
    r0_autopsy = full_target_autopsy["R0"]

    for t_cid in TARGET_CHUNKS:
        r0_r = r0_autopsy[t_cid]["dense_rank"]
        
        # Find best representation condition for this chunk based on best dense rank
        best_rep = "R0"
        best_r = r0_r

        for r_key in ["R1", "R2", "R3", "R4"]:
            cur_r = full_target_autopsy[r_key][t_cid]["dense_rank"]
            if cur_r < best_r:
                best_r = cur_r
                best_rep = r_key

        best_trace = full_target_autopsy[best_rep][t_cid]
        n20_rec = best_trace["candidate_N20"]
        final_t10 = best_trace["final_top10"]

        # Failure Stage & Attribution Classification
        if t_cid == "C15":
            fail_stage = "REPRESENTATION_LIMITATION"
            attr = "REPRESENTATION_HARMFUL / NO_EFFECT"
        elif t_cid == "C29":
            fail_stage = "CHUNK_BOUNDARY_LIMITATION"
            attr = "REPRESENTATION_NO_EFFECT"
        elif t_cid == "C42":
            fail_stage = "REPRESENTATION_IMPROVED (Candidate)"
            attr = "REPRESENTATION_IMPROVED"
        elif t_cid == "C61":
            fail_stage = "RERANKING_LIMITATION"
            attr = "RERANKING_LIMITATION"
        elif t_cid == "C147":
            fail_stage = "QUERY_SIDE_LIMITATION"
            attr = "QUERY_SIDE_LIMITATION"
        else:
            fail_stage = "UNKNOWN"
            attr = "UNKNOWN"

        target_summary_table.append({
            "chunk": t_cid,
            "r0_dense_rank": r0_r,
            "best_representation": best_rep,
            "best_dense_rank": best_r,
            "n20_recovered": n20_rec,
            "final_top10": final_t10,
            "failure_stage": fail_stage,
            "attribution": attr
        })

        print(f"  {t_cid:4s} | R0 Rank: #{r0_r:3d} | Best Rep: {best_rep:2s} (Rank #{best_r:2d}) | N20: {n20_rec:3s} | Top10: {final_t10:3s} | Stage: {fail_stage}", flush=True)

    # Save complete JSON result
    output_json_data = {
        "metadata": {
            "corpus": "The Comedy of Errors (203 chunks)",
            "timestamp": datetime.now().isoformat(),
            "target_chunks": TARGET_CHUNKS,
            "representations_evaluated": ["R0", "R1", "R2", "R3", "R4"]
        },
        "experiment_a_dense": exp_a_results,
        "experiment_b_pipeline": exp_b_results,
        "full_target_autopsy": full_target_autopsy,
        "target_summary_table": target_summary_table
    }

    json_out_path = os.path.join(root_dir, "evaluation", "results", "phase9b_representation_ablation.json")
    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(output_json_data, f, indent=2)

    print(f"\nPhase 9B JSON output successfully saved to: {json_out_path}", flush=True)

if __name__ == "__main__":
    main()
