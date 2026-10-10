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

# Canonical Benchmark Suite & Gold Mapping
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

UNREACHABLE_DIAGNOSTIC_CHUNKS = ["C15", "C29", "C42", "C61", "C147"]

def format_chunk_item(doc, pipeline_chunks):
    el_id = doc.metadata.get("element_id")
    c_idx = next((i for i, c in enumerate(pipeline_chunks) if c.metadata.get("element_id") == el_id), -1)
    chunk_label = f"C{c_idx}" if c_idx >= 0 else str(el_id)
    return {
        "chunk_id": chunk_label,
        "chunk_idx": c_idx,
        "element_id": el_id,
        "parent_id": doc.metadata.get("parent_id") or doc.metadata.get("parent_text", "")[:80],
        "parent_text": doc.metadata.get("parent_text", ""),
        "breadcrumb": doc.metadata.get("breadcrumb_path", ""),
        "text": doc.page_content
    }

def get_frozen_q2_outputs(q_id):
    """Returns exact frozen Q2 multi-queries from Phase 7A/7B/7C artifacts (NO regeneration)."""
    q2_frozen = {
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
    return q2_frozen[q_id]

# -----------------------------------------------------------------------------
# LLM Query Transformation Generators for 8A, 8B, 8C
# -----------------------------------------------------------------------------

def generate_8a_aspects(question):
    """Phase 8A: Semantic Aspect Decomposition (2 to 4 aspects)."""
    t0 = time.time()
    prompt = f"""You are a search-query decomposition assistant.

Given a user's question, identify 2 to 4 distinct information requirements that would help retrieve evidence needed to answer the question.

Rules:
- Each aspect must represent a distinct information need.
- Focus on entities, events, causes, conditions, relationships, actions, or consequences explicitly implied by the question.
- Do not answer the question.
- Do not invent document-specific facts.
- Do not assume facts that are not present in the question.
- Do not reference chunk IDs, gold evidence, retrieval results, or document metadata.
- Avoid producing simple paraphrases of the original question.
- Keep each aspect concise and retrieval-oriented.
- Output ONLY a JSON array of aspect strings.

Question:
{question}"""

    response = client.chat.completions.create(
        model="nvidia/nemotron-3-ultra-550b-a55b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        top_p=0.95,
        max_tokens=512
    )
    lat_ms = (time.time() - t0) * 1000
    content = response.choices[0].message.content.strip()

    match = re.search(r"\[\s*\".*?\"\s*(?:,\s*\".*?\"\s*)*\]", content, re.DOTALL)
    if not match:
        match = re.search(r"\[.*\]", content, re.DOTALL)

    if match:
        try:
            aspects = json.loads(match.group(0))
            if isinstance(aspects, list) and 2 <= len(aspects) <= 4:
                return aspects, lat_ms, content
        except Exception:
            pass

    # Fallback aspect extraction by line
    lines = [l.strip().lstrip("- 1234. \"").rstrip('"') for l in content.split("\n") if l.strip()]
    cleaned = [l for l in lines if len(l) > 5 and not l.startswith("```") and not l.startswith("{")]
    if 2 <= len(cleaned) <= 4:
        return cleaned, lat_ms, content
    elif len(cleaned) > 4:
        return cleaned[:4], lat_ms, content

    return [f"{question} background entity", f"{question} cause event"], lat_ms, content

def generate_8b_hyde(question):
    """Phase 8B: HyDE (Hypothetical Document Embedding)."""
    t0 = time.time()
    prompt = f"""You are a retrieval-query formulation assistant.

Write a short hypothetical passage that expresses the type of information that would be useful for answering the user's question.

Rules:
- Produce approximately 2–3 sentences.
- Represent the semantic concepts, entities, relationships, events, causes, or conditions implied by the question.
- Do not use quotations.
- Do not mention that the passage is hypothetical.
- Do not reference retrieval, embeddings, chunks, or this instruction.
- Do not access or assume knowledge of the actual document.
- Do not invent highly specific document facts that are not implied by the question.
- Do not intentionally reproduce an answer from a known source.
- Output ONLY the hypothetical passage.

Question:
{question}"""

    response = client.chat.completions.create(
        model="nvidia/nemotron-3-ultra-550b-a55b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        top_p=0.95,
        max_tokens=256
    )
    lat_ms = (time.time() - t0) * 1000
    passage = response.choices[0].message.content.strip()
    if passage.startswith('"') and passage.endswith('"'):
        passage = passage[1:-1].strip()
    return passage, lat_ms

def generate_8c_vocabulary_bridge(question):
    """Phase 8C: Dense-Lexical Vocabulary Bridging."""
    t0 = time.time()
    prompt = f"""You are a search vocabulary transformation assistant.

Given a user question, generate alternative terminology that could express the same concepts in a document.

Rules:
- Extract the important concepts from the question.
- Generate concise synonyms and alternative phrasings for those concepts.
- Include alternative terminology only when it is semantically justified by the question.
- Do not invent document-specific facts.
- Do not inspect or assume the wording of the source document.
- Do not answer the question.
- Do not reference chunk IDs, retrieval results, gold evidence, or metadata.
- Avoid generic unrelated keywords.
- Keep the vocabulary set compact.
- Output ONLY valid JSON:

{{
  "dense_query": "one concise semantic retrieval query",
  "lexical_keywords": ["term1", "term2", "term3"]
}}

Question:
{question}"""

    response = client.chat.completions.create(
        model="nvidia/nemotron-3-ultra-550b-a55b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        top_p=0.95,
        max_tokens=256
    )
    lat_ms = (time.time() - t0) * 1000
    content = response.choices[0].message.content.strip()

    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if "dense_query" in parsed and "lexical_keywords" in parsed:
                return parsed["dense_query"], parsed["lexical_keywords"], lat_ms, content
        except Exception:
            pass

    # Fallback
    dense_q = question
    keywords = [w for w in re.findall(r'\b[A-Za-z]{4,}\b', question) if w.lower() not in ["what", "when", "where", "which", "with", "from", "does", "have"]]
    return dense_q, keywords, lat_ms, content

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
    print("   EXECUTING PHASE 8A–8C — SEMANTIC CANDIDATE GENERATION DIAGNOSTIC     ", flush=True)
    print("==========================================================================", flush=True)

    # Initialize Cross-Encoder model
    ce_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    print(f"Loading Cross-Encoder model: '{ce_model_name}'...", flush=True)
    t_ce_load_start = time.time()
    ce_model = CrossEncoder(ce_model_name)
    ce_load_time_s = time.time() - t_ce_load_start
    ce_device = str(ce_model.model.device)
    print(f"Cross-Encoder loaded successfully on device: {ce_device} ({ce_load_time_s:.2f}s)\n", flush=True)

    doc_path = os.path.join(root_dir, "Documents", "The Comedy of Errors document.pdf")
    session_id = f"phase8_exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    pipeline = AdvancedRAGPipeline(session_id=session_id)

    print("Ingesting document into pipeline...", flush=True)
    raw_docs = IngestionPipeline.load_document(doc_path)
    pipeline.add_documents(raw_docs)
    
    # 203-Chunk Deterministic Integrity Check
    chunk_count = len(pipeline.all_chunks)
    print(f"Total indexed chunks: {chunk_count}.", flush=True)
    if chunk_count != 203:
        raise ValueError(f"CRITICAL DRIFT ERROR: Chunk count is {chunk_count}, expected exactly 203!")
    print("Chunk Integrity Check PASSED: 203 semantic chunks verified.\n", flush=True)

    # Generate Transformations for 8A, 8B, 8C
    print("Generating LLM Query Transformations (8A Aspects, 8B HyDE, 8C Vocabulary Bridges)...", flush=True)
    transformations_audit = {}
    
    for case in BENCHMARK_SUITE:
        q_id = case["q_id"]
        q_orig = case["question"]
        print(f"\nProcessing {q_id}: '{q_orig}'", flush=True)
        
        q2_frozen = get_frozen_q2_outputs(q_id)
        
        aspects_8a, lat_8a, raw_8a = generate_8a_aspects(q_orig)
        print(f"  8A Aspects ({len(aspects_8a)}): {aspects_8a}", flush=True)

        hyde_8b, lat_8b = generate_8b_hyde(q_orig)
        print(f"  8B HyDE Passage: '{hyde_8b[:100]}...'", flush=True)

        dense_q_8c, keywords_8c, lat_8c, raw_8c = generate_8c_vocabulary_bridge(q_orig)
        print(f"  8C Dense Query: '{dense_q_8c}' | Lexical Keywords: {keywords_8c}", flush=True)

        transformations_audit[q_id] = {
            "q_id": q_id,
            "original_question": q_orig,
            "q2_frozen": q2_frozen,
            "8a_aspects": aspects_8a,
            "8a_latency_ms": lat_8a,
            "8b_hyde_passage": hyde_8b,
            "8b_latency_ms": lat_8b,
            "8c_dense_query": dense_q_8c,
            "8c_lexical_keywords": keywords_8c,
            "8c_latency_ms": lat_8c
        }

    search_filter = Filter(must=[FieldCondition(key="metadata.session_id", match=MatchValue(value=pipeline.session_id))])

    # Pre-retrieve vector candidates up to N=100 for all query channels
    retrieved_cache = {}

    for case in BENCHMARK_SUITE:
        q_id = case["q_id"]
        trans = transformations_audit[q_id]
        q_orig = trans["original_question"]

        # Q2 Sub-query retrievals (k=100 each)
        q2_sem_100 = []
        for sq in trans["q2_frozen"]:
            sq_emb = np.array(pipeline.embeddings.embed_query(sq))
            try:
                res_sq = pipeline.vector_store.similarity_search_with_score_by_vector(sq_emb.tolist(), k=100, filter=search_filter)
            except Exception:
                res_sq = pipeline.vector_store.similarity_search_with_score_by_vector(sq_emb.tolist(), k=100)
                res_sq = [(doc, s) for doc, s in res_sq if doc.metadata.get("session_id") == pipeline.session_id]
            q2_sem_100.append([format_chunk_item(doc, pipeline.all_chunks) for doc, _ in res_sq])

        # BM25 Q0 retrieval (k=100)
        bm25_raw = pipeline.bm25_retriever.invoke(q_orig) if pipeline.bm25_retriever else []
        bm25_q0_100 = [format_chunk_item(doc, pipeline.all_chunks) for doc in bm25_raw if doc.metadata.get("session_id") == pipeline.session_id][:100]

        # 8A Aspect Retrievals (k=100 each)
        aspect_sem_100 = []
        for asp in trans["8a_aspects"]:
            asp_emb = np.array(pipeline.embeddings.embed_query(asp))
            try:
                res_asp = pipeline.vector_store.similarity_search_with_score_by_vector(asp_emb.tolist(), k=100, filter=search_filter)
            except Exception:
                res_asp = pipeline.vector_store.similarity_search_with_score_by_vector(asp_emb.tolist(), k=100)
                res_asp = [(doc, s) for doc, s in res_asp if doc.metadata.get("session_id") == pipeline.session_id]
            aspect_sem_100.append([format_chunk_item(doc, pipeline.all_chunks) for doc, _ in res_asp])

        # 8B HyDE Retrieval (k=100)
        hyde_emb = np.array(pipeline.embeddings.embed_query(trans["8b_hyde_passage"]))
        try:
            res_hyde = pipeline.vector_store.similarity_search_with_score_by_vector(hyde_emb.tolist(), k=100, filter=search_filter)
        except Exception:
            res_hyde = pipeline.vector_store.similarity_search_with_score_by_vector(hyde_emb.tolist(), k=100)
            res_hyde = [(doc, s) for doc, s in res_hyde if doc.metadata.get("session_id") == pipeline.session_id]
        hyde_sem_100 = [format_chunk_item(doc, pipeline.all_chunks) for doc, _ in res_hyde]

        # 8C Dense Bridge Retrieval (k=100)
        bridge_emb = np.array(pipeline.embeddings.embed_query(trans["8c_dense_query"]))
        try:
            res_bridge = pipeline.vector_store.similarity_search_with_score_by_vector(bridge_emb.tolist(), k=100, filter=search_filter)
        except Exception:
            res_bridge = pipeline.vector_store.similarity_search_with_score_by_vector(bridge_emb.tolist(), k=100)
            res_bridge = [(doc, s) for doc, s in res_bridge if doc.metadata.get("session_id") == pipeline.session_id]
        bridge_sem_100 = [format_chunk_item(doc, pipeline.all_chunks) for doc, _ in res_bridge]

        # 8C BM25 Lexical Keywords Retrieval (k=100)
        lex_query_str = " ".join(trans["8c_lexical_keywords"])
        bm25_lex_raw = pipeline.bm25_retriever.invoke(lex_query_str) if pipeline.bm25_retriever else []
        bm25_lex_100 = [format_chunk_item(doc, pipeline.all_chunks) for doc in bm25_lex_raw if doc.metadata.get("session_id") == pipeline.session_id][:100]

        retrieved_cache[q_id] = {
            "q2_sem_100": q2_sem_100,
            "bm25_q0_100": bm25_q0_100,
            "aspect_sem_100": aspect_sem_100,
            "hyde_sem_100": hyde_sem_100,
            "bridge_sem_100": bridge_sem_100,
            "bm25_lex_100": bm25_lex_100
        }

    # Evaluate Candidate Expansion Conditions across N=20 (Primary) and N=50 (Secondary)
    depths_to_test = [20, 50]
    master_experiments = {}

    for N in depths_to_test:
        print(f"\n=========================================================================", flush=True)
        print(f"                EVALUATING CANDIDATE POOL DEPTH N = {N}                 ", flush=True)
        print(f"=========================================================================", flush=True)

        conditions = [
            {"id": "Control", "name": "Control (Frozen Q2 + CE)", "type": "control"},
            {"id": "Phase_8A", "name": "8A Aspect Decomposition", "type": "aspect"},
            {"id": "Phase_8B", "name": "8B HyDE", "type": "hyde"},
            {"id": "Phase_8C", "name": "8C Vocabulary Bridging", "type": "vocab_bridge"}
        ]

        cond_results = {c["id"]: [] for c in conditions}
        unreachable_tracking = {c["id"]: [] for c in conditions}
        per_question_analysis = {c["id"]: [] for c in conditions}

        for case_idx, case in enumerate(BENCHMARK_SUITE):
            q_id = case["q_id"]
            q_orig = case["question"]
            req_units = case["required_evidence"]
            gold_map = case["gold_map"]
            claims = case["claims"]

            cache = retrieved_cache[q_id]
            trans = transformations_audit[q_id]

            # Construct Frozen Q2 Base Candidate Union at depth N
            q2_sub_N = [sub[:N] for sub in cache["q2_sem_100"]]
            bm25_q0_N = cache["bm25_q0_100"][:N]

            base_q2_map = {}
            for sub_list in q2_sub_N:
                for item in sub_list:
                    base_q2_map[item["chunk_id"]] = item
            for item in bm25_q0_N:
                base_q2_map[item["chunk_id"]] = item

            base_q2_candidates = list(base_q2_map.values())
            base_q2_cids = list(base_q2_map.keys())

            base_cand_retrieved = [cid for cid in req_units if cid in base_q2_cids]
            base_cand_recall = len(base_cand_retrieved) / len(req_units)

            # Evaluate each condition independently
            for cond in conditions:
                c_id = cond["id"]
                c_type = cond["type"]

                # Build candidate map with full provenance tracking
                candidate_map = {}
                provenance_map = {}

                # 1. First add base Q2 candidates
                for rank_i, item in enumerate(base_q2_candidates, 1):
                    cid = item["chunk_id"]
                    candidate_map[cid] = item
                    provenance_map[cid] = {
                        "chunk_id": cid,
                        "sources": [{"channel": "q2", "source_query": "Q2_subqueries", "source_rank": rank_i}],
                        "parent_id": item.get("parent_id"),
                        "in_base_q2": True,
                        "newly_recovered": False
                    }

                # 2. Add condition-specific channels
                gen_latency = 0.0

                if c_type == "control":
                    pass # Q2 only
                elif c_type == "aspect":
                    gen_latency = trans["8a_latency_ms"]
                    for asp_idx, asp_list in enumerate(cache["aspect_sem_100"]):
                        asp_str = trans["8a_aspects"][asp_idx]
                        for rank_i, item in enumerate(asp_list[:N], 1):
                            cid = item["chunk_id"]
                            if cid not in candidate_map:
                                candidate_map[cid] = item
                                provenance_map[cid] = {
                                    "chunk_id": cid,
                                    "sources": [],
                                    "parent_id": item.get("parent_id"),
                                    "in_base_q2": False,
                                    "newly_recovered": True
                                }
                            provenance_map[cid]["sources"].append({
                                "channel": "aspect",
                                "source_query": asp_str,
                                "source_rank": rank_i
                            })
                elif c_type == "hyde":
                    gen_latency = trans["8b_latency_ms"]
                    for rank_i, item in enumerate(cache["hyde_sem_100"][:N], 1):
                        cid = item["chunk_id"]
                        if cid not in candidate_map:
                            candidate_map[cid] = item
                            provenance_map[cid] = {
                                "chunk_id": cid,
                                "sources": [],
                                "parent_id": item.get("parent_id"),
                                "in_base_q2": False,
                                "newly_recovered": True
                            }
                        provenance_map[cid]["sources"].append({
                            "channel": "hyde",
                            "source_query": trans["8b_hyde_passage"][:60],
                            "source_rank": rank_i
                        })
                elif c_type == "vocab_bridge":
                    gen_latency = trans["8c_latency_ms"]
                    # Dense bridge channel
                    for rank_i, item in enumerate(cache["bridge_sem_100"][:N], 1):
                        cid = item["chunk_id"]
                        if cid not in candidate_map:
                            candidate_map[cid] = item
                            provenance_map[cid] = {
                                "chunk_id": cid,
                                "sources": [],
                                "parent_id": item.get("parent_id"),
                                "in_base_q2": False,
                                "newly_recovered": True
                            }
                        provenance_map[cid]["sources"].append({
                            "channel": "dense_bridge",
                            "source_query": trans["8c_dense_query"],
                            "source_rank": rank_i
                        })
                    # BM25 bridge channel
                    for rank_i, item in enumerate(cache["bm25_lex_100"][:N], 1):
                        cid = item["chunk_id"]
                        if cid not in candidate_map:
                            candidate_map[cid] = item
                            provenance_map[cid] = {
                                "chunk_id": cid,
                                "sources": [],
                                "parent_id": item.get("parent_id"),
                                "in_base_q2": False,
                                "newly_recovered": True
                            }
                        provenance_map[cid]["sources"].append({
                            "channel": "bm25_bridge",
                            "source_query": " ".join(trans["8c_lexical_keywords"]),
                            "source_rank": rank_i
                        })

                expanded_candidates = list(candidate_map.values())
                exp_cids = list(candidate_map.keys())

                exp_cand_retrieved = [cid for cid in req_units if cid in exp_cids]
                exp_cand_recall = len(exp_cand_retrieved) / len(req_units)

                new_cids = [cid for cid in exp_cids if cid not in base_q2_cids]
                new_gold_recovered = [cid for cid in req_units if cid in exp_cids and cid not in base_q2_cids]

                # Cross-Encoder Reranking
                ce_pairs = [(q_orig, doc["text"]) for doc in expanded_candidates]
                t_ce_start = time.time()
                ce_scores = ce_model.predict(ce_pairs, batch_size=32, show_progress_bar=False)
                ce_lat_ms = (time.time() - t_ce_start) * 1000

                ce_ranked_docs = []
                for doc, score in zip(expanded_candidates, ce_scores):
                    doc_copy = dict(doc)
                    doc_copy["ce_score"] = float(score)
                    ce_ranked_docs.append(doc_copy)

                ce_ranked_docs = sorted(ce_ranked_docs, key=lambda x: x["ce_score"], reverse=True)
                for ce_rank, doc in enumerate(ce_ranked_docs, 1):
                    doc["ce_rank"] = ce_rank

                top10_docs = ce_ranked_docs[:10]
                top10_cids = [item["chunk_id"] for item in top10_docs]

                top10_retrieved = [cid for cid in req_units if cid in top10_cids]
                final_recall_10 = len(top10_retrieved) / len(req_units)

                sec = calculate_secondary_metrics(top10_docs, gold_map, top_k=10)
                supported_claims = sum(1 for claim in claims if all(cid in top10_cids for cid in claim["required_chunks"]))
                claim_avail = supported_claims / len(claims)

                tot_lat_ms = gen_latency + 132.4 + ce_lat_ms # 132.4ms candidate retrieval

                cond_results[c_id].append({
                    "q_id": q_id,
                    "base_cand_count": len(base_q2_candidates),
                    "expanded_cand_count": len(expanded_candidates),
                    "new_cands_count": len(new_cids),
                    "base_cand_recall": base_cand_recall,
                    "exp_cand_recall": exp_cand_recall,
                    "new_gold_recovered_count": len(new_gold_recovered),
                    "new_gold_recovered": new_gold_recovered,
                    "final_recall_10": final_recall_10,
                    "claim_avail": claim_avail,
                    "secondary": sec,
                    "ce_pairs_count": len(ce_pairs),
                    "gen_latency_ms": gen_latency,
                    "ce_latency_ms": ce_lat_ms,
                    "tot_latency_ms": tot_lat_ms
                })

                # Track Unreachable Diagnostic Chunks (C15, C29, C42, C61, C147)
                for unreach_cid in UNREACHABLE_DIAGNOSTIC_CHUNKS:
                    if unreach_cid in req_units:
                        in_base = unreach_cid in base_q2_cids
                        in_exp = unreach_cid in exp_cids
                        ce_doc = next((d for d in ce_ranked_docs if d["chunk_id"] == unreach_cid), None)
                        ce_r = ce_doc["ce_rank"] if ce_doc else "—"
                        ce_s = round(ce_doc["ce_score"], 4) if ce_doc else "—"
                        in_t10 = unreach_cid in top10_cids

                        prov_info = provenance_map.get(unreach_cid, None)

                        unreachable_tracking[c_id].append({
                            "q_id": q_id,
                            "N": N,
                            "unreachable_chunk": unreach_cid,
                            "in_base_candidate_pool": in_base,
                            "in_expanded_candidate_pool": in_exp,
                            "recovered_by_expansion": not in_base and in_exp,
                            "provenance": prov_info,
                            "ce_score": ce_s,
                            "ce_rank": ce_r,
                            "in_top10": "YES" if in_t10 else "NO"
                        })

                # Per-Question Analysis Entry
                per_question_analysis[c_id].append({
                    "q_id": q_id,
                    "N": N,
                    "base_recall": base_cand_recall,
                    "exp_recall": exp_cand_recall,
                    "ce_top10_recall": final_recall_10,
                    "new_gold_recovered": new_gold_recovered,
                    "top10_cids": top10_cids
                })

        # Calculate Aggregate Metrics for Depth N
        tot_req_evidence = sum(len(c["required_evidence"]) for c in BENCHMARK_SUITE)
        comparison_table = []

        for cond in conditions:
            c_id = cond["id"]
            res_list = cond_results[c_id]

            agg_base_cand_recall = sum(r["base_cand_recall"] * len(BENCHMARK_SUITE[idx]["required_evidence"]) for idx, r in enumerate(res_list)) / tot_req_evidence
            agg_exp_cand_recall = sum(r["exp_cand_recall"] * len(BENCHMARK_SUITE[idx]["required_evidence"]) for idx, r in enumerate(res_list)) / tot_req_evidence
            
            agg_base_count = sum(r["base_cand_count"] for r in res_list)
            agg_exp_count = sum(r["expanded_cand_count"] for r in res_list)
            cand_exp_factor = agg_exp_count / agg_base_count if agg_base_count > 0 else 1.0

            agg_final_recall = sum(r["final_recall_10"] * len(BENCHMARK_SUITE[idx]["required_evidence"]) for idx, r in enumerate(res_list)) / tot_req_evidence
            
            ev_prec = float(np.mean([r["secondary"]["ev_precision"] for r in res_list]))
            pri_prec = float(np.mean([r["secondary"]["pri_precision"] for r in res_list]))
            mrr = float(np.mean([r["secondary"]["mrr"] for r in res_list]))
            ndcg10 = float(np.mean([r["secondary"]["ndcg10"] for r in res_list]))

            completeness = sum(1 for r in res_list if r["final_recall_10"] == 1.0) / len(BENCHMARK_SUITE)
            
            tot_new_gold = sum(r["new_gold_recovered_count"] for r in res_list)
            tot_new_cands = sum(r["new_cands_count"] for r in res_list)

            recovery_efficiency = tot_new_gold / tot_new_cands if tot_new_cands > 0 else 0.0
            added_cand_precision = recovery_efficiency # Metric alias
            candidate_conversion = agg_final_recall / agg_exp_cand_recall if agg_exp_cand_recall > 0 else 0.0

            comparison_table.append({
                "condition_id": c_id,
                "condition_name": cond["name"],
                "base_candidate_recall": agg_base_cand_recall,
                "expanded_candidate_recall": agg_exp_cand_recall,
                "candidate_expansion_factor": round(cand_exp_factor, 2),
                "final_recall_at_10": agg_final_recall,
                "evidence_precision": ev_prec,
                "primary_precision": pri_prec,
                "mrr": mrr,
                "ndcg10": ndcg10,
                "evidence_completeness": completeness,
                "required_evidence_completeness": agg_final_recall,
                "new_gold_recovered": tot_new_gold, # Reported explicitly as requested
                "total_new_candidates_added": tot_new_cands,
                "recovery_efficiency": recovery_efficiency,
                "added_candidate_precision": added_cand_precision,
                "candidate_to_top10_conversion": candidate_conversion,
                "mean_ce_pairs": float(np.mean([r["ce_pairs_count"] for r in res_list])),
                "mean_gen_latency_ms": float(np.mean([r["gen_latency_ms"] for r in res_list])),
                "mean_ce_latency_ms": float(np.mean([r["ce_latency_ms"] for r in res_list])),
                "mean_total_latency_ms": float(np.mean([r["tot_latency_ms"] for r in res_list]))
            })

        master_experiments[f"N_{N}"] = {
            "comparison_table": comparison_table,
            "unreachable_tracking": unreachable_tracking,
            "per_question_analysis": per_question_analysis
        }

        # Print Comparison Table for Depth N
        print(f"\nPhase 8 Master Comparison Table (N={N}):", flush=True)
        print("-" * 155, flush=True)
        print(f"{'Condition':28s} | {'BaseCandRec':11s} | {'ExpCandRec':10s} | {'ExpFactor':9s} | {'Rec@10':7s} | {'EvPrec':7s} | {'MRR':5s} | {'nDCG@10':7s} | {'Completeness':12s} | {'NewGold':7s} | {'RecovEff':8s}", flush=True)
        print("-" * 155, flush=True)
        for row in comparison_table:
            print(f"{row['condition_name']:28s} | {row['base_candidate_recall']:11.1%} | {row['expanded_candidate_recall']:10.1%} | {row['candidate_expansion_factor']:8.2f}x | {row['final_recall_at_10']:7.1%} | {row['evidence_precision']:7.3f} | {row['mrr']:5.3f} | {row['ndcg10']:7.3f} | {row['evidence_completeness']:12.1%} | {row['new_gold_recovered']:7d} | {row['recovery_efficiency']:8.2%}", flush=True)

    # Output full experiment JSON payload
    out_payload = {
        "timestamp": datetime.now().isoformat(),
        "model_metadata": {
            "cross_encoder_model": ce_model_name,
            "device": ce_device,
            "vector_embedding_model": "all-MiniLM-L6-v2",
            "unreachable_diagnostic_chunks": UNREACHABLE_DIAGNOSTIC_CHUNKS,
            "batch_size": 32
        },
        "transformations_audit": transformations_audit,
        "experiments": master_experiments
    }

    out_file = os.path.join(current_dir, "results", "phase8_diagnostic_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2, ensure_ascii=False)

    print(f"\n=========================================================================", flush=True)
    print(f"Phase 8 diagnostic execution complete. Saved results to:\n{out_file}", flush=True)
    print(f"=========================================================================\n", flush=True)

if __name__ == "__main__":
    main()
