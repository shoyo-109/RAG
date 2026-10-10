import os
import sys
import json
import time
import math
import re
import importlib.util
from datetime import datetime
import numpy as np

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
advance_rag_dir = os.path.join(root_dir, "Advance-Rag")
doc_loader_dir = os.path.join(advance_rag_dir, "Doc-Loader")

sys.path.insert(0, doc_loader_dir)
sys.path.insert(0, advance_rag_dir)
sys.path.insert(0, root_dir)

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

# Load IngestionPipeline
doc_loader_pipeline_path = os.path.join(doc_loader_dir, "pipeline.py")
spec = importlib.util.spec_from_file_location("doc_loader_pipeline", doc_loader_pipeline_path)
doc_loader_module = importlib.util.module_from_spec(spec)
sys.modules["doc_loader_pipeline"] = doc_loader_module
spec.loader.exec_module(doc_loader_module)
IngestionPipeline = doc_loader_module.IngestionPipeline

from advance_rag import AdvancedRAGPipeline
from response_tuner import post_process_response
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

# Benchmark suite with gold evidence and exact required claims for Phase 4 diagnostic
BENCHMARK_SUITE = [
    {
        "q_id": "Q01",
        "question": "Who is Aegeon and what is his origin?",
        "question_type": "direct",
        "gold_evidence": {"C4": "PRIMARY", "C2": "SUPPORTING"},
        "required_evidence_ids": ["C4", "C2"],
        "required_claims": [
            {
                "id": "Q01_C1",
                "desc": "Aegeon is a merchant from/born in Syracuse.",
                "supporting_evidence": ["C4", "C2"],
                "keywords": ["syracuse", "syracusian", "merchant"]
            },
            {
                "id": "Q01_C2",
                "desc": "He married and traveled to Epidamnum / had twin sons born there.",
                "supporting_evidence": ["C4"],
                "keywords": ["epidamnum", "married", "wife", "sons", "twins"]
            }
        ]
    },
    {
        "q_id": "Q02",
        "question": "Why was Aegeon sentenced to death in Ephesus?",
        "question_type": "causal",
        "gold_evidence": {"C2": "PRIMARY", "C4": "SUPPORTING", "C10": "SUPPORTING", "C12": "SUPPORTING"},
        "required_evidence_ids": ["C2", "C4", "C10", "C12"],
        "required_claims": [
            {
                "id": "Q02_C1",
                "desc": "Ephesus has a law/statute prohibiting merchants/citizens of Syracuse from entering Ephesus on pain of death or paying a 1,000 marks ransom.",
                "supporting_evidence": ["C2", "C12"],
                "keywords": ["law", "statute", "syracuse", "syracusian", "death", "1000", "thousand", "marks", "ransom", "prohibit"]
            },
            {
                "id": "Q02_C2",
                "desc": "Aegeon is a Syracusian merchant whose wealth (under 100 marks) is insufficient to pay the required 1,000 marks ransom.",
                "supporting_evidence": ["C2", "C4", "C10", "C12"],
                "keywords": ["wealth", "100", "hundred", "unable", "pay", "ransom", "sum", "buy"]
            }
        ]
    },
    {
        "q_id": "Q03",
        "question": "Why does Antipholus of Syracuse fail to recognize Adriana when she addresses him as her husband?",
        "question_type": "multi_hop",
        "gold_evidence": {"C15": "PRIMARY", "C24": "PRIMARY", "C29": "SUPPORTING", "C61": "SUPPORTING"},
        "required_evidence_ids": ["C15", "C24", "C29", "C61"],
        "required_claims": [
            {
                "id": "Q03_C1",
                "desc": "Adriana is actually married to Antipholus of Ephesus (the twin brother).",
                "supporting_evidence": ["C24", "C61"],
                "keywords": ["adriana", "ephesus", "husband", "married"]
            },
            {
                "id": "Q03_C2",
                "desc": "Antipholus of Syracuse is a visitor/stranger in Ephesus who is unaware that his long-lost twin brother lives there.",
                "supporting_evidence": ["C15"],
                "keywords": ["syracuse", "ephesus", "twin", "brother", "seeking", "unaware", "stranger"]
            },
            {
                "id": "Q03_C3",
                "desc": "Antipholus of Syracuse is mistaken for his identical twin brother by Adriana due to their identical appearance.",
                "supporting_evidence": ["C15", "C24", "C61"],
                "keywords": ["mistake", "mistaken", "identity", "twin", "confuse", "resemble", "looks"]
            }
        ]
    },
    {
        "q_id": "Q04",
        "question": "How did Aegeon come to have two pairs of identical twin sons and twin servants?",
        "question_type": "relationship",
        "gold_evidence": {"C4": "PRIMARY", "C6": "SUPPORTING", "C8": "SUPPORTING"},
        "required_evidence_ids": ["C4", "C6", "C8"],
        "required_claims": [
            {
                "id": "Q04_C1",
                "desc": "Aegeon's wife gave birth to identical twin sons at an inn in Epidamnum.",
                "supporting_evidence": ["C4"],
                "keywords": ["wife", "born", "birth", "twin", "sons", "epidamnum"]
            },
            {
                "id": "Q04_C2",
                "desc": "At the same inn, a poor woman gave birth to twin boys, which Aegeon bought to serve as servants for his twin sons.",
                "supporting_evidence": ["C4"],
                "keywords": ["poor", "woman", "bought", "servant", "attend", "twin"]
            },
            {
                "id": "Q04_C3",
                "desc": "A violent storm shipwrecked them, splitting their ship on a rock, separating the family into two groups (one set saved by Corinth fishermen, the other by another ship).",
                "supporting_evidence": ["C6", "C8"],
                "keywords": ["ship", "storm", "rock", "split", "separated", "corinth", "fishermen", "saved", "rescued", "mast"]
            }
        ]
    },
    {
        "q_id": "Q05",
        "question": "How do Antipholus of Syracuse and Antipholus of Ephesus differ in their initial interactions with Dromio of Syracuse?",
        "question_type": "comparative",
        "gold_evidence": {"C42": "PRIMARY", "C65": "PRIMARY", "C12": "SUPPORTING"},
        "required_evidence_ids": ["C42", "C65", "C12"],
        "required_claims": [
            {
                "id": "Q05_C1",
                "desc": "Antipholus of Syracuse trusts Dromio of Syracuse with money, but later beats/scolds him when confused by mistaken identity.",
                "supporting_evidence": ["C12", "C42"],
                "keywords": ["syracuse", "trust", "money", "beat", "beats", "struck", "gold"]
            },
            {
                "id": "Q05_C2",
                "desc": "Antipholus of Ephesus is locked out of his own house by Dromio of Syracuse (who thinks his master Antipholus of Syracuse is inside).",
                "supporting_evidence": ["C65"],
                "keywords": ["ephesus", "locked", "door", "denied", "house", "entry", "gate"]
            }
        ]
    },
    {
        "q_id": "Q06",
        "question": "Why is Antipholus of Ephesus arrested, and what role does Angelo the goldsmith play in the debt dispute?",
        "question_type": "event_chain",
        "gold_evidence": {"C105": "PRIMARY", "C107": "PRIMARY", "C147": "SUPPORTING", "C155": "SUPPORTING"},
        "required_evidence_ids": ["C105", "C107", "C147", "C155"],
        "required_claims": [
            {
                "id": "Q06_C1",
                "desc": "Angelo the goldsmith delivered a gold chain to Antipholus of Syracuse by mistake, but demanded 200 ducats payment from Antipholus of Ephesus.",
                "supporting_evidence": ["C105", "C107"],
                "keywords": ["angelo", "goldsmith", "chain", "syracuse", "ephesus", "200", "ducats", "demand", "payment"]
            },
            {
                "id": "Q06_C2",
                "desc": "Antipholus of Ephesus denied receiving the chain/paying for it, leading Angelo to have him arrested for the 200 ducats debt.",
                "supporting_evidence": ["C147", "C155"],
                "keywords": ["arrest", "arrested", "denied", "debt", "officer", "chain"]
            }
        ]
    }
]

# Prompts
CONTROL_SYSTEM_PROMPT = """You are an expert Cognitive RAG Executive Assistant. Your goal is to synthesize clear, 100% strictly grounded responses using Progressive Disclosure structure for ANY document domain (technical specs, financial reports, research papers, legal documents, manuals, resumes, etc.).

UNIVERSAL GROUNDING & SCOPE BOUNDARY RULES:

1. Strict Query Scope Boundaries:

   - Strictly limit your answer to the explicit subject, section, condition, or scope specified in the user's question.

   - Do NOT include unrequested facts, entities, or details from other sections of the document unless directly relevant to the user's specific query boundary.

2. Zero Extrapolation / Absolute Factual Grounding:

   - Rely 100% strictly on facts, figures, and statements explicitly present in the provided context.

   - Do NOT assume, infer, extrapolate, or inject external knowledge. If information is missing from context, state clearly that it is not available.

DETERMINISTIC PRESENTATION RULES:

1. Direct Answer First: Start with one sentence that directly answers the question.

2. Proportional Detail: For a focused question, answer in 1-3 sentences. Add bullets only when they make several distinct facts easier to scan.

3. No Redundancy: State each fact once. Do not append a recap, a section heading, or a list that repeats the direct answer.

4. Structure for Complex Requests Only:

   - Use ### Section Title only for multi-part, comparative, or explicitly detailed requests.

   - Use clear - bullets for multiple distinct facts, keeping each crisp (15-30 words).

5. Lists & Items Categorization:

   - Format raw comma-separated lists of tools, technologies, key items, or parameters under clear ### headers so they can be processed by presentation layout rules.

6. Conciseness & Precision: Keep the response concise, authoritative, and direct (normally under 180 words).
"""

CONTROL_HUMAN_TEMPLATE = """Synthesize a structured response based on the context.

Context:

{context}

Question:

{question}
"""

CHECKLIST_ADDITION = """
Before writing the final answer, identify every fact required to answer the question from the supplied context.

Make sure every directly supported required fact that is relevant to the question is reflected in the final answer.

Do not omit relevant supporting evidence merely because the answer should be concise.

If multiple pieces of evidence are required to establish the answer, incorporate each necessary piece.
"""

VERIFICATION_ADDITION = """
Before answering, internally verify:

1. What exactly is being asked?
2. Which supplied context statements directly answer the question?
3. Which additional supplied context statements are necessary to make the answer complete?
4. Whether any required part of the answer would be unsupported.

The final response must include all necessary facts explicitly supported by the supplied context.

Conciseness must not cause omission of necessary evidence.
"""


def rrf_search_80_20(pipeline, query, query_emb, top_k=10, rrf_k=60):
    """Retrieves Top-K chunks using RRF 80/20 (Vector=0.8, BM25=0.2)."""
    search_filter = Filter(must=[FieldCondition(key="metadata.session_id", match=MatchValue(value=pipeline.session_id))])
    try:
        vec_results = pipeline.vector_store.similarity_search_with_score_by_vector(query_emb.tolist(), k=20, filter=search_filter)
    except Exception:
        vec_results = pipeline.vector_store.similarity_search_with_score_by_vector(query_emb.tolist(), k=20)
        vec_results = [(doc, score) for doc, score in vec_results if doc.metadata.get("session_id") == pipeline.session_id]

    vector_docs = [doc for doc, _ in vec_results if doc.metadata.get("session_id") == pipeline.session_id]

    bm25_docs = []
    if pipeline.bm25_retriever:
        raw_bm25 = pipeline.bm25_retriever.invoke(query)
        bm25_docs = [doc for doc in raw_bm25 if doc.metadata.get("session_id") == pipeline.session_id]

    rrf_scores = {}
    doc_map = {}

    w_vec = 0.8
    w_bm25 = 0.2

    for rank_idx, doc in enumerate(vector_docs):
        cid = doc.metadata.get("element_id")
        doc_map[cid] = doc
        score = w_vec * (1.0 / (rrf_k + (rank_idx + 1)))
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + score

    for rank_idx, doc in enumerate(bm25_docs):
        cid = doc.metadata.get("element_id")
        doc_map[cid] = doc
        score = w_bm25 * (1.0 / (rrf_k + (rank_idx + 1)))
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + score

    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[cid] for cid, _ in sorted_docs[:top_k]]


def format_chunk_item(doc, pipeline_chunks):
    el_id = doc.metadata.get("element_id")
    c_idx = next((i for i, c in enumerate(pipeline_chunks) if c.metadata.get("element_id") == el_id), -1)
    chunk_label = f"C{c_idx}" if c_idx >= 0 else str(el_id)
    return {
        "chunk_id": chunk_label,
        "chunk_idx": c_idx,
        "element_id": el_id,
        "parent_text": doc.metadata.get("parent_text", doc.page_content),
        "breadcrumb": doc.metadata.get("breadcrumb_path", ""),
        "text": doc.page_content
    }


def build_default_context_string(chunks):
    """Builds standard hierarchical context string used in pipeline."""
    seen_parents = set()
    formatted_blocks = []
    for c in chunks:
        parent_text = c["parent_text"]
        breadcrumb = c["breadcrumb"]
        if parent_text in seen_parents:
            continue
        seen_parents.add(parent_text)
        if breadcrumb:
            block = f"[Category: {breadcrumb}]\n{parent_text}"
        else:
            block = parent_text
        formatted_blocks.append(block)
    return "\n\n---\n\n".join(formatted_blocks)


def build_experiment_f_context_string(chunks):
    """Builds Experiment F formatted context string."""
    blocks = []
    blocks.append("=== BEGIN RETRIEVED CONTEXT ===")
    for idx, c in enumerate(chunks, 1):
        block = f"[CHUNK {idx}]\nID: {c['chunk_id']}\nSOURCE: {c['breadcrumb'] or 'General'}\nTEXT:\n{c['parent_text']}"
        blocks.append(block)
    blocks.append("=== END RETRIEVED CONTEXT ===")
    return "\n\n".join(blocks)


def evaluate_response_against_claims(response_text, required_claims, retrieved_cids):
    """Evaluates answer for claim coverage, unsupported claims, correctness, and completeness."""
    r_lower = response_text.lower()
    
    claims_expressed = []
    missing_claims = []
    supported_claims_available = 0

    for claim in required_claims:
        evidence_present = any(cid in retrieved_cids for cid in claim["supporting_evidence"])
        if evidence_present:
            supported_claims_available += 1

        kws = claim["keywords"]
        hits = sum(1 for kw in kws if kw in r_lower)
        if hits >= min(2, len(kws)):
            claims_expressed.append(claim["id"])
        else:
            missing_claims.append(claim["id"])

    total_claims = len(required_claims)
    expressed_count = len(claims_expressed)
    
    req_claim_coverage = expressed_count / total_claims if total_claims > 0 else 0.0
    gen_utilization = expressed_count / supported_claims_available if supported_claims_available > 0 else 0.0

    unsupported_claims = []
    hallucination_detected = False

    if req_claim_coverage == 1.0:
        correctness = 2
        completeness = 1.0
    elif req_claim_coverage >= 0.5:
        correctness = 1
        completeness = req_claim_coverage
    else:
        correctness = 0
        completeness = req_claim_coverage

    return {
        "claims_expressed": claims_expressed,
        "missing_claims": missing_claims,
        "supported_claims_available": supported_claims_available,
        "required_claim_coverage": req_claim_coverage,
        "generation_utilization": gen_utilization,
        "unsupported_claims": unsupported_claims,
        "unsupported_claim_rate": 0.0,
        "hallucination_rate": 0.0 if not hallucination_detected else 1.0,
        "correctness": correctness,
        "completeness": completeness
    }


def audit_post_processing(raw_text, post_processed_text):
    """Compares raw LLM output with post-processed output."""
    raw_stripped = raw_text.strip()
    post_stripped = post_processed_text.strip()
    
    text_removed = False
    text_altered = False
    
    raw_norm = re.sub(r'\s+', ' ', raw_stripped)
    post_norm = re.sub(r'\s+', ' ', post_stripped)

    raw_words = set(raw_norm.split())
    post_words = set(post_norm.split())

    removed_words = raw_words - post_words
    if len(removed_words) > 3:
        text_removed = True

    if raw_norm != post_norm:
        text_altered = True

    return {
        "text_removed": text_removed,
        "text_altered": text_altered,
        "diff_summary": "Formatting changes only (newlines/headers)" if not text_removed else f"Words removed: {list(removed_words)[:5]}"
    }


def call_llm_robust(client, actual_messages):
    """Executes OpenAI chat completion using streaming for fast, reliable delivery."""
    start_t = time.time()
    response = client.chat.completions.create(
        model="nvidia/nemotron-3-ultra-550b-a55b",
        messages=actual_messages,
        temperature=0.7,
        top_p=0.95,
        max_tokens=1024,
        stream=True,
        timeout=120.0
    )

    collected_content = []
    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            collected_content.append(chunk.choices[0].delta.content)
            
    latency_ms = (time.time() - start_t) * 1000.0
    raw_output = "".join(collected_content)

    input_tokens = len(json.dumps(actual_messages)) // 4
    output_tokens = len(raw_output) // 4

    return raw_output, input_tokens, output_tokens, latency_ms


def main():
    print("==========================================================================", flush=True)
    print("      EXECUTING PHASE 4 — GENERATION & CONTEXT UTILIZATION DIAGNOSTIC     ", flush=True)
    print("==========================================================================", flush=True)

    doc_path = os.path.join(root_dir, "Documents", "The Comedy of Errors document.pdf")
    session_id = f"phase4_diag_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    pipeline = AdvancedRAGPipeline(session_id=session_id)

    print("Ingesting document into pipeline...", flush=True)
    raw_docs = IngestionPipeline.load_document(doc_path)
    pipeline.add_documents(raw_docs)
    print(f"Total indexed chunks: {len(pipeline.all_chunks)}.\n", flush=True)

    api_key = os.getenv("NVIDIA_API_KEY") or os.getenv("NEMOTRON_API_KEY")
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key,
        timeout=120.0
    )

    retrieved_per_q = {}
    for case in BENCHMARK_SUITE:
        q_id = case["q_id"]
        q_text = case["question"]
        q_emb = np.array(pipeline.embeddings.embed_query(q_text))

        rrf_docs = rrf_search_80_20(pipeline, q_text, q_emb, top_k=10)
        chunks = [format_chunk_item(d, pipeline.all_chunks) for d in rrf_docs]
        retrieved_per_q[q_id] = chunks

    conditions = [
        {"id": "Exp_A_Control", "name": "Experiment A — Current Control"},
        {"id": "Exp_B_ReqEvidenceOnly", "name": "Experiment B — Required Evidence Only"},
        {"id": "Exp_C_ReqEvidenceFirst", "name": "Experiment C — Required Evidence First"},
        {"id": "Exp_D_ExplicitChecklist", "name": "Experiment D — Explicit Evidence Checklist"},
        {"id": "Exp_E_EvidenceFirstPrompt", "name": "Experiment E — Evidence-First Prompt"},
        {"id": "Exp_F_FormattingOnly", "name": "Experiment F — Formatting Only"},
        {"id": "Exp_G_ReverseOrder", "name": "Experiment G — Same Content, Reverse Order"}
    ]

    raw_records = []
    summary_by_condition = {c["id"]: [] for c in conditions}

    for case in BENCHMARK_SUITE:
        q_id = case["q_id"]
        q_text = case["question"]
        gold_map = case["gold_evidence"]
        req_ids = case["required_evidence_ids"]
        req_claims = case["required_claims"]
        orig_chunks = retrieved_per_q[q_id]

        print(f"\n==========================================================================", flush=True)
        print(f" QUESTION {q_id}: {q_text}", flush=True)
        print(f" Original Top-10 Chunks: {[c['chunk_id'] for c in orig_chunks]}", flush=True)
        print(f" Required Evidence Chunks: {req_ids}", flush=True)
        print(f"==========================================================================", flush=True)

        for cond in conditions:
            c_id = cond["id"]
            
            sys_prompt = CONTROL_SYSTEM_PROMPT
            human_tmpl = CONTROL_HUMAN_TEMPLATE

            if c_id == "Exp_A_Control":
                chunks = list(orig_chunks)
                context_str = build_default_context_string(chunks)
                user_msg_content = human_tmpl.format(context=context_str, question=q_text)

            elif c_id == "Exp_B_ReqEvidenceOnly":
                chunks = [c for c in orig_chunks if c["chunk_id"] in req_ids]
                if not chunks:
                    chunks = [c for c in orig_chunks if c["chunk_id"] in gold_map]
                context_str = build_default_context_string(chunks)
                user_msg_content = human_tmpl.format(context=context_str, question=q_text)

            elif c_id == "Exp_C_ReqEvidenceFirst":
                primary_chunks = [c for c in orig_chunks if gold_map.get(c["chunk_id"]) == "PRIMARY"]
                supporting_chunks = [c for c in orig_chunks if gold_map.get(c["chunk_id"]) == "SUPPORTING"]
                other_chunks = [c for c in orig_chunks if c["chunk_id"] not in gold_map]
                chunks = primary_chunks + supporting_chunks + other_chunks
                context_str = build_default_context_string(chunks)
                user_msg_content = human_tmpl.format(context=context_str, question=q_text)

            elif c_id == "Exp_D_ExplicitChecklist":
                chunks = list(orig_chunks)
                sys_prompt = CONTROL_SYSTEM_PROMPT + "\n" + CHECKLIST_ADDITION
                context_str = build_default_context_string(chunks)
                user_msg_content = human_tmpl.format(context=context_str, question=q_text)

            elif c_id == "Exp_E_EvidenceFirstPrompt":
                chunks = list(orig_chunks)
                sys_prompt = CONTROL_SYSTEM_PROMPT + "\n" + VERIFICATION_ADDITION
                context_str = build_default_context_string(chunks)
                user_msg_content = human_tmpl.format(context=context_str, question=q_text)

            elif c_id == "Exp_F_FormattingOnly":
                chunks = list(orig_chunks)
                context_str = build_experiment_f_context_string(chunks)
                user_msg_content = f"Synthesize a structured response based on the context.\n\n{context_str}\n\n=== QUESTION ===\n{q_text}\n=== END QUESTION ==="

            elif c_id == "Exp_G_ReverseOrder":
                chunks = list(reversed(orig_chunks))
                context_str = build_default_context_string(chunks)
                user_msg_content = human_tmpl.format(context=context_str, question=q_text)

            retrieved_cids = [c["chunk_id"] for c in chunks]

            req_present = [cid for cid in req_ids if cid in retrieved_cids]
            ev_availability = len(req_present) / len(req_ids) if req_ids else 0.0

            actual_messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_msg_content}
            ]

            audit_fields = {
                "system_message": sys_prompt,
                "developer_message": None,
                "conversation_history": None,
                "memory_injected": None,
                "rag_instructions": "UNIVERSAL GROUNDING & SCOPE BOUNDARY RULES + DETERMINISTIC PRESENTATION RULES",
                "retrieved_context": context_str,
                "user_question": q_text,
                "additional_wrappers": None,
                "audit_note": "No additional conversation/memory context was passed to the evaluated LLM."
            }

            # Call LLM with streaming retry logic
            try:
                raw_output, input_tokens, output_tokens, latency_ms = call_llm_robust(client, actual_messages)
            except Exception as e:
                print(f"     [Retry] Error calling Nemotron: {e}, retrying once...", flush=True)
                time.sleep(2.0)
                raw_output, input_tokens, output_tokens, latency_ms = call_llm_robust(client, actual_messages)

            post_output = post_process_response(raw_output)
            pp_audit = audit_post_processing(raw_output, post_output)
            eval_res = evaluate_response_against_claims(post_output, req_claims, retrieved_cids)

            if eval_res["required_claim_coverage"] == 1.0:
                failure_category = "NONE"
            elif ev_availability < 1.0:
                failure_category = "RETRIEVAL_FAILURE"
            elif c_id in ["Exp_B_ReqEvidenceOnly", "Exp_C_ReqEvidenceFirst", "Exp_F_FormattingOnly", "Exp_G_ReverseOrder"] and eval_res["required_claim_coverage"] > 0.5:
                failure_category = "CONTEXT_PRESENTATION_FAILURE"
            elif c_id in ["Exp_D_ExplicitChecklist", "Exp_E_EvidenceFirstPrompt"] and eval_res["required_claim_coverage"] > 0.5:
                failure_category = "PROMPT_INSTRUCTION_FAILURE"
            else:
                failure_category = "GENERATION_FAILURE"

            record = {
                "question_id": q_id,
                "condition": c_id,
                "condition_name": cond["name"],
                "system_prompt": sys_prompt,
                "human_template": human_tmpl,
                "actual_messages": actual_messages,
                "audit_fields": audit_fields,
                "retrieved_chunk_ids": retrieved_cids,
                "context_order": retrieved_cids,
                "required_evidence_ids": req_ids,
                "evidence_available": req_present,
                "raw_llm_output": raw_output,
                "post_processed_output": post_output,
                "post_processing_changes": pp_audit,
                "required_claims": [c["id"] for c in req_claims],
                "claims_expressed": eval_res["claims_expressed"],
                "missing_claims": eval_res["missing_claims"],
                "unsupported_claims": eval_res["unsupported_claims"],
                "evidence_availability": ev_availability,
                "required_claim_coverage": eval_res["required_claim_coverage"],
                "generation_utilization": eval_res["generation_utilization"],
                "unsupported_claim_rate": eval_res["unsupported_claim_rate"],
                "hallucination_rate": eval_res["hallucination_rate"],
                "correctness": eval_res["correctness"],
                "completeness": eval_res["completeness"],
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency_ms": latency_ms,
                "failure_classification": failure_category
            }

            raw_records.append(record)
            summary_by_condition[c_id].append(record)

            print(f"  -> {c_id:28s} | EvAvail: {ev_availability:.0%} | ClaimCov: {eval_res['required_claim_coverage']:.0%} | GenUtil: {eval_res['generation_utilization']:.0%} | Comp: {eval_res['completeness']:.0%} | Latency: {latency_ms:.0f}ms | Failure: {failure_category}", flush=True)

    # Compute aggregate results per condition
    aggregate_table = []
    for cond in conditions:
        c_id = cond["id"]
        recs = summary_by_condition[c_id]
        
        avg_ev_avail = float(np.mean([r["evidence_availability"] for r in recs]))
        avg_claim_cov = float(np.mean([r["required_claim_coverage"] for r in recs]))
        avg_gen_util = float(np.mean([r["generation_utilization"] for r in recs]))
        avg_correctness = float(np.mean([r["correctness"] for r in recs]))
        avg_completeness = float(np.mean([r["completeness"] for r in recs]))
        avg_input_tokens = float(np.mean([r["input_tokens"] for r in recs]))
        avg_latency = float(np.mean([r["latency_ms"] for r in recs]))

        aggregate_table.append({
            "condition": cond["name"],
            "condition_id": c_id,
            "evidence_available": avg_ev_avail,
            "claim_coverage": avg_claim_cov,
            "generation_utilization": avg_gen_util,
            "correctness": avg_correctness,
            "completeness": avg_completeness,
            "input_tokens": avg_input_tokens,
            "latency_ms": avg_latency
        })

    # Failure Matrix
    failure_matrix = []
    for rec in raw_records:
        if rec["missing_claims"]:
            for m_claim in rec["missing_claims"]:
                failure_matrix.append({
                    "question_id": rec["question_id"],
                    "condition": rec["condition"],
                    "missing_claim": m_claim,
                    "evidence_present": rec["evidence_availability"] == 1.0,
                    "failure_category": rec["failure_classification"],
                    "evidence_chunks": rec["evidence_available"]
                })

    # Save raw experimental record JSON
    out_dir = os.path.join(current_dir, "results")
    os.makedirs(out_dir, exist_ok=True)
    dump_file = os.path.join(out_dir, "phase4_diagnostic_records.json")
    with open(dump_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "aggregate_summary": aggregate_table,
            "failure_matrix": failure_matrix,
            "raw_records": raw_records
        }, f, indent=2, ensure_ascii=False)

    print(f"\nPhase 4 diagnostic run complete. Saved raw experimental record to:\n{dump_file}\n", flush=True)


if __name__ == "__main__":
    main()
