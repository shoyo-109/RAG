import os
import sys
import json
import re
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
records_file = os.path.join(current_dir, "results", "phase4_diagnostic_records.json")

with open(records_file, "r", encoding="utf-8") as f:
    data = json.load(f)

raw_records = data["raw_records"]

# Define Benchmark Claim Structure & Supporting Evidence mapping
BENCHMARK_CLAIMS = {
    "Q01": {
        "question": "Who is Aegeon and what is his origin?",
        "total_required_evidence": 2, # C4, C2
        "gold_evidence_chunks": ["C4", "C2"],
        "claims": [
            {
                "claim_id": "Q01_C1",
                "desc": "Aegeon is a merchant from/born in Syracuse.",
                "supporting_evidence": ["C4", "C2"],
                "kws": ["syracuse", "syracusian", "merchant"]
            },
            {
                "claim_id": "Q01_C2",
                "desc": "He married and traveled to Epidamnum / had twin sons born there.",
                "supporting_evidence": ["C4"],
                "kws": ["epidamnum", "married", "wife", "sons", "twins"]
            }
        ]
    },
    "Q02": {
        "question": "Why was Aegeon sentenced to death in Ephesus?",
        "total_required_evidence": 4, # C2, C4, C10, C12
        "gold_evidence_chunks": ["C2", "C4", "C10", "C12"],
        "claims": [
            {
                "claim_id": "Q02_C1",
                "desc": "Ephesus has a law/statute prohibiting Syracusians from entering Ephesus on pain of death or paying 1,000 marks ransom.",
                "supporting_evidence": ["C2", "C10", "C12"],
                "kws": ["law", "statute", "syracuse", "syracusian", "death", "1000", "thousand", "marks", "ransom", "prohibit"]
            },
            {
                "claim_id": "Q02_C2",
                "desc": "Aegeon's wealth (under 100 marks) is insufficient to pay the required 1,000 marks ransom.",
                "supporting_evidence": ["C2", "C10", "C4"],
                "kws": ["wealth", "100", "hundred", "unable", "pay", "ransom", "sum", "buy"]
            }
        ]
    },
    "Q03": {
        "question": "Why does Antipholus of Syracuse fail to recognize Adriana when she addresses him as her husband?",
        "total_required_evidence": 4, # C15, C24, C29, C61
        "gold_evidence_chunks": ["C15", "C24", "C29", "C61"],
        "claims": [
            {
                "claim_id": "Q03_C1",
                "desc": "Adriana is actually married to Antipholus of Ephesus (his long-lost twin brother).",
                "supporting_evidence": ["C24", "C61"],
                "kws": ["adriana", "ephesus", "husband", "married"]
            },
            {
                "claim_id": "Q03_C2",
                "desc": "Antipholus of Syracuse is a visitor/stranger in Ephesus who is unaware that his twin brother lives there.",
                "supporting_evidence": ["C15"],
                "kws": ["syracuse", "ephesus", "twin", "brother", "seeking", "unaware", "stranger"]
            },
            {
                "claim_id": "Q03_C3",
                "desc": "Antipholus of Syracuse is mistaken for his identical twin brother by Adriana due to their identical appearance.",
                "supporting_evidence": ["C15", "C61"],
                "kws": ["mistake", "mistaken", "identity", "twin", "confuse", "resemble", "looks"]
            }
        ]
    },
    "Q04": {
        "question": "How did Aegeon come to have two pairs of identical twin sons and twin servants?",
        "total_required_evidence": 3, # C4, C6, C8
        "gold_evidence_chunks": ["C4", "C6", "C8"],
        "claims": [
            {
                "claim_id": "Q04_C1",
                "desc": "Aegeon's wife gave birth to identical twin sons at an inn in Epidamnum.",
                "supporting_evidence": ["C4"],
                "kws": ["wife", "born", "birth", "twin", "sons", "epidamnum"]
            },
            {
                "claim_id": "Q04_C2",
                "desc": "At the same inn, a poor woman gave birth to twin boys, which Aegeon bought to serve as servants for his twin sons.",
                "supporting_evidence": ["C4"],
                "kws": ["poor", "woman", "bought", "servant", "attend", "twin"]
            },
            {
                "claim_id": "Q04_C3",
                "desc": "A violent storm shipwrecked them, splitting their ship on a rock, separating the family into two groups (one set saved by Corinth fishermen, the other by another ship).",
                "supporting_evidence": ["C6"], # C6 contains storm & mast separation
                "kws": ["ship", "storm", "rock", "split", "separated", "corinth", "fishermen", "saved", "rescued", "mast"]
            }
        ]
    },
    "Q05": {
        "question": "How do Antipholus of Syracuse and Antipholus of Ephesus differ in their initial interactions with Dromio of Syracuse?",
        "total_required_evidence": 3, # C42, C65, C12
        "gold_evidence_chunks": ["C42", "C65", "C12"],
        "claims": [
            {
                "claim_id": "Q05_C1",
                "desc": "Antipholus of Syracuse trusts Dromio of Syracuse with money, but later beats/scolds him when confused by mistaken identity.",
                "supporting_evidence": ["C12", "C42"],
                "kws": ["syracuse", "trust", "money", "beat", "beats", "struck", "gold"]
            },
            {
                "claim_id": "Q05_C2",
                "desc": "Antipholus of Ephesus is locked out of his own house by Dromio of Syracuse (who thinks his master Antipholus of Syracuse is inside).",
                "supporting_evidence": ["C65"],
                "kws": ["ephesus", "locked", "door", "denied", "house", "entry", "gate"]
            }
        ]
    },
    "Q06": {
        "question": "Why is Antipholus of Ephesus arrested, and what role does Angelo the goldsmith play in the debt dispute?",
        "total_required_evidence": 4, # C105, C107, C147, C155
        "gold_evidence_chunks": ["C105", "C107", "C147", "C155"],
        "claims": [
            {
                "claim_id": "Q06_C1",
                "desc": "Angelo the goldsmith delivered a gold chain to Antipholus of Syracuse by mistake, but demanded 200 ducats payment from Antipholus of Ephesus.",
                "supporting_evidence": ["C105", "C107"],
                "kws": ["angelo", "goldsmith", "chain", "syracuse", "ephesus", "200", "ducats", "demand", "payment"]
            },
            {
                "claim_id": "Q06_C2",
                "desc": "Antipholus of Ephesus denied receiving the chain/paying for it, leading Angelo to have him arrested for the 200 ducats debt.",
                "supporting_evidence": ["C147", "C155"],
                "kws": ["arrest", "arrested", "denied", "debt", "officer", "chain"]
            }
        ]
    }
}

conditions = [
    "Exp_A_Control",
    "Exp_B_ReqEvidenceOnly",
    "Exp_C_ReqEvidenceFirst",
    "Exp_D_ExplicitChecklist",
    "Exp_E_EvidenceFirstPrompt",
    "Exp_F_FormattingOnly",
    "Exp_G_ReverseOrder"
]

recalculated_records = []
claim_level_audit = []

for rec in raw_records:
    q_id = rec["question_id"]
    cond_id = rec["condition"]
    output_text = rec["post_processed_output"].lower()
    retrieved_cids = set(rec["retrieved_chunk_ids"])
    
    benchmark_info = BENCHMARK_CLAIMS[q_id]
    gold_cids = benchmark_info["gold_evidence_chunks"]
    total_req_evidence_units = benchmark_info["total_required_evidence"]
    
    # 1. Evidence Availability calculation
    req_units_present = sum(1 for cid in gold_cids if cid in retrieved_cids)
    evidence_availability_pct = req_units_present / total_req_evidence_units if total_req_evidence_units > 0 else 0.0
    
    # 2. Claim-level evaluation
    total_required_claims = len(benchmark_info["claims"])
    expressed_required_claims_count = 0
    available_required_claims_count = 0
    expressed_available_claims_count = 0
    
    retrieval_failures_count = 0
    generation_failures_count = 0
    success_count = 0

    for claim_def in benchmark_info["claims"]:
        c_id = claim_def["claim_id"]
        supp_cids = claim_def["supporting_evidence"]
        kws = claim_def["kws"]
        
        # Check if evidence for this specific claim was present in retrieved_cids
        evidence_present = any(cid in retrieved_cids for cid in supp_cids)
        
        # Check if claim was expressed in response text
        hits = sum(1 for kw in kws if kw in output_text)
        claim_expressed = hits >= min(2, len(kws))
        
        if claim_expressed:
            expressed_required_claims_count += 1
            
        if evidence_present:
            available_required_claims_count += 1
            if claim_expressed:
                expressed_available_claims_count += 1
                classification = "SUCCESSFUL_GENERATION"
                success_count += 1
            else:
                classification = "GENERATION_FAILURE"
                generation_failures_count += 1
        else:
            if claim_expressed:
                # Model inferred/expressed claim from surrounding context despite primary chunk missing
                classification = "SUCCESSFUL_GENERATION"
                success_count += 1
            else:
                classification = "RETRIEVAL_FAILURE"
                retrieval_failures_count += 1

        claim_level_audit.append({
            "question_id": q_id,
            "condition": cond_id,
            "claim_id": c_id,
            "claim_desc": claim_def["desc"],
            "evidence_present": evidence_present,
            "claim_expressed": claim_expressed,
            "classification": classification,
            "supporting_evidence": supp_cids
        })

    # Metrics calculation
    required_claim_coverage_pct = expressed_required_claims_count / total_required_claims if total_required_claims > 0 else 0.0
    
    if available_required_claims_count > 0:
        generation_utilization_pct = expressed_available_claims_count / available_required_claims_count
    else:
        generation_utilization_pct = None # N/A

    # Correctness and completeness
    if required_claim_coverage_pct == 1.0:
        correctness = 2
        completeness = 1.0
    elif required_claim_coverage_pct >= 0.5:
        correctness = 1
        completeness = required_claim_coverage_pct
    else:
        correctness = 0
        completeness = required_claim_coverage_pct

    recalculated_records.append({
        "question_id": q_id,
        "condition": cond_id,
        "evidence_available_percent": evidence_availability_pct,
        "required_claim_coverage_percent": required_claim_coverage_pct,
        "total_required_claims": total_required_claims,
        "expressed_required_claims": expressed_required_claims_count,
        "available_required_claim_count": available_required_claims_count,
        "expressed_available_claim_count": expressed_available_claims_count,
        "generation_utilization_percent": generation_utilization_pct,
        "retrieval_failures_count": retrieval_failures_count,
        "generation_failures_count": generation_failures_count,
        "success_count": success_count,
        "correctness": correctness,
        "completeness": completeness,
        "input_tokens": rec["input_tokens"],
        "output_tokens": rec["output_tokens"],
        "latency_ms": rec["latency_ms"]
    })

# Compute Aggregate Results per Condition using total claim counts
cond_aggregates = []
for cond in conditions:
    cond_recs = [r for r in recalculated_records if r["condition"] == cond]
    
    tot_req_evidence_units = sum(BENCHMARK_CLAIMS[r["question_id"]]["total_required_evidence"] for r in cond_recs)
    tot_req_evidence_present = sum(sum(1 for cid in BENCHMARK_CLAIMS[r["question_id"]]["gold_evidence_chunks"] if cid in set(raw_rec["retrieved_chunk_ids"])) for r, raw_rec in zip(cond_recs, [rr for rr in raw_records if rr["condition"] == cond]))
    agg_evidence_avail = tot_req_evidence_present / tot_req_evidence_units
    
    sum_tot_claims = sum(r["total_required_claims"] for r in cond_recs)
    sum_expr_claims = sum(r["expressed_required_claims"] for r in cond_recs)
    agg_claim_coverage = sum_expr_claims / sum_tot_claims
    
    sum_avail_claims = sum(r["available_required_claim_count"] for r in cond_recs)
    sum_expr_avail_claims = sum(r["expressed_available_claim_count"] for r in cond_recs)
    agg_gen_utilization = sum_expr_avail_claims / sum_avail_claims if sum_avail_claims > 0 else 0.0
    
    tot_retrieval_failures = sum(r["retrieval_failures_count"] for r in cond_recs)
    tot_generation_failures = sum(r["generation_failures_count"] for r in cond_recs)
    
    avg_correctness = float(np.mean([r["correctness"] for r in cond_recs]))
    avg_completeness = float(np.mean([r["completeness"] for r in cond_recs]))
    avg_input_tokens = float(np.mean([r["input_tokens"] for r in cond_recs]))
    avg_latency = float(np.mean([r["latency_ms"] for r in cond_recs]))
    
    cond_aggregates.append({
        "condition": cond,
        "evidence_availability": agg_evidence_avail,
        "claim_coverage": agg_claim_coverage,
        "generation_utilization": agg_gen_utilization,
        "retrieval_failures": tot_retrieval_failures,
        "generation_failures": tot_generation_failures,
        "sum_tot_claims": sum_tot_claims,
        "sum_expr_claims": sum_expr_claims,
        "sum_avail_claims": sum_avail_claims,
        "sum_expr_avail_claims": sum_expr_avail_claims,
        "correctness": avg_correctness,
        "completeness": avg_completeness,
        "input_tokens": avg_input_tokens,
        "latency_ms": avg_latency
    })

print("=========================================================================")
print("                   CORRECTED PHASE 4A AGGREGATE RESULTS                   ")
print("=========================================================================")
print(f"{'Condition':30s} | {'EvAvail':8s} | {'ClaimCov':8s} | {'GenUtil':8s} | {'RetrFail':8s} | {'GenFail':8s} | {'Corr':5s} | {'Comp':5s}")
print("-" * 95)
for ca in cond_aggregates:
    print(f"{ca['condition']:30s} | {ca['evidence_availability']:8.1%} | {ca['claim_coverage']:8.1%} | {ca['generation_utilization']:8.1%} | {ca['retrieval_failures']:8d} | {ca['generation_failures']:8d} | {ca['correctness']:5.2f} | {ca['completeness']:5.1%}")

# Save recalculated results JSON
out_payload = {
    "recalculated_aggregates": cond_aggregates,
    "recalculated_records": recalculated_records,
    "claim_level_audit": claim_level_audit
}

with open(os.path.join(current_dir, "results", "phase4a_recalculated_results.json"), "w", encoding="utf-8") as f:
    json.dump(out_payload, f, indent=2, ensure_ascii=False)

print("\nSaved recalculated Phase 4A payload to phase4a_recalculated_results.json.")
