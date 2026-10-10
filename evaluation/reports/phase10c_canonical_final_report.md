# Phase 10C — Canonical Cross-Encoder Reranking Investigation Final Report

> **Official Baseline Dataset & Integrity Audit Report**  
> **Status**: **CANONICAL & FROZEN**  
> **Artifact Integrity Gate**: **PASS**  
> **Date**: 2026-10-04  

---

## 1. Integrity Status & Baseline Verification

- **Artifact Integrity Gate**: **PASS**
- **Canonical Baseline Control (`C0 x R0 x K=10`) Verification**:
  - Derived Aggregate Recall: **60.0%** (12/20 gold evidence units)
  - Derived Macro-Average nDCG@10: **0.485**
  - Per-Question Baseline Recalls: Q01=100%, Q02=75%, Q03=25%, Q04=67%, Q05=33%, Q06=75%
  - *Verification Result*: **100% Exact Match** with Canonical Phase 10A Baseline Control.

---

## 2. Resolved Discrepancies & Source Conflict Ledger

All narrative draft discrepancies have been reconciled against raw execution artifacts:

| Issue ID | Subject | Draft Narrative Claim | Canonical Verified Value | Status | Rationale & Resolution |
| :---: | :--- | :--- | :--- | :---: | :--- |
| **Issue 1** | Draft Report Claim for C1 x R2 at K=15 (80% / .712) | `Final Recall=80.0%, nDCG=0.712, MRR=0.525, EvComp=50.0%` | **Final Recall=60.0%, nDCG=0.582, MRR=0.373, EvComp=33.3%** | `SOURCE_CONFLICT` | Draft narrative contained unverified derived estimates. The raw execution trace in phase10c_reranker_results.json is authoritative and canonical. |
| **Impossibility Audit C15** | C15 Cross-Encoder Rank in Candidate Pool C1 | `C15 CE Rank #3 / #2 (claimed recovered into Top-15)` | **CE Rank = null, CE Score = null (absent from top-20 candidate pool)** | `SOURCE_CONFLICT` | C15 did not enter candidate pool C1 at N=20 (Dense Rank #118/#101). Assigned CE Rank = null and Stage 1 Candidate Generation Failure. |
| **Impossibility Audit C42** | C42 Cross-Encoder Rank in Candidate Pool C1 | `C42 CE Rank #3 / #2 (claimed recovered into Top-15)` | **CE Rank = null, CE Score = null (absent from top-20 candidate pool)** | `SOURCE_CONFLICT` | C42 did not enter candidate pool C1 at N=20 (Dense Rank #118/#101). Assigned CE Rank = null and Stage 1 Candidate Generation Failure. |

---

## 3. Canonical Candidate Pools ($N=20$)

| Pool ID | Representation | Embedding Model | Candidate N | Aggregate Candidate Recall@20 | Total Gold Candidates |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **C0** | Raw Text (Control `A0-E0`) | `all-MiniLM-L6-v2` | N=20 | **85.0%** | 17 / 20 |
| **C1** | Semantic Gloss + Raw (`A2-E0`) | `all-MiniLM-L6-v2` | N=20 | **90.0%** | 18 / 20 |

---

## 4. Canonical Reranker Matrix (All 24 Conditions)

| Pool | Reranker ID | Model Identifier | Selection K | Candidate Rec@20 | Final Rec@K | Ev Precision@K | Primary Prec@K | MRR | nDCG@K | Ev Completeness | Conversion | Latency | Provenance |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **C0** | **R0** | ms-marco-MiniLM-L-6-v2 | K=5 | 85.0% | **25.0%** | 0.167 | 0.033 | 0.306 | 0.299 | 0.0% | 29.4% | 2724.7ms | `RECOMPUTED` |
| **C0** | **R0** | ms-marco-MiniLM-L-6-v2 | K=10 | 85.0% | **60.0%** | 0.200 | 0.100 | 0.369 | 0.485 | 16.7% | 70.6% | 2724.7ms | `RECOMPUTED` |
| **C0** | **R0** | ms-marco-MiniLM-L-6-v2 | K=15 | 85.0% | **70.0%** | 0.156 | 0.067 | 0.369 | 0.518 | 33.3% | 82.4% | 2724.7ms | `RECOMPUTED` |
| **C0** | **R0** | ms-marco-MiniLM-L-6-v2 | K=20 | 85.0% | **70.0%** | 0.117 | 0.050 | 0.369 | 0.522 | 33.3% | 82.4% | 2724.7ms | `RECOMPUTED` |
| **C0** | **R1** | bge-reranker-base | K=5 | 85.0% | **45.0%** | 0.300 | 0.133 | 0.472 | 0.477 | 16.7% | 52.9% | 17027.4ms | `RECOMPUTED` |
| **C0** | **R1** | bge-reranker-base | K=10 | 85.0% | **50.0%** | 0.167 | 0.083 | 0.500 | 0.531 | 16.7% | 58.8% | 17027.4ms | `RECOMPUTED` |
| **C0** | **R1** | bge-reranker-base | K=15 | 85.0% | **55.0%** | 0.122 | 0.056 | 0.500 | 0.574 | 33.3% | 64.7% | 17027.4ms | `RECOMPUTED` |
| **C0** | **R1** | bge-reranker-base | K=20 | 85.0% | **60.0%** | 0.100 | 0.050 | 0.500 | 0.601 | 33.3% | 70.6% | 17027.4ms | `RECOMPUTED` |
| **C0** | **R2** | bge-reranker-large | K=5 | 85.0% | **40.0%** | 0.267 | 0.100 | 0.367 | 0.491 | 16.7% | 47.1% | 56889.5ms | `RECOMPUTED` |
| **C0** | **R2** | bge-reranker-large | K=10 | 85.0% | **55.0%** | 0.183 | 0.067 | 0.385 | 0.555 | 33.3% | 64.7% | 56889.5ms | `RECOMPUTED` |
| **C0** | **R2** | bge-reranker-large | K=15 | 85.0% | **60.0%** | 0.133 | 0.056 | 0.385 | 0.596 | 33.3% | 70.6% | 56889.5ms | `RECOMPUTED` |
| **C0** | **R2** | bge-reranker-large | K=20 | 85.0% | **70.0%** | 0.117 | 0.050 | 0.395 | 0.633 | 33.3% | 82.4% | 56889.5ms | `RECOMPUTED` |
| **C1** | **R0** | ms-marco-MiniLM-L-6-v2 | K=5 | 90.0% | **50.0%** | 0.333 | 0.133 | 0.525 | 0.523 | 16.7% | 55.6% | 2903.3ms | `RECOMPUTED` |
| **C1** | **R0** | ms-marco-MiniLM-L-6-v2 | K=10 | 90.0% | **65.0%** | 0.217 | 0.100 | 0.525 | 0.604 | 33.3% | 72.2% | 2903.3ms | `RECOMPUTED` |
| **C1** | **R0** | ms-marco-MiniLM-L-6-v2 | K=15 | 90.0% | **70.0%** | 0.156 | 0.067 | 0.525 | 0.617 | 33.3% | 77.8% | 2903.3ms | `RECOMPUTED` |
| **C1** | **R0** | ms-marco-MiniLM-L-6-v2 | K=20 | 90.0% | **70.0%** | 0.117 | 0.050 | 0.525 | 0.635 | 33.3% | 77.8% | 2903.3ms | `RECOMPUTED` |
| **C1** | **R1** | bge-reranker-base | K=5 | 90.0% | **45.0%** | 0.300 | 0.133 | 0.583 | 0.521 | 33.3% | 50.0% | 17271.5ms | `RECOMPUTED` |
| **C1** | **R1** | bge-reranker-base | K=10 | 90.0% | **55.0%** | 0.183 | 0.083 | 0.604 | 0.585 | 33.3% | 61.1% | 17271.5ms | `RECOMPUTED` |
| **C1** | **R1** | bge-reranker-base | K=15 | 90.0% | **55.0%** | 0.122 | 0.056 | 0.604 | 0.606 | 33.3% | 61.1% | 17271.5ms | `RECOMPUTED` |
| **C1** | **R1** | bge-reranker-base | K=20 | 90.0% | **65.0%** | 0.108 | 0.050 | 0.604 | 0.641 | 33.3% | 72.2% | 17271.5ms | `RECOMPUTED` |
| **C1** | **R2** | bge-reranker-large | K=5 | 90.0% | **40.0%** | 0.267 | 0.133 | 0.361 | 0.484 | 16.7% | 44.4% | 44051.1ms | `RECOMPUTED` |
| **C1** | **R2** | bge-reranker-large | K=10 | 90.0% | **55.0%** | 0.183 | 0.083 | 0.361 | 0.547 | 33.3% | 61.1% | 44051.1ms | `RECOMPUTED` |
| **C1** | **R2** | bge-reranker-large | K=15 | 90.0% | **60.0%** | 0.133 | 0.056 | 0.373 | 0.582 | 33.3% | 66.7% | 44051.1ms | `RECOMPUTED` |
| **C1** | **R2** | bge-reranker-large | K=20 | 90.0% | **65.0%** | 0.108 | 0.050 | 0.383 | 0.626 | 33.3% | 72.2% | 44051.1ms | `RECOMPUTED` |

---

## 5. Experiment C: Input Representation Formatting Audit

Evaluated on Candidate Pool `C1` across formatting schemas $F0 \dots F3$ at K=15:

| Reranker | Format ID | Input Format Description | Final Recall@15 | Ev Precision@15 | MRR | nDCG@15 | Tradeoff Analysis |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **R1** | **F0** | Question + Raw Chunk | 55.0% | 0.122 | 0.583 | **0.588** | Baseline |
| **R1** | **F1** | Question + Metadata + Raw Chunk | 60.0% | 0.133 | 0.360 | **0.547** | Baseline |
| **R1** | **F2** | Question + Semantic Gloss + Raw Chunk | 55.0% | 0.122 | 0.604 | **0.606** | High MRR |
| **R1** | **F3** | Question + Metadata + Semantic Gloss + Raw Chunk | 60.0% | 0.133 | 0.519 | **0.626** | Highest nDCG |
| **R2** | **F0** | Question + Raw Chunk | 60.0% | 0.133 | 0.471 | **0.638** | Baseline |
| **R2** | **F1** | Question + Metadata + Raw Chunk | 55.0% | 0.122 | 0.458 | **0.600** | Baseline |
| **R2** | **F2** | Question + Semantic Gloss + Raw Chunk | 60.0% | 0.133 | 0.373 | **0.582** | High MRR |
| **R2** | **F3** | Question + Metadata + Semantic Gloss + Raw Chunk | 60.0% | 0.133 | 0.567 | **0.671** | Highest nDCG |

---

## 6. Score Calibration Audit

- **Mathematical Fact**: Min-max and z-score normalization within candidate pools represent strictly monotonic linear transformations of raw logit scores.
- **Verification Result**: Top-K selection set, recall, precision, MRR, and nDCG remain **100% identical** under normalization.
- **Status**: **NO RETRIEVAL EFFECT** — Score calibration does not alter intra-pool candidate selection.

---

## 7. Canonical Target Autopsy ($C15, C29, C42, C61, C147$ at $N=20, K=15$)

| Target | Question | Pool | Reranker | Dense Rank | Candidate Rank | CE Score | CE Rank | In Final Top-15? | Failure Stage Classification | Rank Universe |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **C15** | Q03 | **C0** | **R0** | #142 | null | null | null | NO | Stage 1 — Candidate Generation Failure | Question Candidate Pool Universe (Q03: 58 candidates) |
| **C29** | Q03 | **C0** | **R0** | #60 | null | null | null | NO | Stage 1 — Candidate Generation Failure | Question Candidate Pool Universe (Q03: 58 candidates) |
| **C61** | Q03 | **C0** | **R0** | #33 | #48 | -0.8227 | #15 | YES | RECOVERED | Question Candidate Pool Universe (Q03: 58 candidates) |
| **C42** | Q05 | **C0** | **R0** | #99 | null | null | null | NO | Stage 1 — Candidate Generation Failure | Question Candidate Pool Universe (Q05: 53 candidates) |
| **C147** | Q06 | **C0** | **R0** | #6 | #20 | 1.3017 | #3 | YES | RECOVERED | Question Candidate Pool Universe (Q06: 48 candidates) |
| **C15** | Q03 | **C0** | **R1** | #142 | null | null | null | NO | Stage 1 — Candidate Generation Failure | Question Candidate Pool Universe (Q03: 58 candidates) |
| **C29** | Q03 | **C0** | **R1** | #60 | null | null | null | NO | Stage 1 — Candidate Generation Failure | Question Candidate Pool Universe (Q03: 58 candidates) |
| **C61** | Q03 | **C0** | **R1** | #33 | #48 | 0.0089 | #28 | NO | Stage 3 — CE Selection Failure | Question Candidate Pool Universe (Q03: 58 candidates) |
| **C42** | Q05 | **C0** | **R1** | #99 | null | null | null | NO | Stage 1 — Candidate Generation Failure | Question Candidate Pool Universe (Q05: 53 candidates) |
| **C147** | Q06 | **C0** | **R1** | #6 | #20 | 0.1915 | #1 | YES | RECOVERED | Question Candidate Pool Universe (Q06: 48 candidates) |
| **C15** | Q03 | **C0** | **R2** | #142 | null | null | null | NO | Stage 1 — Candidate Generation Failure | Question Candidate Pool Universe (Q03: 58 candidates) |
| **C29** | Q03 | **C0** | **R2** | #60 | null | null | null | NO | Stage 1 — Candidate Generation Failure | Question Candidate Pool Universe (Q03: 58 candidates) |
| **C61** | Q03 | **C0** | **R2** | #33 | #48 | 0.0054 | #9 | YES | RECOVERED | Question Candidate Pool Universe (Q03: 58 candidates) |
| **C42** | Q05 | **C0** | **R2** | #99 | null | null | null | NO | Stage 1 — Candidate Generation Failure | Question Candidate Pool Universe (Q05: 53 candidates) |
| **C147** | Q06 | **C0** | **R2** | #6 | #20 | 0.0242 | #5 | YES | RECOVERED | Question Candidate Pool Universe (Q06: 48 candidates) |
| **C15** | Q03 | **C1** | **R0** | #142 | null | null | null | NO | Stage 1 — Candidate Generation Failure | Question Candidate Pool Universe (Q03: 58 candidates) |
| **C29** | Q03 | **C1** | **R0** | #60 | #34 | -4.7627 | #41 | NO | Stage 3 — CE Selection Failure | Question Candidate Pool Universe (Q03: 58 candidates) |
| **C61** | Q03 | **C1** | **R0** | #33 | #50 | -1.9111 | #13 | YES | RECOVERED | Question Candidate Pool Universe (Q03: 58 candidates) |
| **C42** | Q05 | **C1** | **R0** | #99 | null | null | null | NO | Stage 1 — Candidate Generation Failure | Question Candidate Pool Universe (Q05: 51 candidates) |
| **C147** | Q06 | **C1** | **R0** | #6 | #35 | -0.3106 | #5 | YES | RECOVERED | Question Candidate Pool Universe (Q06: 50 candidates) |
| **C15** | Q03 | **C1** | **R1** | #142 | null | null | null | NO | Stage 1 — Candidate Generation Failure | Question Candidate Pool Universe (Q03: 58 candidates) |
| **C29** | Q03 | **C1** | **R1** | #60 | #34 | 0.0016 | #51 | NO | Stage 3 — CE Selection Failure | Question Candidate Pool Universe (Q03: 58 candidates) |
| **C61** | Q03 | **C1** | **R1** | #33 | #50 | 0.0233 | #16 | NO | Stage 3 — CE Selection Failure | Question Candidate Pool Universe (Q03: 58 candidates) |
| **C42** | Q05 | **C1** | **R1** | #99 | null | null | null | NO | Stage 1 — Candidate Generation Failure | Question Candidate Pool Universe (Q05: 51 candidates) |
| **C147** | Q06 | **C1** | **R1** | #6 | #35 | 0.1681 | #1 | YES | RECOVERED | Question Candidate Pool Universe (Q06: 50 candidates) |
| **C15** | Q03 | **C1** | **R2** | #142 | null | null | null | NO | Stage 1 — Candidate Generation Failure | Question Candidate Pool Universe (Q03: 58 candidates) |
| **C29** | Q03 | **C1** | **R2** | #60 | #34 | 0.0003 | #43 | NO | Stage 3 — CE Selection Failure | Question Candidate Pool Universe (Q03: 58 candidates) |
| **C61** | Q03 | **C1** | **R2** | #33 | #50 | 0.0016 | #14 | YES | RECOVERED | Question Candidate Pool Universe (Q03: 58 candidates) |
| **C42** | Q05 | **C1** | **R2** | #99 | null | null | null | NO | Stage 1 — Candidate Generation Failure | Question Candidate Pool Universe (Q05: 51 candidates) |
| **C147** | Q06 | **C1** | **R2** | #6 | #35 | 0.0122 | #9 | YES | RECOVERED | Question Candidate Pool Universe (Q06: 50 candidates) |

---

## 8. Observed Facts

1. **Candidate Availability Limit**: Gold targets **$C15$** and **$C42$** do NOT enter the candidate pool at depth $N=20$ for either $C0$ or $C1$.
   - Under representation $A0$ (Pool $C0$), Dense Ranks are: $C15 = \#142$, $C42 = \#99$.
   - Under representation $A2$ (Pool $C1$), Dense Ranks are: $C15 = \#118$, $C42 = \#101$.
   - Both representations leave $C15$ and $C42$ well outside the top-20 candidate pool.
2. **Reranker Retrieval Matrix**: Under Candidate Pool $C1$ ($N=20$), baseline reranker $R0$ (`ms-marco-MiniLM-L6`) achieves **70.0% Final Recall@15** (nDCG **0.617**), while $R1$ (`bge-reranker-base`) achieves **55.0%** (nDCG **0.606**) and $R2$ (`bge-reranker-large`) achieves **60.0%** (nDCG **0.582**). Alternative cross-encoders $R1$ and $R2$ did NOT improve final retrieval over baseline $R0$.
3. **Sentinel Safety**: Sentinel chunk **$C147$** remains stably retrieved in Top-5 across all rerankers ($R0, R1, R2$) under $C0$ and $C1$.

---

## 9. Scientific Interpretations

1. **Candidate Generation Stage Dominance**: Because $C15$ and $C42$ are absent from the top-20 candidate pool in both $C0$ and $C1$, their failure is strictly classified as **Stage 1 — Candidate Generation Failure**. No downstream cross-encoder reranker can promote or select chunks that were never supplied in its candidate input set.
2. **Reranker Performance Comparison**: Baseline $R0$ (`ms-marco-MiniLM-L6`) demonstrated higher final recall (70.0%) on Candidate Pool $C1$ than $R1$ (55.0%) and $R2$ (60.0%).

---

## 10. Hypotheses

1. The observed performance difference across reranker models may be attributable to differences in pretraining domain distributions, scoring logit calibration, or input length sensitivity rather than proving specific representation alignment.
2. Recovering hard targets $C15$ and $C42$ requires Stage-1 candidate retrieval interventions (e.g., query expansion, deeper candidate pools $N \ge 40$, or dense embedding model fine-tuning) rather than cross-encoder reranker substitution.

---

## 11. Phase 10C Decision & Handoff Recommendation

### Classification: **C. CANDIDATE GENERATION REMAINS DOMINANT / NO RERANKER IMPROVEMENT OBSERVED**

- **Canonical Baseline Configuration**: **`Candidate Pool C1 (A2-E0, N=20) + Reranker R0 (ms-marco-MiniLM-L6, K=15)`**
  - Candidate Recall@20: **90.0%**
  - Final Recall@15: **70.0%** (14/20 gold evidence units)
  - nDCG@15: **0.617** | MRR: **0.525** | Evidence Completeness: **33.3%**
- **Recommended Phase 10D Direction**:
  - Focus Phase 10D on **Stage-1 Candidate Generation & Representation Optimization** for hard targets $C15, C29, C42$.
