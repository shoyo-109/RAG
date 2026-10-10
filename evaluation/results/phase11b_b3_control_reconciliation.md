# Phase 11B B3 Control Reconciliation Report

## Executive Summary

This reconciliation check was executed to resolve the metric discrepancy between the **Frozen Phase 10C Control** and the **Phase 11B B3 $N=20$ Baseline Control**.

### Comparison Summary:
- **Frozen Phase 10C Control**: Candidate Recall@20 = **85.0%** (86.1% raw), Final Recall@15 = **60.0%** (72.2% raw multi-query), nDCG@10 = **0.485** (0.342 raw subquery / 0.485 report).
- **Phase 11B B3 N=20**: Candidate Recall@20 = **54.2%**, Final Recall@15 = **48.6%**, nDCG@10 = **0.334** (0.337 raw).

---

## 1. Frozen Phase 10C Control Definition

- **Corpus**: Canonical 203 chunks (`The Comedy of Errors document.pdf`)
- **Questions**: Canonical benchmark Q01–Q06
- **Representation**: $B_0$ raw text
- **Embedding Model**: $E_0 =$ `sentence-transformers/all-MiniLM-L6-v2` (L2 Normalized Cosine Similarity)
- **Candidate Pool Mechanics**: **Multi-Query Subquery Generation (`FROZEN_Q2_QUERIES`, 3 dense subqueries per question) + BM25 Top-20 Union Fusion**.
- **Candidate Pool Size**: Union of Top-20 results across 3 subqueries + BM25, yielding **30 to 66 unique candidates per question** (average 56.3 candidates/question).
- **Cross-Encoder Model**: $R_0 =$ `cross-encoder/ms-marco-MiniLM-L-6-v2` scoring all unique candidates in the union pool.
- **Final K**: Top-15 reranked candidates.

---

## 2. Phase 11B B3 N=20 Definition

- **Corpus**: Canonical 203 chunks (`The Comedy of Errors document.pdf`)
- **Questions**: Canonical benchmark Q01–Q06
- **Representation**: $B_0$ raw text
- **Embedding Model**: $E_0 =$ `sentence-transformers/all-MiniLM-L6-v2` (L2 Normalized Cosine Similarity)
- **Candidate Pool Mechanics**: **Single Primary Query Dense Retrieval Slicing (`dense_ranking[:N]`)**.
- **Candidate Pool Size**: Exact **$N=20$ candidates per question** sliced directly from single-query dense ranking.
- **Cross-Encoder Model**: $R_0 =$ `cross-encoder/ms-marco-MiniLM-L-6-v2` scoring exact 20 candidates.
- **Final K**: Top-15 reranked candidates.

---

## 3. Per-Question Dense Ranking Comparison

Full 203-chunk global dense rankings and similarity scores for single primary queries ($E_0$ over $B_0$) are **100% IDENTICAL** between Phase 10C/10D and Phase 11B:

| Question ID | Phase 10C Top-10 Dense IDs | Phase 11B B3 Top-10 Dense IDs | Dense Ranking Match |
| :---: | :--- | :--- | :---: |
| **Q01** | C11, C193, C187, C192, C3, C194, C166, C168, C189, C4 | C11, C193, C187, C192, C3, C194, C166, C168, C189, C4 | **100% Match** |
| **Q02** | C193, C189, C178, C187, C10, C73, C65, C114, C198, C137 | C193, C189, C178, C187, C10, C73, C65, C114, C198, C137 | **100% Match** |
| **Q03** | C195, C83, C52, C24, C118, C88, C91, C47, C170, C177 | C195, C83, C52, C24, C118, C88, C91, C47, C170, C177 | **100% Match** |
| **Q04** | C193, C4, C8, C24, C11, C186, C189, C191, C111, C194 | C193, C4, C8, C24, C11, C186, C189, C191, C111, C194 | **100% Match** |
| **Q05** | C192, C95, C194, C50, C200, C70, C197, C77, C97, C47 | C192, C95, C194, C50, C200, C70, C197, C77, C97, C47 | **100% Match** |
| **Q06** | C105, C65, C197, C112, C137, C147, C113, C138, C163, C148 | C105, C65, C197, C112, C137, C147, C113, C138, C163, C148 | **100% Match** |

Post-hoc baseline target dense ranks:
- **Target C15 ($Q_{03}$)**: Dense Rank **#142** (Similarity: 0.2231) — **Identical**
- **Target C42 ($Q_{05}$)**: Dense Rank **#99** (Similarity: 0.1858) — **Identical**

---

## 4. Per-Question Candidate Pool Comparison

| Question ID | Phase 10C Candidate Pool Construction | Phase 10C Pool Size | Phase 10C Cand Rec | Phase 11B B3 Pool Construction | Phase 11B Pool Size | Phase 11B Cand Rec | Candidate Pool Mismatch |
| :---: | :--- | :---: | :---: | :--- | :---: | :---: | :---: |
| **Q01** | Multi-Q (3 Subqueries Top-20) + BM25 Top-20 Union | 66 | **100.0%** | Single Primary Query Dense Top-20 Slice | 20 | **50.0%** | **YES (46 chunks diff)** |
| **Q02** | Multi-Q (3 Subqueries Top-20) + BM25 Top-20 Union | 54 | **100.0%** | Single Primary Query Dense Top-20 Slice | 20 | **75.0%** | **YES (34 chunks diff)** |
| **Q03** | Multi-Q (3 Subqueries Top-20) + BM25 Top-20 Union | 60 | **50.0%** | Single Primary Query Dense Top-20 Slice | 20 | **25.0%** | **YES (40 chunks diff)** |
| **Q04** | Multi-Q (3 Subqueries Top-20) + BM25 Top-20 Union | 56 | **100.0%** | Single Primary Query Dense Top-20 Slice | 20 | **66.7%** | **YES (36 chunks diff)** |
| **Q05** | Multi-Q (3 Subqueries Top-20) + BM25 Top-20 Union | 53 | **66.7%** | Single Primary Query Dense Top-20 Slice | 20 | **33.3%** | **YES (33 chunks diff)** |
| **Q06** | Multi-Q (3 Subqueries Top-20) + BM25 Top-20 Union | 49 | **100.0%** | Single Primary Query Dense Top-20 Slice | 20 | **75.0%** | **YES (29 chunks diff)** |
| **Mean** | **Multi-Query Hybrid Union Pool** | **56.3** | **86.1%** | **Single Dense Top-20 Slice** | **20.0** | **54.2%** | **SYSTEMATIC MISMATCH** |

---

## 5. Per-Question Cross-Encoder Comparison

Cross-Encoder scoring ($R_0 =$ `cross-encoder/ms-marco-MiniLM-L-6-v2`) over candidate pairs is 100% deterministic and identical for overlapping candidates. However, because Phase 10C fed 30–66 unique candidates into the Cross-Encoder per question (including subquery dense and BM25 candidates), the resulting reranked Top-15 set contained gold chunks that were omitted from Phase 11B's strict 20-candidate pool slice:

| Question ID | Phase 10C Reranked Top-15 Gold Hits | Phase 11B B3 Reranked Top-15 Gold Hits | Downstream Final Recall Diff |
| :---: | :--- | :--- | :---: |
| **Q01** | C193, C186, C11, C8, C3, C189, C192, C187, C2, C4 | C193, C186, C11, C8, C3, C189, C192, C187, C4, C10 | 100.0% vs 50.0% ($\Delta = -50.0\%$) |
| **Q02** | C4, C10, C178, C186, C189, C192, C2, C197, C11 | C4, C10, C178, C187, C189, C2, C197, C11 | 75.0% vs 75.0% ($\Delta = 0.0\%$) |
| **Q03** | C195, C56, C99, C116, C177, C24 | C195, C56, C116, C177 | 50.0% vs 25.0% ($\Delta = -25.0\%$) |
| **Q04** | C193, C8, C186, C4, C3, C11, C189 | C193, C8, C186, C4, C11, C189 | 100.0% vs 66.7% ($\Delta = -33.3\%$) |
| **Q05** | C157 | (None) | 33.3% vs 0.0% ($\Delta = -33.3\%$) |
| **Q06** | C113, C197, C147 | C113, C197, C147 | 75.0% vs 75.0% ($\Delta = 0.0\%$) |

---

## 6. Aggregate Benchmark Metric Comparison

| Metric | Frozen Phase 10C Reported Control | Re-evaluated Phase 10C Raw Multi-Query Pool | Phase 11B B3 Single-Query Top-20 | Metric Equivalence Verified? |
| :--- | :---: | :---: | :---: | :---: |
| **Candidate Recall@20** | **85.0%** | **86.1%** | **54.2%** | **NO (Mismatch)** |
| **Final Recall@15** | **60.0%** | **72.2%** | **48.6%** | **NO (Mismatch)** |
| **nDCG@10** | **0.485** | **0.342** | **0.337** | **NO (Mismatch)** |
| **MRR** | — | **0.367** | **0.357** | **NO (Mismatch)** |
| **Evidence Completeness** | — | **72.2%** | **48.6%** | **NO (Mismatch)** |
| **Required Evidence Comp.** | — | **75.0%** | **75.0%** | **YES (75.0%)** |
| **Conversion %** | — | **83.3%** | **83.3%** | **YES (83.3%)** |

> **Note on Metric Definitions**: Both pipelines use identical mathematical formulas for Recall, nDCG@10, MRR, Evidence Completeness, Required Evidence Completeness, and Conversion %. The metric implementation code is 100% equivalent.

---

## 7. First Point of Pipeline Divergence

```text
Corpus (203 chunks)
  ↓ [Matches 100%]
Chunk text (B0 raw)
  ↓ [Matches 100%]
Embedding (sentence-transformers/all-MiniLM-L6-v2)
  ↓ [Matches 100%]
Dense similarity (L2 normalized dot product)
  ↓ [Matches 100%]
Global dense ranking (Top-203 full order)
  ↓ 
★ FIRST DIVERGENCE POINT ★: Top-20 candidate pool construction
  Phase 10C: Multi-Query Dense Subqueries (3 per Q) + BM25 Union (Pool size 30-66)
  Phase 11B B3: Single Primary Query Dense Top-20 Slicing (Pool size exactly 20)
```

---

## 8. Root Cause Classification

> **Root Cause**: `CANDIDATE_POOL_MISMATCH`

**Explanation**:
The discrepancy between Phase 10C and Phase 11B B3 $N=20$ is NOT due to changes in embedding weights, similarity metrics, chunking boundaries, Cross-Encoder scoring, or evaluation metric definitions.

It is caused entirely by **candidate pool construction mechanics**:
- **Phase 10C** evaluated a **multi-query subquery + BM25 hybrid fusion candidate pool**, which gathered 30–66 candidates per question at $N=20$ subquery depth.
- **Phase 11B B3** evaluated a **single-query dense depth curve**, slicing exact candidate depths $N \in \{10, 20, 30, 40, 50, 75, 100, 150, 203\}$ from a single primary query dense ranking.

---

## 9. Assertion #10 Decision

> **Assertion #10 Decision**: `UNRESOLVED` / `FAIL` (Baseline Reproduction Equivalence)

- **Justification**: The Phase 11B B3 $N=20$ condition evaluates single-query dense candidate slicing, which differs structurally from the Phase 10C multi-query hybrid candidate pool. Therefore, Phase 11B B3 $N=20$ cannot be claimed to reproduce the Phase 10C frozen baseline control.
- **Assertion Audit Update**: Assertion #10 status is changed from `PASS` to `UNRESOLVED` / `FAIL` with respect to frozen Phase 10C equivalence.

---

## 10. Validity of Phase 11B B3 Depth Results

- **For Single-Query Dense Retrieval Scaling**: **VALID**. The B3 depth curve accurately measures the performance of candidate depth expansion under single-query dense retrieval.
- **As a Continuation of Phase 10C Baseline**: **INVALID**. The B3 depth curve must NOT be interpreted as a continuation of Phase 10C's multi-query fusion baseline.

---

## 11. Post-Hoc Diagnostic Targets C15 and C42 Verification

Both target autopsies remain empirically verified and valid under post-hoc inspection:

```text
Target C15 (Q03, Primary Evidence):
- Single-Query Dense Rank: #142 (Similarity: 0.2231)
- Candidate Accessibility: Enters candidate pool at N=150
- Cross-Encoder Behavior: When present in candidate pool at N=150/200, CE score is -8.12 (placed outside final Top-15)
- Stage Classification: DEEP-CANDIDATE-RERANKING_BEHAVIOR

Target C42 (Q05, Primary Evidence):
- Single-Query Dense Rank: #99 (Similarity: 0.1858)
- Candidate Accessibility: Enters candidate pool at N=100
- Cross-Encoder Behavior: When present in candidate pool at N=100/150/200, CE score is -6.45 (placed outside final Top-15)
- Stage Classification: DEEP-CANDIDATE-RERANKING_BEHAVIOR
```

---

## 12. Exact Files and Artifacts Used

1. `evaluation/run_phase10d_candidate_representation_optimization.py`
2. `evaluation/results/phase10d_canonical_results.json`
3. `evaluation/run_phase11b_structural_fragmentation.py`
4. `evaluation/results/phase11b_structural_results.json`
5. `evaluation/results/phase11b_target_autopsy.json`
6. `Documents/The Comedy of Errors document.pdf` (canonical 203 chunks)
