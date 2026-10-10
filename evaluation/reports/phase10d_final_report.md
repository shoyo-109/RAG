# Phase 10D — Candidate Generation & Representation Optimization Final Report

> **Official Dataset & Integrity Audit Report**  
> **Status**: **CANONICAL & FROZEN**  
> **Date**: 2026-10-05  
> **Gloss Cache MD5**: `ee4243c2c0b575c87fe11a9f73f81de3`  
> **BM25 Corpus**: Frozen on `B0/raw_text` across all 60 runs  
> **Total Evaluated Runs**: 60  

---

## 1. Objective

The objective of Phase 10D was to investigate whether candidate depth expansion ($N \in \{10, 20, 30, 40, 50\}$), document representation enhancements ($B0 \dots B3$), or bi-encoder embedding model selection ($E0, E1, E2$) can recover Stage-1 missing evidence chunks **$C15$** and **$C42$** into the candidate pool and final Top-15 without degrading retrieval quality on the complete $Q01 \dots Q06$ benchmark.

---

## 2. Frozen Baseline Control Comparison

All Phase 10D configurations are evaluated against the frozen Phase 10C control baseline (`E0 + B0 + N20 + K15`):

- **Phase 10C Control Baseline (`E0+B0+N20+K15`)**:
  - Candidate Recall@20: **85.0%**
  - Final Recall@15: **60.0%** (12/20 gold evidence units)
  - Macro nDCG@10: **0.485**
  - Sentinel $C147$ Status: **RECOVERED** (Top-15)

> **Methodology Note on nDCG Metric Alignment**: Previous Phase 10D report drafts displayed nDCG evaluated at cutoff 15 (`nDCG@15`). To ensure a strict, like-for-like comparison against the frozen Phase 10C baseline cutoff (`nDCG@10 = 0.485`), all selection-gate evaluations and report metrics in this canonical artifact compute **`nDCG@10`** (evaluated at rank cutoff 10). Both `nDCG@10` and `nDCG@15` are recorded in the canonical JSON results.

---

## 3. Experimental Matrix Overview

Evaluated **60 controlled execution runs** across:
- **3 Embedding Models**: $E0$ (`all-MiniLM-L6-v2`), $E1$ (`bge-small-en-v1.5`), $E2$ (`e5-small-v2`)
- **4 Document Representations**: $B0$ (`raw_text`), $B1$ (`meta_raw`), $B2$ (`gloss_raw`), $B3$ (`meta_gloss_raw`)
- **5 Candidate Depths**: $N \in \{10, 20, 30, 40, 50\}$
- **Frozen Reranker Stage**: $R0$ (`ms-marco-MiniLM-L-6-v2`) at $K=15$

---

## 4. Complete 60-Run Selection Gate Audit Table vs Control (`E0+B0+N20+K15`)

| Configuration ID | Embedder | Rep ID | Candidate Depth N | Candidate Rec@N | Final Rec@15 | nDCG@10 | C147 Status | CandRec Gate | FinRec Gate | nDCG Gate | C147 Gate | Target Obj Gate | Overall Gate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`E0_B0_N10`** | E0 | B0 | N=10 | 75.0% | 65.0% | 0.529 | RECOVERED | FAIL | PASS | PASS | PASS | FAIL | **FAIL** |
| **`E0_B0_N20`** | E0 | B0 | N=20 | 85.0% | 70.0% | 0.485 | RECOVERED | PASS | PASS | FAIL | PASS | FAIL | **FAIL** |
| **`E0_B0_N30`** | E0 | B0 | N=30 | 90.0% | 65.0% | 0.468 | RECOVERED | PASS | PASS | FAIL | PASS | FAIL | **FAIL** |
| **`E0_B0_N40`** | E0 | B0 | N=40 | 90.0% | 65.0% | 0.468 | RECOVERED | PASS | PASS | FAIL | PASS | FAIL | **FAIL** |
| **`E0_B0_N50`** | E0 | B0 | N=50 | 90.0% | 60.0% | 0.468 | RECOVERED | PASS | PASS | FAIL | PASS | FAIL | **FAIL** |
| **`E0_B1_N10`** | E0 | B1 | N=10 | 80.0% | 60.0% | 0.554 | RECOVERED | FAIL | PASS | PASS | PASS | FAIL | **FAIL** |
| **`E0_B1_N20`** | E0 | B1 | N=20 | 85.0% | 60.0% | 0.508 | RECOVERED | PASS | PASS | PASS | PASS | FAIL | **FAIL** |
| **`E0_B1_N30`** | E0 | B1 | N=30 | 95.0% | 65.0% | 0.519 | RECOVERED | PASS | PASS | PASS | PASS | FAIL | **FAIL** |
| **`E0_B1_N40`** | E0 | B1 | N=40 | 100.0% | 65.0% | 0.519 | RECOVERED | PASS | PASS | PASS | PASS | FAIL | **FAIL** |
| **`E0_B1_N50`** | E0 | B1 | N=50 | 100.0% | 65.0% | 0.518 | RECOVERED | PASS | PASS | PASS | PASS | FAIL | **FAIL** |
| **`E0_B2_N10`** | E0 | B2 | N=10 | 80.0% | 65.0% | 0.608 | RECOVERED | FAIL | PASS | PASS | PASS | FAIL | **FAIL** |
| **`E0_B2_N20`** | E0 | B2 | N=20 | 90.0% | 70.0% | 0.604 | RECOVERED | PASS | PASS | PASS | PASS | FAIL | **FAIL** |
| **`E0_B2_N30`** | E0 | B2 | N=30 | 90.0% | 70.0% | 0.603 | RECOVERED | PASS | PASS | PASS | PASS | FAIL | **FAIL** |
| **`E0_B2_N40`** | E0 | B2 | N=40 | 90.0% | 70.0% | 0.603 | RECOVERED | PASS | PASS | PASS | PASS | FAIL | **FAIL** |
| **`E0_B2_N50`** | E0 | B2 | N=50 | 90.0% | 70.0% | 0.603 | RECOVERED | PASS | PASS | PASS | PASS | FAIL | **FAIL** |
| **`E0_B3_N10`** | E0 | B3 | N=10 | 80.0% | 60.0% | 0.652 | RECOVERED | FAIL | PASS | PASS | PASS | FAIL | **FAIL** |
| **`E0_B3_N20`** | E0 | B3 | N=20 | 90.0% | 65.0% | 0.644 | RECOVERED | PASS | PASS | PASS | PASS | FAIL | **FAIL** |
| **`E0_B3_N30`** | E0 | B3 | N=30 | 95.0% | 65.0% | 0.619 | RECOVERED | PASS | PASS | PASS | PASS | FAIL | **FAIL** |
| **`E0_B3_N40`** | E0 | B3 | N=40 | 95.0% | 65.0% | 0.619 | RECOVERED | PASS | PASS | PASS | PASS | FAIL | **FAIL** |
| **`E0_B3_N50`** | E0 | B3 | N=50 | 95.0% | 65.0% | 0.619 | RECOVERED | PASS | PASS | PASS | PASS | FAIL | **FAIL** |
| **`E1_B0_N10`** | E1 | B0 | N=10 | 75.0% | 65.0% | 0.524 | REGRESSED | FAIL | PASS | PASS | FAIL | FAIL | **FAIL** |
| **`E1_B0_N20`** | E1 | B0 | N=20 | 85.0% | 70.0% | 0.490 | RECOVERED | PASS | PASS | PASS | PASS | FAIL | **FAIL** |
| **`E1_B0_N30`** | E1 | B0 | N=30 | 85.0% | 60.0% | 0.485 | RECOVERED | PASS | PASS | PASS | PASS | FAIL | **FAIL** |
| **`E1_B0_N40`** | E1 | B0 | N=40 | 90.0% | 60.0% | 0.473 | RECOVERED | PASS | PASS | FAIL | PASS | FAIL | **FAIL** |
| **`E1_B0_N50`** | E1 | B0 | N=50 | 90.0% | 60.0% | 0.471 | RECOVERED | PASS | PASS | FAIL | PASS | FAIL | **FAIL** |
| **`E1_B1_N10`** | E1 | B1 | N=10 | 85.0% | 65.0% | 0.551 | RECOVERED | PASS | PASS | PASS | PASS | FAIL | **FAIL** |
| **`E1_B1_N20`** | E1 | B1 | N=20 | 95.0% | 65.0% | 0.526 | RECOVERED | PASS | PASS | PASS | PASS | FAIL | **FAIL** |
| **`E1_B1_N30`** | E1 | B1 | N=30 | 100.0% | 65.0% | 0.518 | RECOVERED | PASS | PASS | PASS | PASS | FAIL | **FAIL** |
| **`E1_B1_N40`** | E1 | B1 | N=40 | 100.0% | 65.0% | 0.518 | RECOVERED | PASS | PASS | PASS | PASS | FAIL | **FAIL** |
| **`E1_B1_N50`** | E1 | B1 | N=50 | 100.0% | 65.0% | 0.518 | RECOVERED | PASS | PASS | PASS | PASS | FAIL | **FAIL** |
| **`E1_B2_N10`** | E1 | B2 | N=10 | 75.0% | 65.0% | 0.614 | REGRESSED | FAIL | PASS | PASS | FAIL | FAIL | **FAIL** |
| **`E1_B2_N20`** | E1 | B2 | N=20 | 85.0% | 70.0% | 0.600 | RECOVERED | PASS | PASS | PASS | PASS | FAIL | **FAIL** |
| **`E1_B2_N30`** | E1 | B2 | N=30 | 85.0% | 70.0% | 0.605 | RECOVERED | PASS | PASS | PASS | PASS | FAIL | **FAIL** |
| **`E1_B2_N40`** | E1 | B2 | N=40 | 85.0% | 70.0% | 0.605 | RECOVERED | PASS | PASS | PASS | PASS | FAIL | **FAIL** |
| **`E1_B2_N50`** | E1 | B2 | N=50 | 90.0% | 70.0% | 0.605 | RECOVERED | PASS | PASS | PASS | PASS | FAIL | **FAIL** |
| **`E1_B3_N10`** | E1 | B3 | N=10 | 70.0% | 60.0% | 0.653 | REGRESSED | FAIL | PASS | PASS | FAIL | FAIL | **FAIL** |
| **`E1_B3_N20`** | E1 | B3 | N=20 | 85.0% | 60.0% | 0.656 | REGRESSED | PASS | PASS | PASS | FAIL | FAIL | **FAIL** |
| **`E1_B3_N30`** | E1 | B3 | N=30 | 100.0% | 65.0% | 0.644 | RECOVERED | PASS | PASS | PASS | PASS | FAIL | **FAIL** |
| **`E1_B3_N40`** | E1 | B3 | N=40 | 100.0% | 65.0% | 0.619 | RECOVERED | PASS | PASS | PASS | PASS | FAIL | **FAIL** |
| **`E1_B3_N50`** | E1 | B3 | N=50 | 100.0% | 65.0% | 0.619 | RECOVERED | PASS | PASS | PASS | PASS | FAIL | **FAIL** |
| **`E2_B0_N10`** | E2 | B0 | N=10 | 85.0% | 70.0% | 0.527 | RECOVERED | PASS | PASS | PASS | PASS | FAIL | **FAIL** |
| **`E2_B0_N20`** | E2 | B0 | N=20 | 85.0% | 70.0% | 0.476 | RECOVERED | PASS | PASS | FAIL | PASS | FAIL | **FAIL** |
| **`E2_B0_N30`** | E2 | B0 | N=30 | 95.0% | 60.0% | 0.468 | RECOVERED | PASS | PASS | FAIL | PASS | FAIL | **FAIL** |
| **`E2_B0_N40`** | E2 | B0 | N=40 | 95.0% | 60.0% | 0.468 | RECOVERED | PASS | PASS | FAIL | PASS | FAIL | **FAIL** |
| **`E2_B0_N50`** | E2 | B0 | N=50 | 95.0% | 60.0% | 0.468 | RECOVERED | PASS | PASS | FAIL | PASS | FAIL | **FAIL** |
| **`E2_B1_N10`** | E2 | B1 | N=10 | 95.0% | 70.0% | 0.522 | RECOVERED | PASS | PASS | PASS | PASS | FAIL | **FAIL** |
| **`E2_B1_N20`** | E2 | B1 | N=20 | 95.0% | 65.0% | 0.518 | RECOVERED | PASS | PASS | PASS | PASS | FAIL | **FAIL** |
| **`E2_B1_N30`** | E2 | B1 | N=30 | 95.0% | 65.0% | 0.518 | RECOVERED | PASS | PASS | PASS | PASS | FAIL | **FAIL** |
| **`E2_B1_N40`** | E2 | B1 | N=40 | 95.0% | 65.0% | 0.518 | RECOVERED | PASS | PASS | PASS | PASS | FAIL | **FAIL** |
| **`E2_B1_N50`** | E2 | B1 | N=50 | 95.0% | 65.0% | 0.518 | RECOVERED | PASS | PASS | PASS | PASS | FAIL | **FAIL** |
| **`E2_B2_N10`** | E2 | B2 | N=10 | 80.0% | 70.0% | 0.616 | RECOVERED | FAIL | PASS | PASS | PASS | FAIL | **FAIL** |
| **`E2_B2_N20`** | E2 | B2 | N=20 | 80.0% | 70.0% | 0.603 | RECOVERED | FAIL | PASS | PASS | PASS | FAIL | **FAIL** |
| **`E2_B2_N30`** | E2 | B2 | N=30 | 90.0% | 70.0% | 0.603 | RECOVERED | PASS | PASS | PASS | PASS | FAIL | **FAIL** |
| **`E2_B2_N40`** | E2 | B2 | N=40 | 95.0% | 70.0% | 0.603 | RECOVERED | PASS | PASS | PASS | PASS | FAIL | **FAIL** |
| **`E2_B2_N50`** | E2 | B2 | N=50 | 100.0% | 70.0% | 0.603 | RECOVERED | PASS | PASS | PASS | PASS | FAIL | **FAIL** |
| **`E2_B3_N10`** | E2 | B3 | N=10 | 85.0% | 65.0% | 0.669 | RECOVERED | PASS | PASS | PASS | PASS | FAIL | **FAIL** |
| **`E2_B3_N20`** | E2 | B3 | N=20 | 95.0% | 65.0% | 0.640 | RECOVERED | PASS | PASS | PASS | PASS | FAIL | **FAIL** |
| **`E2_B3_N30`** | E2 | B3 | N=30 | 95.0% | 65.0% | 0.615 | RECOVERED | PASS | PASS | PASS | PASS | FAIL | **FAIL** |
| **`E2_B3_N40`** | E2 | B3 | N=40 | 95.0% | 65.0% | 0.619 | RECOVERED | PASS | PASS | PASS | PASS | FAIL | **FAIL** |
| **`E2_B3_N50`** | E2 | B3 | N=50 | 100.0% | 65.0% | 0.619 | RECOVERED | PASS | PASS | PASS | PASS | FAIL | **FAIL** |

### Gate Audit Summary

- **Total Evaluated Configurations**: 60
- **Candidate Recall Gate PASS ($\ge 85.0\%$)**: 51/60
- **Final Recall Gate PASS ($\ge 60.0\%$)**: 60/60
- **nDCG@10 Gate PASS ($\ge 0.485$)**: 50/60
- **Sentinel $C147$ Recovery Gate PASS (Top-15)**: 56/60
- **Target Evidence Recovery Objective PASS (C15 or C42 in Top-15)**: **0/60**
- **OVERALL SELECTION GATE PASS**: **0/60**

---

## 5. Candidate Depth Results (Experiment A: E0 x [B0, B2])

| Rep ID | Representation | Depth N | Candidate Rec@N | Final Rec@15 | Ev Prec@15 | Pri Prec@15 | MRR | nDCG@10 | Ev Comp | Conversion | Latency |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0** | raw_text | N=10 | 75.0% | **65.0%** | 0.144 | 0.067 | 0.385 | 0.529 | 33.3% | 86.7% | 7782.4ms |
| **B0** | raw_text | N=20 | 85.0% | **70.0%** | 0.156 | 0.067 | 0.369 | 0.485 | 33.3% | 82.4% | 10399.4ms |
| **B0** | raw_text | N=30 | 90.0% | **65.0%** | 0.144 | 0.067 | 0.366 | 0.468 | 33.3% | 72.2% | 12550.2ms |
| **B0** | raw_text | N=40 | 90.0% | **65.0%** | 0.144 | 0.067 | 0.365 | 0.468 | 33.3% | 72.2% | 14454.5ms |
| **B0** | raw_text | N=50 | 90.0% | **60.0%** | 0.133 | 0.067 | 0.364 | 0.468 | 16.7% | 66.7% | 15842.6ms |
| **B2** | gloss_raw | N=10 | 80.0% | **65.0%** | 0.144 | 0.067 | 0.492 | 0.608 | 33.3% | 81.2% | 7995.7ms |
| **B2** | gloss_raw | N=20 | 90.0% | **70.0%** | 0.156 | 0.067 | 0.525 | 0.604 | 33.3% | 77.8% | 12326.0ms* |
| **B2** | gloss_raw | N=30 | 90.0% | **70.0%** | 0.156 | 0.067 | 0.525 | 0.603 | 33.3% | 77.8% | 15294.4ms |
| **B2** | gloss_raw | N=40 | 90.0% | **70.0%** | 0.156 | 0.067 | 0.525 | 0.603 | 33.3% | 77.8% | 17642.9ms |
| **B2** | gloss_raw | N=50 | 90.0% | **70.0%** | 0.156 | 0.067 | 0.525 | 0.603 | 33.3% | 77.8% | 19498.9ms |

*\* Footnote: `E0_B2_N20` latency was impacted by a documented infrastructure network disconnection during execution. It is excluded from latency performance comparisons (see Section 11).*

---

## 6. Representation Results (Experiment B: E0 x [B0..B3])

| Rep ID | Description | Depth N | Candidate Rec@N | Final Rec@15 | Ev Prec@15 | MRR | nDCG@10 | Ev Comp | Conversion |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0** | raw_text | N=10 | 75.0% | **65.0%** | 0.144 | 0.385 | 0.529 | 33.3% | 86.7% |
| **B0** | raw_text | N=20 | 85.0% | **70.0%** | 0.156 | 0.369 | 0.485 | 33.3% | 82.4% |
| **B0** | raw_text | N=30 | 90.0% | **65.0%** | 0.144 | 0.366 | 0.468 | 33.3% | 72.2% |
| **B0** | raw_text | N=40 | 90.0% | **65.0%** | 0.144 | 0.365 | 0.468 | 33.3% | 72.2% |
| **B0** | raw_text | N=50 | 90.0% | **60.0%** | 0.133 | 0.364 | 0.468 | 16.7% | 66.7% |
| **B1** | meta_raw | N=10 | 80.0% | **60.0%** | 0.133 | 0.375 | 0.554 | 16.7% | 75.0% |
| **B1** | meta_raw | N=20 | 85.0% | **60.0%** | 0.133 | 0.360 | 0.508 | 16.7% | 70.6% |
| **B1** | meta_raw | N=30 | 95.0% | **65.0%** | 0.144 | 0.413 | 0.519 | 16.7% | 68.4% |
| **B1** | meta_raw | N=40 | 100.0% | **65.0%** | 0.144 | 0.413 | 0.519 | 16.7% | 65.0% |
| **B1** | meta_raw | N=50 | 100.0% | **65.0%** | 0.144 | 0.413 | 0.518 | 16.7% | 65.0% |
| **B2** | gloss_raw | N=10 | 80.0% | **65.0%** | 0.144 | 0.492 | 0.608 | 33.3% | 81.2% |
| **B2** | gloss_raw | N=20 | 90.0% | **70.0%** | 0.156 | 0.525 | 0.604 | 33.3% | 77.8% |
| **B2** | gloss_raw | N=30 | 90.0% | **70.0%** | 0.156 | 0.525 | 0.603 | 33.3% | 77.8% |
| **B2** | gloss_raw | N=40 | 90.0% | **70.0%** | 0.156 | 0.525 | 0.603 | 33.3% | 77.8% |
| **B2** | gloss_raw | N=50 | 90.0% | **70.0%** | 0.156 | 0.525 | 0.603 | 33.3% | 77.8% |
| **B3** | meta_gloss_raw | N=10 | 80.0% | **60.0%** | 0.133 | 0.607 | 0.652 | 33.3% | 75.0% |
| **B3** | meta_gloss_raw | N=20 | 90.0% | **65.0%** | 0.144 | 0.769 | 0.644 | 33.3% | 72.2% |
| **B3** | meta_gloss_raw | N=30 | 95.0% | **65.0%** | 0.144 | 0.764 | 0.619 | 33.3% | 68.4% |
| **B3** | meta_gloss_raw | N=40 | 95.0% | **65.0%** | 0.144 | 0.764 | 0.619 | 33.3% | 68.4% |
| **B3** | meta_gloss_raw | N=50 | 95.0% | **65.0%** | 0.144 | 0.764 | 0.619 | 33.3% | 68.4% |

---

## 7. Embedding Results (Experiment C: E1 & E2)

| Embedder | Model Identifier | Rep ID | Depth N | Candidate Rec@N | Final Rec@15 | Ev Prec@15 | MRR | nDCG@10 | Ev Comp |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **E1** | BAAI/bge-small-en-v1.5 | B0 | N=10 | 75.0% | **65.0%** | 0.144 | 0.371 | 0.524 | 33.3% |
| **E1** | BAAI/bge-small-en-v1.5 | B0 | N=20 | 85.0% | **70.0%** | 0.156 | 0.379 | 0.490 | 33.3% |
| **E1** | BAAI/bge-small-en-v1.5 | B0 | N=30 | 85.0% | **60.0%** | 0.133 | 0.369 | 0.485 | 16.7% |
| **E1** | BAAI/bge-small-en-v1.5 | B0 | N=40 | 90.0% | **60.0%** | 0.133 | 0.366 | 0.473 | 16.7% |
| **E1** | BAAI/bge-small-en-v1.5 | B0 | N=50 | 90.0% | **60.0%** | 0.133 | 0.366 | 0.471 | 16.7% |
| **E1** | BAAI/bge-small-en-v1.5 | B1 | N=10 | 85.0% | **65.0%** | 0.144 | 0.464 | 0.551 | 16.7% |
| **E1** | BAAI/bge-small-en-v1.5 | B1 | N=20 | 95.0% | **65.0%** | 0.144 | 0.424 | 0.526 | 16.7% |
| **E1** | BAAI/bge-small-en-v1.5 | B1 | N=30 | 100.0% | **65.0%** | 0.144 | 0.413 | 0.518 | 16.7% |
| **E1** | BAAI/bge-small-en-v1.5 | B1 | N=40 | 100.0% | **65.0%** | 0.144 | 0.413 | 0.518 | 16.7% |
| **E1** | BAAI/bge-small-en-v1.5 | B1 | N=50 | 100.0% | **65.0%** | 0.144 | 0.413 | 0.518 | 16.7% |
| **E1** | BAAI/bge-small-en-v1.5 | B2 | N=10 | 75.0% | **65.0%** | 0.144 | 0.575 | 0.614 | 33.3% |
| **E1** | BAAI/bge-small-en-v1.5 | B2 | N=20 | 85.0% | **70.0%** | 0.156 | 0.525 | 0.600 | 33.3% |
| **E1** | BAAI/bge-small-en-v1.5 | B2 | N=30 | 85.0% | **70.0%** | 0.156 | 0.525 | 0.605 | 33.3% |
| **E1** | BAAI/bge-small-en-v1.5 | B2 | N=40 | 85.0% | **70.0%** | 0.156 | 0.525 | 0.605 | 33.3% |
| **E1** | BAAI/bge-small-en-v1.5 | B2 | N=50 | 90.0% | **70.0%** | 0.156 | 0.525 | 0.605 | 33.3% |
| **E1** | BAAI/bge-small-en-v1.5 | B3 | N=10 | 70.0% | **60.0%** | 0.133 | 0.778 | 0.653 | 33.3% |
| **E1** | BAAI/bge-small-en-v1.5 | B3 | N=20 | 85.0% | **60.0%** | 0.133 | 0.767 | 0.656 | 33.3% |
| **E1** | BAAI/bge-small-en-v1.5 | B3 | N=30 | 100.0% | **65.0%** | 0.144 | 0.767 | 0.644 | 33.3% |
| **E1** | BAAI/bge-small-en-v1.5 | B3 | N=40 | 100.0% | **65.0%** | 0.144 | 0.764 | 0.619 | 33.3% |
| **E1** | BAAI/bge-small-en-v1.5 | B3 | N=50 | 100.0% | **65.0%** | 0.144 | 0.764 | 0.619 | 33.3% |
| **E2** | intfloat/e5-small-v2 | B0 | N=10 | 85.0% | **70.0%** | 0.156 | 0.388 | 0.527 | 33.3% |
| **E2** | intfloat/e5-small-v2 | B0 | N=20 | 85.0% | **70.0%** | 0.156 | 0.370 | 0.476 | 33.3% |
| **E2** | intfloat/e5-small-v2 | B0 | N=30 | 95.0% | **60.0%** | 0.133 | 0.364 | 0.468 | 16.7% |
| **E2** | intfloat/e5-small-v2 | B0 | N=40 | 95.0% | **60.0%** | 0.133 | 0.364 | 0.468 | 16.7% |
| **E2** | intfloat/e5-small-v2 | B0 | N=50 | 95.0% | **60.0%** | 0.133 | 0.364 | 0.468 | 16.7% |
| **E2** | intfloat/e5-small-v2 | B1 | N=10 | 95.0% | **70.0%** | 0.156 | 0.418 | 0.522 | 33.3% |
| **E2** | intfloat/e5-small-v2 | B1 | N=20 | 95.0% | **65.0%** | 0.144 | 0.413 | 0.518 | 16.7% |
| **E2** | intfloat/e5-small-v2 | B1 | N=30 | 95.0% | **65.0%** | 0.144 | 0.413 | 0.518 | 16.7% |
| **E2** | intfloat/e5-small-v2 | B1 | N=40 | 95.0% | **65.0%** | 0.144 | 0.413 | 0.518 | 16.7% |
| **E2** | intfloat/e5-small-v2 | B1 | N=50 | 95.0% | **65.0%** | 0.144 | 0.413 | 0.518 | 16.7% |
| **E2** | intfloat/e5-small-v2 | B2 | N=10 | 80.0% | **70.0%** | 0.156 | 0.547 | 0.616 | 33.3% |
| **E2** | intfloat/e5-small-v2 | B2 | N=20 | 80.0% | **70.0%** | 0.156 | 0.525 | 0.603 | 33.3% |
| **E2** | intfloat/e5-small-v2 | B2 | N=30 | 90.0% | **70.0%** | 0.156 | 0.525 | 0.603 | 33.3% |
| **E2** | intfloat/e5-small-v2 | B2 | N=40 | 95.0% | **70.0%** | 0.156 | 0.525 | 0.603 | 33.3% |
| **E2** | intfloat/e5-small-v2 | B2 | N=50 | 100.0% | **70.0%** | 0.156 | 0.525 | 0.603 | 33.3% |
| **E2** | intfloat/e5-small-v2 | B3 | N=10 | 85.0% | **65.0%** | 0.144 | 0.774 | 0.669 | 33.3% |
| **E2** | intfloat/e5-small-v2 | B3 | N=20 | 95.0% | **65.0%** | 0.144 | 0.767 | 0.640 | 33.3% |
| **E2** | intfloat/e5-small-v2 | B3 | N=30 | 95.0% | **65.0%** | 0.144 | 0.765 | 0.615 | 33.3% |
| **E2** | intfloat/e5-small-v2 | B3 | N=40 | 95.0% | **65.0%** | 0.144 | 0.764 | 0.619 | 33.3% |
| **E2** | intfloat/e5-small-v2 | B3 | N=50 | 100.0% | **65.0%** | 0.144 | 0.764 | 0.619 | 33.3% |

---

## 8. Target Autopsy Summary & Stage Classifications

Classified using the **N=50 endpoint** as the final diagnostic state:

- **Stage 1 — Candidate Generation Failure** (`candidate_at_N50 == False`): **10** targets
- **Stage 2 — Candidate Depth Bottleneck** (`candidate_at_N20 == False AND candidate_at_N50 == True`): **13** targets
- **Stage 3 — CE Reranking Failure** (`candidate_at_N50 == True AND final_rank_at_N50 > 15`): **20** targets
- **Stage 4 — Recovered** (`final_rank_at_N50 <= 15`): **17** targets

| Target | Question | Embedder | Rep ID | Dense Rank | BM25 Rank | N=20 Cand? | N=50 Cand? | Final Rank (N50) | Dynamic Failure Stage (N50 Endpoint) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **C15** | Q03 | E0 | B0 | #142 | #None | NO | NO | null | Stage 1 — Candidate Generation Failure |
| **C29** | Q03 | E0 | B0 | #60 | #None | NO | YES | null | Stage 2 — Candidate Depth Bottleneck |
| **C61** | Q03 | E0 | B0 | #33 | #7 | YES | YES | null | Stage 3 — CE Reranking Failure |
| **C42** | Q05 | E0 | B0 | #99 | #None | NO | NO | null | Stage 1 — Candidate Generation Failure |
| **C147** | Q06 | E0 | B0 | #6 | #None | YES | YES | 3 | Stage 4 — Recovered |
| **C15** | Q03 | E0 | B1 | #85 | #None | NO | YES | null | Stage 2 — Candidate Depth Bottleneck |
| **C29** | Q03 | E0 | B1 | #36 | #None | YES | YES | null | Stage 3 — CE Reranking Failure |
| **C61** | Q03 | E0 | B1 | #21 | #7 | YES | YES | 14 | Stage 4 — Recovered |
| **C42** | Q05 | E0 | B1 | #67 | #None | NO | YES | null | Stage 2 — Candidate Depth Bottleneck |
| **C147** | Q06 | E0 | B1 | #4 | #None | YES | YES | 7 | Stage 4 — Recovered |
| **C15** | Q03 | E0 | B2 | #118 | #None | NO | NO | null | Stage 1 — Candidate Generation Failure |
| **C29** | Q03 | E0 | B2 | #51 | #None | YES | YES | null | Stage 3 — CE Reranking Failure |
| **C61** | Q03 | E0 | B2 | #26 | #7 | YES | YES | 15 | Stage 4 — Recovered |
| **C42** | Q05 | E0 | B2 | #101 | #None | NO | NO | null | Stage 1 — Candidate Generation Failure |
| **C147** | Q06 | E0 | B2 | #6 | #None | YES | YES | 5 | Stage 4 — Recovered |

---

## 9. Dynamic Next-Intervention Decision

- **Recommendation**: **KEEP CURRENT (Baseline Phase 10C Control) — No evaluated configuration satisfied all gate criteria.**
- **Dynamic Next-Intervention Statement**:
  > The current retrieval-side interventions do not satisfy the Phase 10D selection gate. Phase 10D indicates that the current retrieval-side optimization space is insufficient for the defined selection gate (specifically, zero of 60 configurations recovered missing evidence chunks C15 or C42 into Top-15); based on that measured result, fine-tuning or targeted re-chunking is indicated as the subsequent intervention.

---

## 10. Phase 11 Handoff

Handoff Phase 11 based on measured canonical Phase 10D results. Because zero of the 60 evaluated retrieval-side configurations satisfied the Phase 10D selection gate (measured N=50 diagnostic state: 10 Stage-1 generation failures, 13 Stage-2 depth bottlenecks, 20 Stage-3 reranking failures, 17 Stage-4 recovered targets), Phase 11 must address the dominant Stage-1 candidate generation failure and Stage-3 cross-encoder selection limitations for hard targets C15 and C42.

---

## 11. Infrastructure & Runtime Latency Anomaly Note

> **Runtime Anomaly Statement**: The recorded execution latency for `E0_B2_N20` (~18,356,383 ms / ~5.1 hours) was caused by a documented network disconnection during model execution. This latency anomaly is preserved in raw data artifacts (`phase10d_canonical_results.json`, `phase10d_comparison.csv`) for provenance auditability, but is explicitly **excluded** from all performance comparisons, efficiency rankings, or algorithmic trade-off conclusions. It does NOT reflect algorithmic latency of representation $B2$ or candidate depth $N=20$, and has zero impact on retrieval accuracy (recalls, nDCG, or target autopsy classifications).
