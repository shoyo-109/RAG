# Phase D — B3 Representation Discoverability Diagnostic Final Report

> **Research Objective**: Determine whether structured $B_3$ representation makes previously inaccessible evidence discoverable through an auxiliary candidate-generation path at controlled depths $N \in \{20, 50, 100\}$ while canonical $B_0$/BM25 retrieval, $R_0$ reranker, and final Top-15 mechanism remain completely frozen.  
> **Status**: **COMPLETE & CANONICAL**  
> **Date**: 2026-10-07  
> **Control Reproduction Gate**: **PASS** (Candidate Recall@20 = 85.0%, Final Recall@15 = 70.0%, nDCG@10 = 0.485, MRR = 0.369)  
> **Verdict**: **`PHASE D VERDICT: [DIAGNOSTIC SUCCESS — DOWNSTREAM BOTTLENECK]`**

---

## 1. Executive Summary
Phase D evaluated auxiliary $B_3$ candidate retrieval at depths $N \in \{20, 50, 100\}$ across all 20 gold evidence units over the 203-chunk *The Comedy of Errors* corpus. Measured execution metrics determined whether $B_3$ candidate generation succeeded in introducing previously inaccessible evidence into the candidate pool and whether unchanged $R_0$ promoted those candidates downstream.

---

## 2. Research Question
> Can the structured $B_3$ representation make previously inaccessible evidence discoverable through an auxiliary candidate-generation path, at controlled depths $N \in \{20, 50, 100\}$, while the canonical $B_0$/BM25 retrieval path, $R_0$ reranker, and final Top-15 mechanism remain completely frozen?

---

## 3. Frozen Control Specification
- **Candidate Pool**: 3 frozen Q2 subqueries ($E_0$ dense Top-20 per subquery) + BM25 Top-20 on primary question.
- **Reranker ($R_0$)**: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- **Selection**: Top-15 final retrieved chunks.

---

## 4. Control Reproduction Results ($D_0$)

| Metric | Canonical Target | Phase D Control ($D_0$) | Delta | Gate Status |
| :--- | :---: | :---: | :---: | :---: |
| **Candidate Recall@20** | **85.0%** | **85.0%** | `+0.0%` | `PASS` |
| **Final Recall@15** | **70.0%** | **70.0%** | `+0.0%` | `PASS` |
| **Macro nDCG@10** | **0.485** | **0.485** | `-0.000` | `PASS` |
| **MRR** | **0.369** | **0.369** | `-0.000` | `PASS` |

---

## 5. Static Audit Results
All pre-execution audit assertions passed successfully before diagnostic execution.

---

## 6. Experimental Configurations
- **$D_0$**: Canonical Baseline Control (Auxiliary $B_3$ disabled).
- **$D_1$**: Canonical + $B_3$ Top-20 × 3 subqueries.
- **$D_2$**: Canonical + $B_3$ Top-50 × 3 subqueries.
- **$D_3$**: Canonical + $B_3$ Top-100 × 3 subqueries.

---

## 7. B3 Retrieval Method
- Indexed deterministic $B_3$ structured text using frozen $E_0$ (`sentence-transformers/all-MiniLM-L6-v2`).
- Retrieved Top-$N$ candidates across the 3 frozen Q2 subqueries per question.

---

## 8. Aggregate Results

| Config | B3 Depth | Candidate Recall (Augmented Pool) | Final Recall@15 | Macro nDCG@10 | MRR | B3 Candidates Added | Pollution Rate (Diagnostic Keyword Heuristic) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$D_0$** | Disabled | **85.0%** (Recall@20) | **70.0%** | **0.485** | **0.369** | **0** | **0.0%** |
| **D1** | Top-20 × 3 | **100.0%** (`+15.0%`) | **60.0%** (`-10.0%`) | **0.468** (`-0.017`) | **0.365** (`-0.004`) | **104** | **97.1%** |
| **D2** | Top-50 × 3 | **100.0%** (`+15.0%`) | **60.0%** (`-10.0%`) | **0.468** (`-0.017`) | **0.364** (`-0.005`) | **290** | **99.0%** |
| **D3** | Top-100 × 3 | **100.0%** (`+15.0%`) | **60.0%** (`-10.0%`) | **0.468** (`-0.017`) | **0.364** (`-0.005`) | **515** | **99.4%** |

---

## 9. Per-Question Results

| Question | Config | Candidate Pool Size | Candidate Recall | Final Recall@15 | Macro nDCG@10 | MRR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Q01** | D1 | 77 | 100.0% | 100.0% | 0.498 | 0.111 |
| **Q02** | D1 | 73 | 100.0% | 75.0% | 0.651 | 1.000 |
| **Q03** | D1 | 77 | 100.0% | 25.0% | 0.356 | 0.167 |
| **Q04** | D1 | 76 | 100.0% | 66.7% | 0.568 | 0.500 |
| **Q05** | D1 | 69 | 100.0% | 33.3% | 0.198 | 0.077 |
| **Q06** | D1 | 63 | 100.0% | 75.0% | 0.540 | 0.333 |
| **Q01** | D2 | 112 | 100.0% | 100.0% | 0.498 | 0.111 |
| **Q02** | D2 | 112 | 100.0% | 75.0% | 0.651 | 1.000 |
| **Q03** | D2 | 107 | 100.0% | 25.0% | 0.356 | 0.167 |
| **Q04** | D2 | 103 | 100.0% | 66.7% | 0.568 | 0.500 |
| **Q05** | D2 | 95 | 100.0% | 33.3% | 0.198 | 0.071 |
| **Q06** | D2 | 92 | 100.0% | 75.0% | 0.540 | 0.333 |
| **Q01** | D3 | 149 | 100.0% | 100.0% | 0.498 | 0.111 |
| **Q02** | D3 | 140 | 100.0% | 75.0% | 0.651 | 1.000 |
| **Q03** | D3 | 137 | 100.0% | 25.0% | 0.356 | 0.167 |
| **Q04** | D3 | 146 | 100.0% | 66.7% | 0.568 | 0.500 |
| **Q05** | D3 | 121 | 100.0% | 33.3% | 0.198 | 0.071 |
| **Q06** | D3 | 153 | 100.0% | 75.0% | 0.540 | 0.333 |

---

## 10. Evidence-Level Provenance

Summary of 20 gold evidence units across $D_0 \dots D_3$:
- **Canonical Candidate Accessible**: 17 / 20
- **Newly Recovered into Candidate Union ($D_3$)**: 3 / 3
- **Newly Recovered into Final Top-15 ($D_3$)**: 0 / 3

---

## 11. Candidate Pollution Analysis (Diagnostic Keyword Heuristic: Non-Canonical)

| Config | B3 Depth | Total B3 Retrieved | Duplicates of D0 | Added Candidates | Added Primary | Added Supporting | Added Relevant | Added Irrelevant | Pollution Rate (Non-Canonical Heuristic) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **D1** | Top-20 | 252 | 148 | 104 | 2 | 1 | 100 | 1 | 97.1% |
| **D2** | Top-50 | 555 | 265 | 290 | 2 | 1 | 284 | 3 | 99.0% |
| **D3** | Top-100 | 828 | 313 | 515 | 2 | 1 | 485 | 27 | 99.4% |

---

## 12. R0 Promotion Analysis

- **$B_3 \rightarrow \text{Candidate Union}$**: **3** evidence units newly introduced into candidate union.
- **$\text{Candidate Union} \rightarrow R_0 \text{ Top-15}$**: **0** evidence units promoted into $R_0$ Top-15.
- **$R_0 \text{ Top-15} \rightarrow \text{Final Top-15}$**: **0** evidence units retained in final Top-15.

---

## 13. Analysis of Target Autopsy Chunks

### Target Autopsy: `C15` (Q03)
- **Canonical $B_0$ Rank**: `#97` (Candidate: `False`, Final: `False`)
- **$B_3$ Best Subquery Rank**: `#6`
- **$D_1$ ($B_3@20$)**: Entered Union = `True`, $R_0$ Rank = `#60`, Final Top-15 = `False`
- **$D_2$ ($B_3@50$)**: Entered Union = `True`, $R_0$ Rank = `#73`, Final Top-15 = `False`
- **$D_3$ ($B_3@100$)**: Entered Union = `True`, $R_0$ Rank = `#80`, Final Top-15 = `False`
- **Phase D Diagnosis**: `B3_DISCOVERABILITY_R0_FAILURE`

### Target Autopsy: `C29` (Q03)
- **Canonical $B_0$ Rank**: `#21` (Candidate: `False`, Final: `False`)
- **$B_3$ Best Subquery Rank**: `#15`
- **$D_1$ ($B_3@20$)**: Entered Union = `True`, $R_0$ Rank = `#51`, Final Top-15 = `False`
- **$D_2$ ($B_3@50$)**: Entered Union = `True`, $R_0$ Rank = `#62`, Final Top-15 = `False`
- **$D_3$ ($B_3@100$)**: Entered Union = `True`, $R_0$ Rank = `#67`, Final Top-15 = `False`
- **Phase D Diagnosis**: `B3_DISCOVERABILITY_R0_FAILURE`

### Target Autopsy: `C42` (Q05)
- **Canonical $B_0$ Rank**: `#71` (Candidate: `False`, Final: `False`)
- **$B_3$ Best Subquery Rank**: `#14`
- **$D_1$ ($B_3@20$)**: Entered Union = `True`, $R_0$ Rank = `#42`, Final Top-15 = `False`
- **$D_2$ ($B_3@50$)**: Entered Union = `True`, $R_0$ Rank = `#46`, Final Top-15 = `False`
- **$D_3$ ($B_3@100$)**: Entered Union = `True`, $R_0$ Rank = `#46`, Final Top-15 = `False`
- **Phase D Diagnosis**: `B3_DISCOVERABILITY_R0_FAILURE`

### Target Autopsy: `C12` (Q02)
- **Canonical $B_0$ Rank**: `#13` (Candidate: `True`, Final: `False`)
- **$B_3$ Best Subquery Rank**: `#11`
- **$D_1$ ($B_3@20$)**: Entered Union = `True`, $R_0$ Rank = `#48`, Final Top-15 = `False`
- **$D_2$ ($B_3@50$)**: Entered Union = `True`, $R_0$ Rank = `#72`, Final Top-15 = `False`
- **$D_3$ ($B_3@100$)**: Entered Union = `True`, $R_0$ Rank = `#76`, Final Top-15 = `False`
- **Phase D Diagnosis**: `B3_DISCOVERABILITY_R0_FAILURE`

### Target Autopsy: `C65` (Q05)
- **Canonical $B_0$ Rank**: `#1` (Candidate: `True`, Final: `False`)
- **$B_3$ Best Subquery Rank**: `#24`
- **$D_1$ ($B_3@20$)**: Entered Union = `True`, $R_0$ Rank = `#47`, Final Top-15 = `False`
- **$D_2$ ($B_3@50$)**: Entered Union = `True`, $R_0$ Rank = `#54`, Final Top-15 = `False`
- **$D_3$ ($B_3@100$)**: Entered Union = `True`, $R_0$ Rank = `#58`, Final Top-15 = `False`
- **Phase D Diagnosis**: `B3_DISCOVERABILITY_R0_FAILURE`

### Target Autopsy: `C155` (Q06)
- **Canonical $B_0$ Rank**: `#6` (Candidate: `True`, Final: `False`)
- **$B_3$ Best Subquery Rank**: `#13`
- **$D_1$ ($B_3@20$)**: Entered Union = `True`, $R_0$ Rank = `#62`, Final Top-15 = `False`
- **$D_2$ ($B_3@50$)**: Entered Union = `True`, $R_0$ Rank = `#91`, Final Top-15 = `False`
- **$D_3$ ($B_3@100$)**: Entered Union = `True`, $R_0$ Rank = `#152`, Final Top-15 = `False`
- **Phase D Diagnosis**: `B3_DISCOVERABILITY_R0_FAILURE`

---

## 14. D0 vs D1 vs D2 vs D3 Comparison

- **Candidate Recall**: $D_0 = 85.0\%$, $D_3 = 100.0\%$ (`+15.0%` gain).
- **Final Recall@15**: $D_0 = 70.0\%$, $D_3 = 60.0\%$ (`-10.0%` change).
- **Macro nDCG@10**: $D_0 = 0.485$, $D_3 = 0.468$.
- **MRR**: $D_0 = 0.369$, $D_3 = 0.364$.

---

## 15. Accessibility Depth Analysis
Evaluated candidate accessibility across shallow ($D_1=20$, $D_2=50$) and deep ($D_3=100$) $B_3$ candidate retrieval budgets.

---

## 16. Failure-Stage Classification
Measured data isolates whether accessibility failures occur during candidate generation or downstream Cross-Encoder reranking.

---

## 17. Limitations
- Benchmark size: 6 questions over 203 corpus chunks.
- Reranker: $R_0$ (`ms-marco-MiniLM-L-6-v2`) is un-tuned for structured semantic text schemas.

---

## 18. Production Implication
Retain canonical $D_0$ in production unless verdict is PASS.

---

## 19. Final Verdict

```text
PHASE D VERDICT: [DIAGNOSTIC SUCCESS — DOWNSTREAM BOTTLENECK]
```

---

## 20. Recommended Next Controlled Experiment
Phase E — Evaluate Cross-Encoder rerankers ($R_1 / R_2$) over $B_3$ structured candidate representations.

---

## 21. Explicit Answers to Predeclared Research Questions (Q1–Q12)

### Q1: Does $B_3$ improve candidate accessibility relative to the frozen $D_0$ representation?
Yes. Candidate Recall increased by +15.0 percentage points.

### Q2: What is the minimum $B_3$ depth that produces a meaningful accessibility improvement?
D1 (B3@20)

### Q3: Does $B_3@20$, $B_3@50$, or $B_3@100$ recover `C15`?
Recovered into candidate union

### Q4: Does $B_3@20$, $B_3@50$, or $B_3@100$ recover `C42`?
Recovered into candidate union

### Q5: What happens to `C29`?
Retrieved in B3 space

### Q6: Do `C12`, `C65`, and `C155` demonstrate that adequate representation can still fail downstream?
Yes. Among the predeclared diagnostic targets, C12, C155, C65 were candidate-accessible under D0 but not selected in final Top-15 by R0.

### Q7: How many previously inaccessible evidence units become candidates?
3 evidence units.

### Q8: How many of those candidates survive $R_0$ into Top-15?
0 evidence units.

### Q9: Does increased $B_3$ depth improve final retrieval or merely increase candidate-pool size?
Increases candidate recall while final recall is constrained downstream by R0.

### Q10: Does $B_3$ introduce measurable candidate pollution?
Measured candidate pollution rate is 99.4% under D3 (diagnostic keyword heuristic: non-canonical, does not gate verdict).

### Q11: At what stage are the remaining failures concentrated?
Downstream R0 Cross-Encoder Reranking Stage.

### Q12: Does the evidence justify a production modification?
Partially. B3 improves candidate accessibility, but unchanged R0 does not convert the recovered candidates into final Top-15 evidence. This supports a downstream reranking bottleneck diagnosis rather than production deployment of B3 alone.
