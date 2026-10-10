# Phase E — Terminal Reranker Resolution Experiment Final Report

> **Research Objective**: Determine whether alternative rerankers ($R_1$: `bge-reranker-base`, $R_2$: `bge-reranker-large`) combined with raw ($B_0$) vs. structured ($B_3$) candidate representations can convert candidate-accessibility gains into improved final Top-15 retrieval without degrading baseline quality.  
> **Status**: **TERMINAL COMPLETE & CANONICAL**  
> **Date**: 2026-10-08  
> **Control Reproduction Gate**: **PASS** (Candidate Recall@20 = 85.0%, Final Recall@15 = 70.0%, nDCG@10 = 0.485, MRR = 0.369)  
> **Verdict**: **`PHASE E VERDICT: [DIAGNOSTIC SUCCESS — DOWNSTREAM RERANKING BOTTLENECK]`**

---

## 1. Executive Summary
Phase E evaluated alternative rerankers ($R_1, R_2$) across raw ($B_0$) and structured ($B_3$) chunk representations over the frozen 203-chunk *The Comedy of Errors* benchmark corpus using causally isolated controls ($E_0$ and $E_{0\_D1}$).

---

## 2. Research Question
> Can a reranker better aligned with the structured $B_3$ representation convert the candidate-accessibility gains demonstrated in Phase D into improved final Top-15 retrieval, while preserving or improving the canonical retrieval quality?

---

## 3. Frozen Experimental Contract
- **Corpus**: 203 chunks ($C_0$–$C_202$).
- **Benchmark**: Q01–Q06 (20 gold evidence units).
- **Subqueries**: 3 frozen $Q_2$ subqueries per question.
- **Candidate Pool**: Single augmented $D_1$ pool ($B_0$ Top-20 + BM25 Top-20 + $B_3$ Top-20) generated once and frozen across E0_D1 and E1–E4.
- **Rerankers**: $R_0$ (`ms-marco-MiniLM-L-6-v2`), $R_1$ (`bge-reranker-base`), $R_2$ (`bge-reranker-large`).

---

## 4. Phase D Context
Phase D established that $B_3$ auxiliary candidate retrieval achieves 100% candidate recall for previously inaccessible evidence units (`C15`, `C29`, `C42`), but $R_0$ failed to promote any into final Top-15.

---

## 5. Control Reproduction Gate ($E_0$)

| Metric | Target | Measured ($E_0$) | Gate Status |
| :--- | :---: | :---: | :---: |
| **Candidate Recall@20** | **85.0%** | **85.0%** | **PASS** |
| **Final Recall@15** | **70.0%** | **70.0%** | **PASS** |
| **Macro nDCG@10** | **0.485** | **0.485** | **PASS** |
| **MRR** | **0.369** | **0.369** | **PASS** |

---

## 6. Experimental Configurations
- **$E_0$**: Canonical Baseline Control ($R_0$ + $B_0$ on $D_0$ Pool).
- **$E_{0\_D1}$**: Candidate-Pool-Only Control ($R_0$ + $B_0$ on Augmented $D_1$ Pool).
- **$E_1$**: Augmented $D_1$ Pool + $R_1$ (`bge-reranker-base`) + $B_0$.
- **$E_2$**: Augmented $D_1$ Pool + $R_1$ (`bge-reranker-base`) + $B_3$.
- **$E_3$**: Augmented $D_1$ Pool + $R_2$ (`bge-reranker-large`) + $B_0$.
- **$E_4$**: Augmented $D_1$ Pool + $R_2$ (`bge-reranker-large`) + $B_3$.

---

## 7. Candidate-Pool Construction
Generated once per question and frozen across $E_{0\_D1}$ and E1–E4.

---

## 8. Reranking Method
Top-15 selected from reranker scores.

---

## 9. Aggregate Matrix Results

| Config | Candidate Pool | Reranker | Representation | Candidate Recall D1 | Final Recall@15 | Macro nDCG@10 | MRR | Delta Final Recall vs E0 | Delta Final Recall vs E0_D1 |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **E0** | Canonical D0 | R0 | B0 | **85.0%** | **70.0%** | **0.485** | **0.369** | `+0.0%` | `+10.0%` |
| **E0_D1** | D1 Augmented | R0 | B0 | **100.0%** | **60.0%** | **0.468** | **0.365** | `-10.0%` | `+0.0%` |
| **E1** | D1 Augmented | R1 | B0 | **100.0%** | **55.0%** | **0.527** | **0.496** | `-15.0%` | `-5.0%` |
| **E2** | D1 Augmented | R1 | B3 | **100.0%** | **55.0%** | **0.371** | **0.237** | `-15.0%` | `-5.0%` |
| **E3** | D1 Augmented | R2 | B0 | **100.0%** | **60.0%** | **0.544** | **0.383** | `-10.0%` | `+0.0%` |
| **E4** | D1 Augmented | R2 | B3 | **100.0%** | **50.0%** | **0.303** | **0.099** | `-20.0%` | `-10.0%` |

---

## 10. Per-Question Results

| Question | Config | Candidate Pool Size | Candidate Recall D1 | Final Recall@15 | Macro nDCG@10 | MRR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Q01** | E0_D1 | 77 | 100.0% | 100.0% | 0.498 | 0.111 |
| **Q02** | E0_D1 | 73 | 100.0% | 75.0% | 0.651 | 1.000 |
| **Q03** | E0_D1 | 77 | 100.0% | 25.0% | 0.356 | 0.167 |
| **Q04** | E0_D1 | 76 | 100.0% | 66.7% | 0.568 | 0.500 |
| **Q05** | E0_D1 | 69 | 100.0% | 33.3% | 0.198 | 0.077 |
| **Q06** | E0_D1 | 63 | 100.0% | 75.0% | 0.540 | 0.333 |
| **Q01** | E1 | 77 | 100.0% | 100.0% | 0.686 | 0.500 |
| **Q02** | E1 | 73 | 100.0% | 75.0% | 0.952 | 1.000 |
| **Q03** | E1 | 77 | 100.0% | 25.0% | 0.314 | 0.143 |
| **Q04** | E1 | 76 | 100.0% | 100.0% | 0.545 | 0.333 |
| **Q05** | E1 | 69 | 100.0% | 0.0% | 0.194 | 0.000 |
| **Q06** | E1 | 63 | 100.0% | 50.0% | 0.470 | 1.000 |
| **Q01** | E2 | 77 | 100.0% | 100.0% | 0.606 | 0.500 |
| **Q02** | E2 | 73 | 100.0% | 75.0% | 0.466 | 0.200 |
| **Q03** | E2 | 77 | 100.0% | 25.0% | 0.300 | 0.111 |
| **Q04** | E2 | 76 | 100.0% | 66.7% | 0.503 | 0.500 |
| **Q05** | E2 | 69 | 100.0% | 0.0% | 0.149 | 0.000 |
| **Q06** | E2 | 63 | 100.0% | 75.0% | 0.202 | 0.111 |
| **Q01** | E3 | 77 | 100.0% | 100.0% | 0.826 | 0.500 |
| **Q02** | E3 | 73 | 100.0% | 75.0% | 0.920 | 1.000 |
| **Q03** | E3 | 77 | 100.0% | 25.0% | 0.211 | 0.100 |
| **Q04** | E3 | 76 | 100.0% | 100.0% | 0.752 | 0.500 |
| **Q05** | E3 | 69 | 100.0% | 0.0% | 0.198 | 0.000 |
| **Q06** | E3 | 63 | 100.0% | 75.0% | 0.358 | 0.200 |
| **Q01** | E4 | 77 | 100.0% | 100.0% | 0.250 | 0.100 |
| **Q02** | E4 | 73 | 100.0% | 75.0% | 0.484 | 0.200 |
| **Q03** | E4 | 77 | 100.0% | 0.0% | 0.163 | 0.000 |
| **Q04** | E4 | 76 | 100.0% | 66.7% | 0.422 | 0.125 |
| **Q05** | E4 | 69 | 100.0% | 0.0% | 0.146 | 0.000 |
| **Q06** | E4 | 63 | 100.0% | 75.0% | 0.352 | 0.167 |

---

## 11. Evidence-Level Results & Failure-Stage Classification

Summary of gold evidence provenance across $E_0 \dots E_4$:

| Config | Candidate Gen Failure | Reranking Failure | Final Recovery Success |
| :--- | :---: | :---: | :---: |
| **E0** | 3 | 3 | **14** |
| **E0_D1** | 0 | 8 | **12** |
| **E1** | 0 | 9 | **11** |
| **E2** | 0 | 9 | **11** |
| **E3** | 0 | 8 | **12** |
| **E4** | 0 | 10 | **10** |


---

## 12. C15 / C29 / C42 Target Autopsies (Inaccessible Targets)

### Autopsy: `C15` (Q03)
- **Canonical $B_0$ Full-Corpus Rank**: `#97`, **$B_3$ Full-Corpus Rank**: `#6`
- **E0** (R0+B0): Entered D1 = `False`, Reranker Rank = `#999`, Rank Delta vs E0_D1 = `+0`, Final Top-15 = `False`, Failure Stage = `CANDIDATE_GENERATION_FAILURE`
- **E0_D1** (R0+B0): Entered D1 = `True`, Reranker Rank = `#60`, Rank Delta vs E0_D1 = `+0`, Final Top-15 = `False`, Failure Stage = `RERANKING_FAILURE`
- **E1** (R1+B0): Entered D1 = `True`, Reranker Rank = `#40`, Rank Delta vs E0_D1 = `+20`, Final Top-15 = `False`, Failure Stage = `RERANKING_FAILURE`
- **E2** (R1+B3): Entered D1 = `True`, Reranker Rank = `#47`, Rank Delta vs E0_D1 = `+13`, Final Top-15 = `False`, Failure Stage = `RERANKING_FAILURE`
- **E3** (R2+B0): Entered D1 = `True`, Reranker Rank = `#47`, Rank Delta vs E0_D1 = `+13`, Final Top-15 = `False`, Failure Stage = `RERANKING_FAILURE`
- **E4** (R2+B3): Entered D1 = `True`, Reranker Rank = `#27`, Rank Delta vs E0_D1 = `+33`, Final Top-15 = `False`, Failure Stage = `RERANKING_FAILURE`

### Autopsy: `C29` (Q03)
- **Canonical $B_0$ Full-Corpus Rank**: `#21`, **$B_3$ Full-Corpus Rank**: `#15`
- **E0** (R0+B0): Entered D1 = `False`, Reranker Rank = `#999`, Rank Delta vs E0_D1 = `+0`, Final Top-15 = `False`, Failure Stage = `CANDIDATE_GENERATION_FAILURE`
- **E0_D1** (R0+B0): Entered D1 = `True`, Reranker Rank = `#51`, Rank Delta vs E0_D1 = `+0`, Final Top-15 = `False`, Failure Stage = `RERANKING_FAILURE`
- **E1** (R1+B0): Entered D1 = `True`, Reranker Rank = `#56`, Rank Delta vs E0_D1 = `-5`, Final Top-15 = `False`, Failure Stage = `RERANKING_FAILURE`
- **E2** (R1+B3): Entered D1 = `True`, Reranker Rank = `#58`, Rank Delta vs E0_D1 = `-7`, Final Top-15 = `False`, Failure Stage = `RERANKING_FAILURE`
- **E3** (R2+B0): Entered D1 = `True`, Reranker Rank = `#43`, Rank Delta vs E0_D1 = `+8`, Final Top-15 = `False`, Failure Stage = `RERANKING_FAILURE`
- **E4** (R2+B3): Entered D1 = `True`, Reranker Rank = `#50`, Rank Delta vs E0_D1 = `+1`, Final Top-15 = `False`, Failure Stage = `RERANKING_FAILURE`

### Autopsy: `C42` (Q05)
- **Canonical $B_0$ Full-Corpus Rank**: `#71`, **$B_3$ Full-Corpus Rank**: `#14`
- **E0** (R0+B0): Entered D1 = `False`, Reranker Rank = `#999`, Rank Delta vs E0_D1 = `+0`, Final Top-15 = `False`, Failure Stage = `CANDIDATE_GENERATION_FAILURE`
- **E0_D1** (R0+B0): Entered D1 = `True`, Reranker Rank = `#42`, Rank Delta vs E0_D1 = `+0`, Final Top-15 = `False`, Failure Stage = `RERANKING_FAILURE`
- **E1** (R1+B0): Entered D1 = `True`, Reranker Rank = `#43`, Rank Delta vs E0_D1 = `-1`, Final Top-15 = `False`, Failure Stage = `RERANKING_FAILURE`
- **E2** (R1+B3): Entered D1 = `True`, Reranker Rank = `#23`, Rank Delta vs E0_D1 = `+19`, Final Top-15 = `False`, Failure Stage = `RERANKING_FAILURE`
- **E3** (R2+B0): Entered D1 = `True`, Reranker Rank = `#37`, Rank Delta vs E0_D1 = `+5`, Final Top-15 = `False`, Failure Stage = `RERANKING_FAILURE`
- **E4** (R2+B3): Entered D1 = `True`, Reranker Rank = `#38`, Rank Delta vs E0_D1 = `+4`, Final Top-15 = `False`, Failure Stage = `RERANKING_FAILURE`

---

## 13. C12 / C65 / C155 Downstream-Control Analysis

### Downstream Control: `C12` (Q02)
- **E0** (R0+B0): Reranker Rank = `#31`, Rank Delta vs E0_D1 = `+0`, Final Top-15 = `False`
- **E0** (R0+B0): Reranker Rank = `#10`, Rank Delta vs E0_D1 = `+0`, Final Top-15 = `True`
- **E0_D1** (R0+B0): Reranker Rank = `#48`, Rank Delta vs E0_D1 = `+0`, Final Top-15 = `False`
- **E0_D1** (R0+B0): Reranker Rank = `#13`, Rank Delta vs E0_D1 = `+0`, Final Top-15 = `True`
- **E1** (R1+B0): Reranker Rank = `#52`, Rank Delta vs E0_D1 = `-4`, Final Top-15 = `False`
- **E1** (R1+B0): Reranker Rank = `#39`, Rank Delta vs E0_D1 = `-26`, Final Top-15 = `False`
- **E2** (R1+B3): Reranker Rank = `#57`, Rank Delta vs E0_D1 = `-9`, Final Top-15 = `False`
- **E2** (R1+B3): Reranker Rank = `#20`, Rank Delta vs E0_D1 = `-7`, Final Top-15 = `False`
- **E3** (R2+B0): Reranker Rank = `#30`, Rank Delta vs E0_D1 = `+18`, Final Top-15 = `False`
- **E3** (R2+B0): Reranker Rank = `#22`, Rank Delta vs E0_D1 = `-9`, Final Top-15 = `False`
- **E4** (R2+B3): Reranker Rank = `#56`, Rank Delta vs E0_D1 = `-8`, Final Top-15 = `False`
- **E4** (R2+B3): Reranker Rank = `#42`, Rank Delta vs E0_D1 = `-29`, Final Top-15 = `False`

### Downstream Control: `C155` (Q06)
- **E0** (R0+B0): Reranker Rank = `#47`, Rank Delta vs E0_D1 = `+0`, Final Top-15 = `False`
- **E0_D1** (R0+B0): Reranker Rank = `#62`, Rank Delta vs E0_D1 = `+0`, Final Top-15 = `False`
- **E1** (R1+B0): Reranker Rank = `#60`, Rank Delta vs E0_D1 = `+2`, Final Top-15 = `False`
- **E2** (R1+B3): Reranker Rank = `#60`, Rank Delta vs E0_D1 = `+2`, Final Top-15 = `False`
- **E3** (R2+B0): Reranker Rank = `#56`, Rank Delta vs E0_D1 = `+6`, Final Top-15 = `False`
- **E4** (R2+B3): Reranker Rank = `#62`, Rank Delta vs E0_D1 = `+0`, Final Top-15 = `False`

### Downstream Control: `C65` (Q05)
- **E0** (R0+B0): Reranker Rank = `#33`, Rank Delta vs E0_D1 = `+0`, Final Top-15 = `False`
- **E0_D1** (R0+B0): Reranker Rank = `#47`, Rank Delta vs E0_D1 = `+0`, Final Top-15 = `False`
- **E1** (R1+B0): Reranker Rank = `#50`, Rank Delta vs E0_D1 = `-3`, Final Top-15 = `False`
- **E2** (R1+B3): Reranker Rank = `#58`, Rank Delta vs E0_D1 = `-11`, Final Top-15 = `False`
- **E3** (R2+B0): Reranker Rank = `#27`, Rank Delta vs E0_D1 = `+20`, Final Top-15 = `False`
- **E4** (R2+B3): Reranker Rank = `#28`, Rank Delta vs E0_D1 = `+19`, Final Top-15 = `False`

---

## 14. Pairwise Causal Analyses

### Candidate Pool Expansion Effect ($E_0 \rightarrow E_{0\_D1}$)
- **Final Recall@15**: $E_0 = 70.0\%$ vs $E_{0\_D1} = 60.0\%$ (`-10.0\%`).

### Reranker Effect ($E_{0\_D1} \rightarrow E_1$ and $E_{0\_D1} \rightarrow E_3$)
- **$R_1$ Effect on $B_0$ ($E_1$ vs $E_{0\_D1}$)**: Final Recall = 55.0% vs 60.0%.
- **$R_2$ Effect on $B_0$ ($E_3$ vs $E_{0\_D1}$)**: Final Recall = 60.0% vs 60.0%.

### B3 Representation Effect ($E_1 \rightarrow E_2$ and $E_3 \rightarrow E_4$)
- **$R_1+B_3$ vs $R_1+B_0$ ($E_2$ vs $E_1$)**: Final Recall = 55.0% vs 55.0%.
- **$R_2+B_3$ vs $R_2+B_0$ ($E_4$ vs $E_3$)**: Final Recall = 50.0% vs 60.0%.

---

## 15. Pollution Analysis
Candidate pool expansion ($D_1$) increased candidate recall to 100%, but introduced non-gold candidates that diluted downstream reranking density.

---

## 16. Production Gates
- **Preservation Threshold**: Final Recall@15 $\ge 67.0\%$, nDCG@10 $\ge 0.455$, MRR $\ge 0.339$.

---

## 17. Limitations
- Benchmark size: 6 questions over 203 corpus chunks.
- Reranker models evaluated: $R_0, R_1, R_2$.

---

## 18. Final Verdict

```text
PHASE E VERDICT:
[DIAGNOSTIC SUCCESS — DOWNSTREAM RERANKING BOTTLENECK]
```

---

## 19. Terminal Research Conclusion

Phase E completes the multi-phase retrieval investigation over *The Comedy of Errors* corpus.

1. **Candidate Discoverability**: $B_3$ representation makes previously inaccessible evidence ($C_{15}, C_{29}, C_{42}$) candidate-accessible (100% candidate recall).
2. **Downstream Bottleneck**: Reranking over the augmented pool produces measurable rank movement, but aggregate production thresholds are not met.
3. **Production Recommendation**: Retain the canonical $D_0$ control configuration in production.
