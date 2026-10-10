# Terminal Architectural Phase Report — Evidence-Set Selection

> **Research Question**: Does replacing independent chunk-level reranking with listwise evidence-set selection recover required evidence that is already present in the candidate pool, while preserving or improving production retrieval quality relative to the canonical D0 system?  
> **Status**: **TERMINAL EVALUATION COMPLETE**  
> **Date**: 2026-10-08  
> **Final Verdict**: **`EVALUATION INCONCLUSIVE — EXPAND BENCHMARK`**

---

## 1. Executive Summary
This terminal architectural evaluation tested listwise evidence-set selection over the frozen 203-chunk *The Comedy of Errors* corpus.

- **Canonical D0 Baseline Control**: Candidate Recall@20 = 85.0%, Final Recall@15 = 70.0%, Macro nDCG@10 = 0.485, MRR = 0.369.
- **Control A ($E_0\_D1$)**: Final Recall@15 = 60.0%, Macro nDCG@10 = 0.468, MRR = 0.365.
- **Experimental B (Listwise LLM Ordering A)**: Final Recall@15 = 65.0%, Macro nDCG@10 = 0.698, MRR = 0.854.

---

## 2. Research Question
Answered strictly via empirical evidence: does listwise evidence-set selection recover candidate-accessible evidence while preserving aggregate production quality?

---

## 3. Frozen Experimental Contract
- **Corpus**: 203 chunks ($C_0$–$C_202$), SHA-256 verified against `baseline/phaseD_corpus_manifest_canonical.csv`.
- **Benchmark**: Q01–Q06 (20 gold evidence units), 3 frozen subqueries per question.

---

## 4. Canonical D0 Control
Reproduced canonical production baseline control within tolerances.

| Metric | Target | Measured ($D_0$) | Status |
| :--- | :---: | :---: | :---: |
| **Candidate Recall@20** | **85.0%** | **85.0%** | **PASS** |
| **Final Recall@15** | **70.0%** | **70.0%** | **PASS** |
| **Macro nDCG@10** | **0.485** | **0.485** | **PASS** |
| **MRR** | **0.369** | **0.369** | **PASS** |

---

## 5. Answer-Level Impact Check
- **Q03** (Hard Targets ['C15', 'C29']): `UNCLASSIFIED — NO BLIND ANSWER EVALUATOR`
- **Q05** (Hard Targets ['C42']): `UNCLASSIFIED — NO BLIND ANSWER EVALUATOR`

---

## 6. Variance & Stability Analysis
Bootstrap 95% Confidence Intervals (1,000 resamples):
- **Canonical_D0 (Final_Recall_15)**: Mean = 0.7222 (95% CI: [0.5417, 0.9167], SE = 0.0955)
- **Listwise_LLM_Ordering_A (Final_Recall_15)**: Mean = 0.6667 (95% CI: [0.4444, 0.8889], SE = 0.1145)
- **Listwise_LLM_Ordering_A (nDCG_10)**: Mean = 0.6985 (95% CI: [0.489, 0.9118], SE = 0.1112)
- **Listwise_LLM_Ordering_A (MRR)**: Mean = 0.8542 (95% CI: [0.5625, 1.0], SE = 0.1289)
- **Delta_Listwise_vs_D0 (Delta_Final_Recall)**: Mean = -0.0556 (95% CI: [-0.1948, 0.0833], SE = 0.0741)

---

## 7. D1 Candidate Pool Definition
Single frozen pool per question ($B_0$ Top-20 + BM25 Top-20 + $B_3$ Top-20 per subquery).

---

## 8. E0_D1 Control
Primary selection control using independent $R_0$ ranking over the $D_1$ candidate pool.

---

## 9. Listwise Selector Method
Structured JSON prompt over full $D_1$ pool optimizing for evidence coverage, complementarity, answer sufficiency, and low redundancy.

---

## 10. Position-Bias / Ordering Robustness Results
Evaluated under Orderings A (canonical), B (shuffle seed 42), C (shuffle seed 123).

---

## 11. Repeatability / Determinism Controls
Recorded temperature=0.0, deterministic random seeds, model tag, and timestamps.

---

## 12. Aggregate Metrics

| Configuration | Selection Mechanism | Final Recall@15 | Macro nDCG@10 | MRR | Delta Recall vs E0_D1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **$E_0$ (Control D0)** | Independent $R_0$ on $D_0$ | **70.0%** | **0.485** | **0.369** | - |
| **$E_0\_D1$ (Control A)** | Independent $R_0$ on $D_1$ | **60.0%** | **0.468** | **0.365** | `0.0%` |
| **Listwise LLM (Ordering A)** | Listwise LLM Evidence Selector | **65.0%** | **0.698** | **0.854** | `+5.0%` |
| **Listwise LLM (Ordering B)** | Listwise LLM (Shuffle 42) | **65.0%** | **0.684** | **0.771** | `+5.0%` |
| **Listwise LLM (Ordering C)** | Listwise LLM (Shuffle 123) | **65.0%** | **0.689** | **0.746** | `+5.0%` |
| **MMR Selector** | Deterministic Coverage MMR | **0.0%** | **0.000** | **0.000** | `-60.0%` |

---

## 13. Per-Question Metrics
Question-level breakdown recorded in `phase_architectural_question_metrics.csv`.

---

## 14. C15 Autopsy
- **C15**: Entered D1 = `True`, E0_D1 Rank = `#60`, Listwise Position = `#999`, Recovery Status = `SELECTION_FAILURE`

---

## 15. C29 Autopsy
- **C29**: Entered D1 = `True`, E0_D1 Rank = `#51`, Listwise Position = `#999`, Recovery Status = `SELECTION_FAILURE`

---

## 16. C42 Autopsy
- **C42**: Entered D1 = `True`, E0_D1 Rank = `#42`, Listwise Position = `#999`, Recovery Status = `SELECTION_FAILURE`

---

## 17. Evidence-Set Coverage
Measured evidence-set coverage for Q01–Q06 dynamically recorded in metrics CSV.

---

## 18. Redundancy Analysis
Average selected set size: 3.0 chunks per query.

---

## 19. E0_D1 vs Listwise Causal Comparison
Comparing independent chunk ranking ($E_0\_D1$) against listwise evidence-set selection over the exact same candidate pool ($D_1$).

---

## 20. Conditional Gating Results
- Escalation Rate: 0.0%
- Gated System Final Recall: 70.0%
- Gated System Macro nDCG: 0.485

---

## 21. Production Viability
Measured retention thresholds ($Recall \ge 67.0\%$, $nDCG \ge 0.455$, $MRR \ge 0.339$).

---

## 22. Limitations
Benchmark scope: 6 questions across 203 corpus chunks.

---

## 23. Final Verdict

```text
FINAL TERMINAL VERDICT:
[EVALUATION INCONCLUSIVE — EXPAND BENCHMARK]
```

---

## 24. Terminal Scientific Conclusion

### Established Findings
- Control reproduction or variance stability limits prevent a conclusive verdict.

### Unsupported Findings
- Single-stage listwise LLM selection is not necessary for queries where $D_0$ already achieves 100% recall.

### Remaining Uncertainty
- Latency and cost bounds under heavy multi-tenant production load.

---

## 25. Terminal Scientific Principle
At the conclusion of this terminal evaluation phase, the evidence supports reporting strictly what the empirical measured results establish.
