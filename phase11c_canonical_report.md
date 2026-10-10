# Phase 11C — Production Candidate-Depth Diagnostic Canonical Report

> **Official Canonical Post-Execution Audit & Reconciliation Report**  
> **Status**: **CANONICAL & FROZEN**  
> **Date**: 2026-10-06  
> **Source Artifact**: `phase_11c_results.json`  
> **Control Reproduction Gate**: **PASS (100% Exact Match)**  

---

## 1. Objective

Phase 11C was designed to answer one central diagnostic question:

> **In the actual Phase 10C production retrieval pipeline, are targets C15 and C42 lost because the candidate-generation depth is too shallow, or are they lost later during reranking?**

This phase resolves the production-comparability limitation discovered in Phase 11B (which sliced a single-query dense ranking) by testing candidate-generation depth directly inside the **actual Phase 10C production candidate-generation pipeline**.

---

## 2. Frozen Control Configuration

The existing Phase 10C production retrieval pipeline was held strictly frozen:

- **Corpus**: 203 chunks ($C_0 \dots C_{202}$) from `The Comedy of Errors document.pdf`
- **Benchmark Suite**: Q01–Q06 and canonical gold evidence annotations (20 total gold evidence units)
- **Text Representation**: Raw text ($A_0$)
- **Dense Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` ($E_0$)
- **BM25 Search**: Default BM25 retriever indexed on raw chunk text
- **Multi-Query Formulation**: 3 frozen Q2 subqueries per benchmark question
- **Candidate Fusion**: Deduplicated union of top-$N$ candidates from each of the 3 dense subqueries + top-$N$ candidates from BM25
- **Cross-Encoder Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2` ($R_0$)
- **Selection Budget**: Top-$K = 15$
- **Anti-Leakage Safeguards**: Zero target boosting, filtering, or gold injection. C15 and C42 inspected strictly post-hoc.

---

## 3. Control Reproduction Gate

The control run ($N=20$) was verified against the canonical Phase 10C control targets:

| Metric | Canonical Phase 10C Target | Phase 11C Control ($N=20$) | Difference | Gate Status |
| :--- | :---: | :---: | :---: | :---: |
| **Candidate Recall@20** | **85.0%** | **85.0%** (17/20) | 0.0% | `PASS` |
| **Final Recall@15** | **70.0%** | **70.0%** (14/20) | 0.0% | `PASS` |
| **Macro nDCG@10** | **0.485** | **0.485** | 0.000 | `PASS` |
| **MRR** | **0.369** | **0.369** | 0.000 | `PASS` |

### Control Gating Semantics
The testing script (`run_phase11c_candidate_depth_diagnostic.py`) enforced a strict execution gate on **Candidate Recall@20 (85.0%)** and **Final Recall@15 (70.0%)** within a $\pm 3.0\%$ numerical tolerance. All four control metrics matched the canonical Phase 10C control with **100% exact numerical precision**.

---

## 4. Candidate Depth Sweep (N=20, 30, 50, 75, 100)

Reconciled directly from raw JSON execution trace (`phase_11c_results.json`):

| Depth N | Mean Pool Size | Candidate Rec@N | Final Rec@15 | Ev Precision@15 | Primary Prec@15 | MRR | nDCG@10 | nDCG@15 | Ev Completeness | Conversion Rate |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **N=20** | **55.2** | **85.0%** | **70.0%** | 0.156 | 0.067 | 0.369 | **0.485** | 0.518 | 33.3% | **82.4%** |
| **N=30** | **75.2** | **90.0%** | **65.0%** | 0.144 | 0.067 | 0.366 | **0.468** | 0.502 | 33.3% | **72.2%** |
| **N=50** | **104.8** | **90.0%** | **60.0%** | 0.133 | 0.067 | 0.364 | **0.468** | 0.489 | 16.7% | **66.7%** |
| **N=75** | **133.5** | **95.0%** | **60.0%** | 0.133 | 0.067 | 0.364 | **0.468** | 0.485 | 16.7% | **63.2%** |
| **N=100** | **149.5** | **100.0%** | **60.0%** | 0.133 | 0.067 | 0.364 | **0.468** | 0.485 | 16.7% | **60.0%** |

---

## 5. Target C15 Autopsy ($Q_{03}$ — Primary Evidence)

| Depth N | In Candidate Pool? | Insertion Position | Dense Retrieval Subquery Source(s) | BM25 Rank | CE Score | CE Rank in Pool | In Final Top-15? | Failure Stage Classification |
| :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **N=20** | `False` | null | None | null | null | null | `False` | `Stage 1 — Candidate Generation Failure` |
| **N=30** | `False` | null | None | null | null | null | `False` | `Stage 1 — Candidate Generation Failure` |
| **N=50** | `False` | null | None | null | null | null | `False` | `Stage 1 — Candidate Generation Failure` |
| **N=75** | `False` | null | None | null | null | null | `False` | `Stage 1 — Candidate Generation Failure` |
| **N=100** | **`True`** | **#127** | `SQ2 ("Adriana marriage..."): Dense Rank #97` | null | `-4.9510` | **#79** | `False` | **`Stage 2 — Reranking Failure`** |

### Ambiguity A Resolution:
Target C15 is **ABSENT** from the candidate pool at depths $N=20, 30, 50, 75$. It **first enters the candidate pool at N=100** at dictionary insertion position #127 via Subquery 2 (`dense_sq2_rank_97`). Its Cross-Encoder score is `-4.9510` and its Cross-Encoder rank in the pool is **#79**. It does NOT enter the final Top-15 selection.

---

## 6. Target C42 Autopsy ($Q_{05}$ — Primary Evidence)

| Depth N | In Candidate Pool? | Insertion Position | Dense Retrieval Subquery Source(s) | BM25 Rank | CE Score | CE Rank in Pool | In Final Top-15? | Failure Stage Classification |
| :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **N=20** | `False` | null | None | null | null | null | `False` | `Stage 1 — Candidate Generation Failure` |
| **N=30** | `False` | null | None | null | null | null | `False` | `Stage 1 — Candidate Generation Failure` |
| **N=50** | `False` | null | None | null | null | null | `False` | `Stage 1 — Candidate Generation Failure` |
| **N=75** | **`True`** | **#71** | `SQ1 ("Antipholus of Syracuse..."): Dense Rank #71` | null | `+0.5802` | **#45** | `False` | **`Stage 2 — Reranking Failure`** |
| **N=100** | **`True`** | **#71** | `SQ1: Dense Rank #71, SQ3: Dense Rank #87` | null | `+0.5802` | **#46** | `False` | **`Stage 2 — Reranking Failure`** |

### Ambiguity B Resolution:
Target C42 is **ABSENT** from the candidate pool at depths $N=20, 30, 50$. It **first enters the candidate pool at N=75** at dictionary insertion position #71 via Subquery 1 (`dense_sq1_rank_71`). Its Cross-Encoder score is `+0.5802` and its Cross-Encoder rank in the pool is **#45**. At N=100, C42 is retrieved by both Subquery 1 (rank #71) and Subquery 3 (rank #87), achieving CE rank **#46**. It does NOT enter the final Top-15 selection.

---

## 7. Retrieval-Stage Diagnosis

Reconstructing the pipeline stage-by-stage across depths:

```text
Query
  ↓
Dense retrieval / Q2 subqueries  [Stage 1: Fails at N < 75/100; Succeeds at N >= 75/100]
  ↓
BM25 retrieval                   [Fails for C15 and C42 at all tested depths N <= 100]
  ↓
Candidate union                  [Deduplicated pool formed]
  ↓
Cross-Encoder reranking          [Stage 2: Fails for C15 (CE #79) and C42 (CE #45/#46)]
  ↓
Final Top-15                     [Targets rejected from final Top-15]
```

- **Shallow Depths ($N < 75$)**: Failure is strictly **Stage 1 — Candidate Generation Failure**. Neither C15 nor C42 enters the candidate pool.
- **Deeper Depths ($N \ge 75$ for C42, $N=100$ for C15)**: Stage 1 candidate generation succeeds in pulling the targets into the pool. The failure shifts entirely to **Stage 2 — Reranking Failure**, as Cross-Encoder $R_0$ assigns higher scores to distractor candidates.

---

## 8. Rank-Semantics Audit

- **Implementation Inspection**: `cand_r = cand_ids.index(target_cid) + 1`
- **Audit Finding**: `cand_ids` records the dictionary key insertion order as 3 dense subqueries and BM25 populate the candidate pool dictionary.
- **Semantics Definition**: `candidate_rank` is **candidate-pool insertion position**, NOT a relevance ranking.
- **Reporting Rule**: Insertion positions (#127 for C15, #71 for C42) are documented strictly as insertion order. Retrieval ranking quality is evaluated strictly via Dense Subquery Ranks (`SQ2 #97` for C15, `SQ1 #71` for C42) and Cross-Encoder Reranker Ranks (`CE #79` for C15, `CE #45/#46` for C42).

---

## 9. BM25 Score-Semantics Audit

- **Implementation Inspection**: `candidate_provenance[cid]["bm25_score"] = float(N - r + 1)`
- **Audit Finding**: The stored value `N - r + 1` is a linear rank-based proxy derived from BM25 rank $r$, not an Okapi BM25 score.
- **Semantics Definition**: Referred to strictly as **BM25 rank proxy**. Neither C15 nor C42 was retrieved by BM25 within top-100 candidates (`bm25_rank: null`).

---

## 10. Aggregate Performance Interpretation

Evaluating the impact of increasing candidate generation depth $N$:

1. **Candidate Recall**: Increases monotonically from **85.0% $\to$ 90.0% $\to$ 90.0% $\to$ 95.0% $\to$ 100.0%** ($N=100$).
2. **Candidate Pool Size**: Nearly triples from **55.2 $\to$ 149.5** unique candidates per question.
3. **Final Recall@15**: **Degrades** from **70.0% (at N=20) $\to$ 65.0% (at N=30) $\to$ 60.0% (at N=50, 75, 100)**.
4. **nDCG@10**: Drops from **0.485 $\to$ 0.468**.
5. **Candidate-to-Final Conversion**: Drops from **82.4% $\to$ 60.0%**.

### Mechanism:
Expanding candidate depth $N$ introduces noisy distractor chunks into the candidate pool. Because Cross-Encoder $R_0$ (`ms-marco-MiniLM-L-6-v2`) has domain sensitivity limitations on archaic/poetic dialogue, these distractor chunks receive high logit scores, pushing valid gold chunks out of the final Top-15 selection.

---

## 11. Final Findings (Answers to Required Questions)

### Q1: Does increasing candidate depth improve candidate accessibility for C15 and/or C42?
**Yes.** Increasing candidate depth $N$ recovers C42 into the candidate pool at $N=75$ and C15 at $N=100$.

### Q2: At what depth does each target first become accessible?
- **C42**: First enters candidate pool at **$N=75$** (via Subquery 1, dense rank #71).
- **C15**: First enters candidate pool at **$N=100$** (via Subquery 2, dense rank #97).

### Q3: Once accessible, does the frozen Cross-Encoder place it in Top-15?
**No.** At $N=75$, C42 is placed at **CE Rank #45**. At $N=100$, C42 is placed at **CE Rank #46** and C15 is placed at **CE Rank #79**. Neither enters final Top-15.

### Q4: Does increasing candidate depth improve or degrade aggregate end-to-end retrieval?
**It degrades aggregate end-to-end retrieval.** Final Recall@15 drops by 10 percentage points (70.0% $\to$ 60.0%) and nDCG@10 drops from 0.485 $\to$ 0.468.

### Q5: Is candidate generation the limiting stage at shallow N?
**Yes.** At shallow depths ($N < 75$), both targets are absent from the candidate pool (`Stage 1 — Candidate Generation Failure`).

### Q6: Does reranking become the limiting stage after the target enters the candidate pool?
**Yes.** Once targets enter the pool ($N \ge 75$), candidate generation succeeds and the bottleneck shifts strictly to `Stage 2 — Reranking Failure`.

### Q7: Is there evidence that simply increasing N is a production solution?
**No.** Increasing $N$ degrades final recall, increases candidate pool noise, inflates reranking latency, and fails to rescue either target into final retrieval.

---

## 12. Limitations

1. **Union Pool Dynamics**: Candidate pool size per question is 2.5x–3x larger than $N$ due to unioning 3 Q2 dense subqueries + BM25.
2. **Frozen Reranker Capability**: Evaluated solely under baseline reranker $R_0$ (`ms-marco-MiniLM-L-6-v2`).
3. **Target Specificity**: C15 and C42 are diagnostic targets for poetic metaphor and Renaissance slang; results reflect their specific semantic characteristics.

---

## 13. Canonical Conclusion

> **FINAL ENGINEERING CONCLUSION**:
>
> In the actual Phase 10C production retrieval pipeline, **C15 and C42 are lost at shallow depth ($N < 75$) due to Candidate Generation Failure**. When candidate depth is expanded ($N=75$ for C42, $N=100$ for C15), both targets enter the candidate pool but are rejected downstream by Cross-Encoder $R_0$ (CE Rank #45 and #79), classifying their ultimate bottleneck as **`RERANKING_FAILURE`**.
>
> Merely increasing candidate generation depth $N$ in production is **counterproductive**: while it achieves 100% Candidate Recall at $N=100$, it degrades Final Recall@15 from 70.0% down to 60.0% due to cross-encoder distractor noise. Resolving target failures requires updating downstream reranking capability or candidate text representations rather than inflating candidate depth $N$.

---

## 14. Final Audit Verification Checklist

```text
CONTROL: PASS
NUMERICAL RECONCILIATION: PASS
C15 RECONCILIATION: PASS
C42 RECONCILIATION: PASS
RANK SEMANTICS: PASS
BM25 SEMANTICS: PASS
REPORT CONSISTENCY: PASS
```
