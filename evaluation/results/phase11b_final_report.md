# Phase 11B — Structural Fragmentation & Candidate Accessibility Diagnosis Final Report

## Executive Summary

Phase 11B evaluated whether persistent retrieval bottlenecks for primary evidence targets **C15** ($Q_{03}$) and **C42** ($Q_{05}$) are driven by **context fragmentation**, **structural metadata omission**, or **candidate depth limits**.

Following strict anti-leakage safeguards, `C15` and `C42` were treated strictly as **post-hoc diagnostic observations**.

### Key Diagnostic Outcome

> **Primary Decision**: `CANDIDATE_DEPTH_BOTTLENECK_SUPPORTED_WITHIN_SINGLE_QUERY_DENSE_DIAGNOSTIC`
> **Phase Status**: `CANONICAL_DIAGNOSTIC_COMPLETE_WITH_PRODUCTION_COMPARABILITY_LIMITATION`

- **Target C15** ($Q_{03}$, Primary Evidence): Baseline Dense Rank `#142` | Stage Localization: `Stage 2 — Candidate Depth Failure (within B3 single-query dense diagnostic)`
- **Target C42** ($Q_{05}$, Primary Evidence): Baseline Dense Rank `#99` | Stage Localization: `Stage 2 — Candidate Depth Failure (within B3 single-query dense diagnostic)`

---

## 1. Experimental Setup & Controls

- **Corpus**: Canonical 203 chunks (`The Comedy of Errors document.pdf`)
- **Embedding Model**: $E_0 =$ `sentence-transformers/all-MiniLM-L6-v2` (L2 Normalized Cosine Sim via Dot Product)
- **Cross-Encoder Model**: $R_0 =$ `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Evaluations**: 102 total runs (24 B1 + 24 B2 + 54 B3)

---

## 2. Family B1 — Context Fragmentation Diagnosis (Mean over Q01–Q06)

| Condition | Window Representation | Rec@10 | Rec@20 | Rec@50 | Prec@10 | MRR | nDCG@10 | EvComp% | C15 Rank (Q03) | C42 Rank (Q05) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B1_0** | Raw Text (Control) | 36.1% | 54.2% | 77.8% | 0.117 | 0.356 | 0.279 | 0.0% | #142 | #99 |
| **B1_1** | Chunk + Prev | 27.8% | 45.8% | 63.9% | 0.083 | 0.318 | 0.205 | 0.0% | #180 | #101 |
| **B1_2** | Chunk + Next | 40.3% | 40.3% | 77.8% | 0.133 | 0.190 | 0.253 | 0.0% | #160 | #96 |
| **B1_3** | Prev + Chunk + Next | 27.8% | 45.8% | 72.2% | 0.083 | 0.286 | 0.187 | 0.0% | #183 | #98 |

---

## 3. Family B2 — Structural Metadata Diagnosis (Mean over Q01–Q06)

| Condition | Metadata Representation | Rec@10 | Rec@20 | Rec@50 | Prec@10 | MRR | nDCG@10 | EvComp% | C15 Rank (Q03) | C42 Rank (Q05) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B2_0** | Raw Text (Control) | 36.1% | 54.2% | 77.8% | 0.117 | 0.356 | 0.279 | 0.0% | #142 | #99 |
| **B2_1** | Speakers | 54.2% | 58.3% | 81.9% | 0.167 | 0.343 | 0.316 | 16.7% | #105 | #55 |
| **B2_2** | Act/Scene Location | 40.3% | 54.2% | 81.9% | 0.133 | 0.404 | 0.316 | 16.7% | #142 | #106 |
| **B2_3** | Location + Speakers | 54.2% | 66.7% | 81.9% | 0.167 | 0.256 | 0.281 | 16.7% | #98 | #77 |

---

## 4. Family B3 — Candidate Depth $N$ Analysis (Mean over Q01–Q06)

| Depth N | Cand Recall@N | Final Recall@15 | nDCG@10 | MRR | EvComp% | Conversion % | C15 In Pool? | C42 In Pool? | C15 Fin Rank | C42 Fin Rank |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| N=010 | 36.1% | 36.1% | 0.256 | 0.385 | 0.0% | 83.3% | False | False | Absent | Absent |
| N=020 | 54.2% | 48.6% | 0.334 | 0.357 | 0.0% | 83.3% | False | False | Absent | Absent |
| N=030 | 73.6% | 68.1% | 0.354 | 0.370 | 33.3% | 91.7% | False | False | Absent | Absent |
| N=040 | 77.8% | 72.2% | 0.350 | 0.370 | 33.3% | 91.7% | False | False | Absent | Absent |
| N=050 | 77.8% | 72.2% | 0.349 | 0.369 | 33.3% | 91.7% | False | False | Absent | Absent |
| N=075 | 86.1% | 62.5% | 0.335 | 0.364 | 16.7% | 70.8% | False | False | Absent | Absent |
| N=100 | 91.7% | 62.5% | 0.335 | 0.364 | 16.7% | 68.1% | False | True | Absent | Absent |
| N=150 | 100.0% | 62.5% | 0.335 | 0.364 | 16.7% | 62.5% | True | True | Absent | Absent |
| N=203 | 100.0% | 62.5% | 0.335 | 0.364 | 16.7% | 62.5% | True | True | Absent | Absent |

---

## 5. Diagnostic Pipeline Stage Localization

### Target C15 ($Q_{03}$ — Primary Evidence)
- **Baseline Dense Rank**: `#142`
- **Stage Localization**: `Stage 2 — Candidate Depth Failure (within B3 single-query dense diagnostic)`
- **Stage Description**: Target absent at production N=20 (Rank #142), but enters candidate pool at deeper N > 20 within B3 single-query dense diagnostic.

### Target C42 ($Q_{05}$ — Primary Evidence)
- **Baseline Dense Rank**: `#99`
- **Stage Localization**: `Stage 2 — Candidate Depth Failure (within B3 single-query dense diagnostic)`
- **Stage Description**: Target absent at production N=20 (Rank #99), but enters candidate pool at deeper N > 20 within B3 single-query dense diagnostic.

---

## 6. Phase 11B Decision & Recommendations

> **Primary Decision**: `CANDIDATE_DEPTH_BOTTLENECK_SUPPORTED_WITHIN_SINGLE_QUERY_DENSE_DIAGNOSTIC`
> **Phase Status**: `CANONICAL_DIAGNOSTIC_COMPLETE_WITH_PRODUCTION_COMPARABILITY_LIMITATION`

### Detailed Interpretation:
> Under the Phase 11B single-query dense diagnostic, candidate depth is demonstrably limiting accessibility of C15 and C42. C15 first enters the dense candidate pool at N=150 and C42 first enters at N=100. However, because Phase 11B B3 does not reproduce the Phase 10C production candidate-construction mechanism, this result must not be interpreted as proof that simply increasing the production Phase 10C candidate depth from N=20 will produce the same outcome.

### Corrected Synthesis:
1. **Context fragmentation**: Not supported. Neighbor-window representations degraded benchmark retrieval relative to B1_0.
2. **Structural metadata**: Weakly supported. Speaker/location metadata produced measurable benchmark-wide improvements and improved target ranks, but neither C15 nor C42 became accessible at N<=20.
3. **Candidate depth**: Supported within the single-query dense diagnostic. C15 and C42 are absent at N=20 but appear at N=150 and N=100 respectively.
4. **Production extrapolation**: Unresolved. The Phase 11B B3 mechanism differs from the Phase 10C production candidate construction, so the experiment cannot establish that increasing the actual production candidate depth alone will recover these targets.
5. **Deep candidate reranking**: Behavior of C15/C42 after they enter deeper diagnostic pools ($N=150$ and $N=100$) is reported as `DEEP-CANDIDATE-RERANKING_BEHAVIOR`. This is not classified as a production Stage 3 failure because neither target is present at production N=20.

### Methodological Limitation:
> Phase 11B B3 isolates dense candidate-depth behavior rather than reproducing the complete Phase 10C candidate-generation pipeline. Consequently, its N-curve is diagnostic evidence about single-query dense accessibility, not a direct production-depth optimization result.

---

## 7. Verification Assertions Audit

- [x] Programmatic validation assertions 1–9 and 11–15 passed cleanly.
- [ ] **Assertion #10 Status**: `UNRESOLVED` (Phase 10C evaluated multi-query subquery + BM25 fusion candidate construction, whereas Phase 11B B3 evaluated single-query dense global ranking followed by N-slicing. Therefore B3 N=20 is not a like-for-like reproduction of the Phase 10C production candidate pool, and B3 N=20 metrics must not be called a regression against Phase 10C). See [Reconciliation Report](file:///d:/Fine%20Tunning/ADVANCE%20RAG/RAG/evaluation/results/phase11b_b3_control_reconciliation.md).
- [x] Full B3 dense ranking frozen across candidate depths $N$.
- [x] Zero data leakage.
