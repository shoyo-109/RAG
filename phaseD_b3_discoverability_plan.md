# Phase D — B3 Representation Discoverability Diagnostic Experimental Plan

## Executive Summary

Phase C established that remaining RAG evidence accessibility failures stem from **representation insufficiency** (`C15`, `C42`) or **multi-chunk discourse distribution** (`C29`), while other failures occur downstream despite adequate representation (`C12`, `C65`, `C155`).

**Phase D** tests the core causal question:
> **Can the structured B3 representation make previously inaccessible evidence discoverable through an auxiliary candidate-generation path, at controlled depths N ∈ {20, 50, 100}, while the canonical B0/BM25 retrieval path, R0 reranker, and final Top-15 mechanism remain completely frozen?**

This is a controlled diagnostic experiment. The canonical production pipeline ($D_0$) remains frozen and unchanged.

---

## 1. Frozen Baseline Control ($D_0$) Specification

The production baseline control $D_0$ must be executed first and reproduce canonical performance exactly:

```text
3 frozen Q2 subqueries per question (E0 dense Top-20 per subquery)
                         +
BM25 Top-20 on primary question
                         ↓
             Deduplicated Candidate Union
                         ↓
        R0 Cross-Encoder Reranking (Top-15)
```

- **Models**:
  - $E_0$: `sentence-transformers/all-MiniLM-L6-v2`
  - $R_0$: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Control Targets**:
  - Candidate Recall@20 = **85.0%** ($\pm 3\%$)
  - Final Recall@15 = **70.0%** ($\pm 3\%$)
  - Macro nDCG@10 = **0.485** ($\pm 0.030$)
  - MRR = **0.369** ($\pm 0.030$)

---

## 2. Experimental Configurations ($D_0 \dots D_3$)

The canonical $B_0$ + BM25 candidate pool remains intact across all runs. Auxiliary $B_3$ candidates are added to the $D_0$ candidate pool via deduplicated union.

| Configuration | Description | $B_0$ Dense Depth | BM25 Depth | Auxiliary $B_3$ Depth per Subquery |
| :--- | :--- | :---: | :---: | :---: |
| **$D_0$** | Canonical Baseline Control | Top-20 × 3 | Top-20 | **0** (Disabled) |
| **$D_1$** | Canonical + $B_3$ Top-20 | Top-20 × 3 | Top-20 | **Top-20 × 3** |
| **$D_2$** | Canonical + $B_3$ Top-50 | Top-20 × 3 | Top-20 | **Top-50 × 3** |
| **$D_3$** | Canonical + $B_3$ Top-100 | Top-20 × 3 | Top-20 | **Top-100 × 3** |

### Independent Run Semantics
- Each configuration $D_1, D_2, D_3$ starts from the exact same frozen canonical $D_0$ pool.
- $D_1 = D_0 + B_3@20$ per subquery.
- $D_2 = D_0 + B_3@50$ per subquery.
- $D_3 = D_0 + B_3@100$ per subquery.
- The $B_3$ depth applies **independently to each of the 3 frozen Q2 subqueries** per question.

---

## 3. Mandatory Static Audit Invariants (18 Pre-Execution Assertions)

Before executing the experiment, a static source audit MUST pass all 18 assertions:

1. **A1**: Canonical $D_0$ uses $B_0$ dense Top-20 × 3 + BM25 Top-20 + $R_0$ Top-15.
2. **A2**: $E_0$ embedding model unchanged (`sentence-transformers/all-MiniLM-L6-v2`).
3. **A3**: $R_0$ cross-encoder unchanged (`cross-encoder/ms-marco-MiniLM-L-6-v2`).
4. **A4**: Exactly 3 frozen Q2 subqueries per question ($Q_{01} \dots Q_{06}$).
5. **A5**: BM25 path unchanged (Top-20 on primary question).
6. **A6**: $B_0$ path unchanged (Dense Top-20 per subquery).
7. **A7**: $B_3$ generated target-blind without gold annotations.
8. **A8**: No hardcoded target chunk IDs (`C15`, `C42`, `C29`, etc.) in retrieval logic.
9. **A9**: No target-specific query modifications.
10. **A10**: No target-specific ranking logic (uniform $R_0$ scoring).
11. **A11**: $B_3$ is auxiliary and never replaces $B_0$.
12. **A12**: Final Top-15 mechanism unchanged (selected exclusively via $R_0$).
13. **A13**: $D_0$ control gate executes before $D_1 \dots D_3$.
14. **A14**: $B_3$ depths set to exactly $\{20, 50, 100\}$.
15. **A15**: Per-evidence provenance retained (tracking $B_0$ rank, $B_3$ rank, union status, $R_0$ rank, final Top-15 status).
16. **A16**: Candidate pollution metrics retained (tracking Primary, Supporting, Relevant, Irrelevant, Duplicate).
17. **A17**: $R_0$ promotion metrics retained ($B_3 \rightarrow \text{candidate} \rightarrow R_0 \text{ Top-15} \rightarrow \text{final}$).
18. **A18**: Diagnostic results separated from production control.

---

## 4. Evidence Provenance & Exhaustive Classifications

For each of the 20 gold evidence units, the experiment records complete provenance:
- `question_id`, `evidence_chunk_id`, `role`
- `canonical_B0_rank`, `canonical_B0_candidate`, `canonical_final`
- `B3_best_rank` (minimum full-corpus rank across frozen subqueries)
- `B3_retrieved_D1`, `B3_retrieved_D2`, `B3_retrieved_D3`
- `B3_new_candidate_D1`, `B3_new_candidate_D2`, `B3_new_candidate_D3`
- `entered_union_D1`, `entered_union_D2`, `entered_union_D3`
- `R0_rank_after_union_D1`, `R0_rank_after_union_D2`, `R0_rank_after_union_D3`
- `final_top15_D1`, `final_top15_D2`, `final_top15_D3`
- **Exhaustive Evidence-Level Classifications**:
  - `B3_DISCOVERABILITY_AND_RERANK_SUCCESS`: Entered candidate union AND placed in final Top-15 by $R_0$.
  - `B3_DISCOVERABILITY_R0_FAILURE`: Entered candidate union BUT $R_0$ failed to place in final Top-15.
  - `B3_NON_DISCOVERY`: Failed to enter candidate union at tested depth.

---

## 5. Pre-Registered Decision Criteria & Verdict Rules

The final verdict is computed dynamically from measured result objects evaluating all 7 pre-registered criteria:

1. **Material Accessibility Improvement**: $B_3$ recovers previously inaccessible gold evidence into the candidate pool.
2. **Shallow-Depth Gain**: Evaluates whether recovery occurs at shallow depth ($D_1$ or $D_2$) vs deep-only ($D_3$).
3. **Final Recall Preservation**: $\text{Final Recall}@15 \ge \text{D0 level}$.
4. **nDCG Preservation**: $\text{nDCG}@10 \ge \text{D0} - 0.030$.
5. **MRR Preservation**: $\text{MRR} \ge \text{D0} - 0.030$.
6. **Candidate Pollution**: Pollution rate calculated cleanly using canonical relevance/gold labels.
7. **Target-Blindness**: Target-blind retrieval logic.

The verdict will be dynamically computed as exactly one of:
- `PASS` (or `PHASE D VERDICT: [PASS — PRODUCTION-RELEVANT]`)
- `DIAGNOSTIC SUCCESS — DOWNSTREAM BOTTLENECK` (or `PHASE D VERDICT: [DIAGNOSTIC SUCCESS — DOWNSTREAM BOTTLENECK]`)
- `NO SUPPORT` (or `PHASE D VERDICT: [NO SUPPORT]`)

---

## 6. Output Artifacts

The experiment produces 5 standardized artifacts (written to both `evaluation/` directories and root):
- `phaseD_b3_discoverability_report.md` / `evaluation/reports/phaseD_b3_discoverability_report.md`
- `phaseD_results.json` / `evaluation/results/phaseD_results.json`
- `phaseD_evidence_trace.csv` / `evaluation/results/phaseD_evidence_trace.csv`
- `phaseD_representation_metrics.csv` / `evaluation/results/phaseD_representation_metrics.csv`
- `phaseD_question_metrics.csv` / `evaluation/results/phaseD_question_metrics.csv`
