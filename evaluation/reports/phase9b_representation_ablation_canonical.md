# Phase 9B — Canonical Retrieval Representation Ablation Report

> **Note:** This report was regenerated directly from the canonical Phase 9B result artifact (`phase9b_canonical_results.json`). No experimental values are manually entered.

**Corpus:** *The Comedy of Errors* (203 semantic chunks)  
**Reconciliation Date:** 2026-09-27  
**Retrieval Architecture:** Q2 Multi-Query Expansion -> Candidate Union N20 -> Cross-Encoder -> Top10  
**Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2 (384-dim)`  
**Cross-Encoder Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`  
**Target Chunks:** `C15, C29, C42, C61, C147`

---

## Executive Summary

Phase 9B evaluated 5 document representation conditions across all 203 chunks without changing chunk boundaries (preserving a strict 1:1 chunk ID mapping):
- **`R0` (Raw Control):** Original chunk text.
- **`R1` (Local Context-Enriched):** `[PREVIOUS]` + `[TARGET]` + `[NEXT]` chunks concatenated.
- **`R2` (Metadata-Enriched):** Act, Scene, Speakers, Document Type, and Entities prepended.
- **`R3` (Context + Metadata):** Header metadata prepended to 3-chunk local context.
- **`R4` (Semantic Gloss Probe):** Neutral 1-2 sentence LLM summary prepended to original text.

### Reconciled Key Findings

1. **Dense Retrieval (Experiment A):**  
   - **R2 Metadata** increased Dense Recall@10 from **35.0%** (R0) to **50.0%** (+15.0% gain).
   - **R4 Gloss Probe** achieved **65.0%** Dense Recall@20 (+10.0% over R0) and **80.0%** Dense Recall@50.

2. **Full Pipeline Retrieval (Experiment B):**  
   - **R4 Gloss Probe** achieved **65.0%** Final Recall@10 (+5.0% over R0), **0.604** nDCG@10 (+0.119 over R0), and **33.3%** Evidence Completeness (doubled over R0).
   - **R1 and R3 Context Appending** caused Cross-Encoder reranking failures (Final Recall@10 collapsed to 30.0% and 40.0%).

3. **Target Chunk Autopsy:**
   - **`C147`:** Dense rank improved from #6 to **#4** under R2, entering Candidate Union $N=20$ and reaching **Final Top-10 (`final_top10 = YES`)**.
   - **`C15`:** Dense rank jumped **72 positions** (from #142 in R0 to **#70** in R3), entering candidate pool $N=20$ (`candidate_N20 = YES`).
   - **`C29`:** Dense rank improved **20 positions** (from #60 in R0 to **#40** in R2), entering candidate pool $N=20$ (`candidate_N20 = YES`).
   - **`C42`:** Dense rank improved **32 positions** (from #99 in R0 to **#67** in R2).
   - **`C61`:** Dense rank improved from #33 to **#23** in R2 (CE rank #11).

---

## 1. Experiment A — Canonical Dense Retrieval Results

| Condition | Dense Recall@10 | Dense Recall@20 | Dense Recall@50 | C15 Rank | C29 Rank | C42 Rank | C61 Rank | C147 Rank |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **R0 Raw** | 35.0% | 55.0% | 75.0% | #142 | #60 | #99 | #33 | #6 |
| **R1 Context** | 25.0% | 45.0% | 65.0% | #178 | #71 | #127 | #135 | #92 |
| **R2 Metadata** | 50.0% | 55.0% | 80.0% | #83 | #40 | #67 | #23 | #4 |
| **R3 Cont+Meta** | 40.0% | 50.0% | 70.0% | #70 | #50 | #81 | #103 | #55 |
| **R4 Gloss Probe** | 50.0% | 65.0% | 80.0% | #118 | #51 | #101 | #26 | #6 |

---

## 2. Experiment B — Canonical Full Pipeline Results

| Condition | Candidate Rec@20 | Final Rec@10 | EvPrec@10 | nDCG@10 | Evidence Completeness | Mean Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **R0 Raw** | 85.0% | **60.0%** | 0.200 | 0.485 | 16.7% | 2480.1 ms |
| **R1 Context** | 75.0% | 30.0% | 0.100 | 0.266 | 16.7% | 3959.5 ms |
| **R2 Metadata** | 85.0% | 50.0% | 0.167 | 0.508 | 16.7% | 3001.2 ms |
| **R3 Cont+Meta** | 85.0% | 40.0% | 0.133 | 0.403 | 16.7% | 4050.3 ms |
| **R4 Gloss Probe** | **90.0%** | **65.0%** | **0.217** | **0.604** | **33.3%** | 2881.9 ms |

---

## 3. Canonical Methodological Categorization

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                     OBSERVED FACT                                      │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. R2 Metadata raises Dense Recall@10 from 35.0% to 50.0% (+15.0%) and recovers C147.   │
│ 2. R4 Gloss probe raises pipeline nDCG@10 from 0.485 to 0.604 and doubles Completeness. │
│ 3. R3 moves C15 dense rank from #142 to #70, surfacing it into N=20 candidate pool.     │
│ 4. R2 moves C29 dense rank from #60 to #40, surfacing it into N=20 candidate pool.      │
│ 5. R1/R3 context appending degrades Cross-Encoder Top-10 recall (collapses to 30%-40%).│
└────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                     INTERPRETATION                                     │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. Metadata headers resolve entity ambiguity for dense vector search (C147 recovered).  │
│ 2. Semantic glossing bridges the vocabulary mismatch between QA prompts & Renaissance verse.│
│ 3. Appending raw neighbor chunks creates long noisy blocks that Cross-Encoder penalizes.│
│ 4. Representation alone cannot fix C15/C29/C42/C61 without altering context aggregation.│
└────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                       HYPOTHESIS                                       │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. Parent-Child / Targeted Chunk Boundary segmentation will resolve C29 dialogue       │
│    fragmentation without text length penalties.                                        │
│ 2. Domain-adapted dense embeddings or fine-tuned Cross-Encoder will resolve C15 & C42. │
└────────────────────────────────────────────────────────────────────────────────────────┘
```
