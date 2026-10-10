# Phase 7C — Targeted Candidate Generation Expansion Diagnostic Report

**Corpus:** *The Comedy of Errors* (203 semantic chunks)  
**Date:** September 26, 2026  
**Primary Reranker Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`  
**Query Expansion Strategy:** Validated Q2 Multi-Query Expansion  

---

## Executive Summary

Phase 7A established that Q2 Multi-Query Expansion achieves **75.0% Candidate Union Recall@20**, but suffers from severe RRF ranking truncation (**40.0% Final Recall@10**).  
Phase 7B demonstrated that Cross-Encoder reranking resolves the ranking bottleneck, elevating **Final Recall@10 to 60.0%** (80.0% candidate conversion efficiency). However, five required gold evidence chunks remained completely absent from the base candidate pool at $N=20$: `C15`, `C29`, `C61`, `C42`, and `C147` (**25.0% Candidate Generation Gap**).

The objective of **Phase 7C** was to test whether **targeted structural candidate expansion** (Parent Expansion and Local Neighbor Expansion) can recover these missing evidence chunks before Cross-Encoder reranking, without introducing gold labels or benchmark-specific rules.

### Key Diagnostic Findings

1. **Limited Candidate Recovery:** Parent Expansion recovered **1 missing gold chunk** (`C61` in Q03), increasing candidate recall from **75.0% to 80.0%**. Neighbor Expansion ($\pm 1, \pm 2$) recovered **0 new gold chunks**.
2. **Severe Noise & Ranking Dilution:** Parent Expansion expanded candidate pool size by **2.27x** (adding 272 non-gold candidates across 6 questions). This candidate explosion introduced severe ranking noise, causing **Final Recall@10 to drop from 60.0% down to 50.0%**.
3. **Extremely Low Recovery Efficiency:** Parent Expansion achieved an **Expansion Recovery Efficiency of only 0.37%** (1 new gold chunk per 272 added candidates).
4. **Structural Ceiling Confirmed:** Structural adjacency and parent relationships are **insufficient** to bridge the remaining candidate gap (`C15`, `C29`, `C42`, `C147` remained completely unrecovered).
5. **Architectural Verdict:** Blind structural expansion is counter-productive before Cross-Encoder reranking. Pipeline improvements must focus on **semantic query decomposition and lexical/vocabulary bridging** rather than structural candidate expansion.

---

## 1. Experimental Setup & Frozen Components

To isolate the causal impact of candidate expansion, all downstream and upstream components were kept strictly frozen:

* **Corpus:** *The Comedy of Errors*, 203 semantic chunks (identical chunk IDs, parent IDs, metadata, and gold annotations).
* **Embeddings:** `all-MiniLM-L6-v2`.
* **BM25 Retriever:** Existing Rank-BM25 implementation on exact chunk text.
* **LLM Generator:** `nvidia/nemotron-3-ultra-550b-a55b` (frozen, no generation in Phase 7C).
* **Cross-Encoder Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (CPU, batch_size=32).
* **Final Context Budget:** Top-$K = 10$ chunks.
* **Strict Anti-Leakage Guardrail:** Retrieval and expansion engines operated strictly on user questions and document layout metadata. Gold IDs (`C15`, `C29`, `C61`, `C42`, `C147`) were inspected ONLY during post-hoc evaluation.

---

## 2. Experimental Conditions

The experiment evaluated five candidate-generation conditions across candidate depths $N=20$ (Primary) and $N=50$ (Secondary):

1. **Condition A — Q2 + Cross-Encoder Control (Phase 7B Control):**
   $$\text{Q2 Base Candidate Union } (N=20) \rightarrow \text{Cross-Encoder Reranker} \rightarrow \text{Top-10}$$
2. **Condition B — Q2 + Parent Expansion + CE:**
   $$\text{Q2 Base Union} \rightarrow \text{Identify Parent IDs} \rightarrow \text{Add All Sibling Chunks} \rightarrow \text{Cross-Encoder} \rightarrow \text{Top-10}$$
3. **Condition C1 — Q2 + Neighbor $\pm 1$ Expansion + CE:**
   $$\text{Q2 Base Union} \rightarrow \text{Add Neighbor Chunks } (\pm 1) \rightarrow \text{Cross-Encoder} \rightarrow \text{Top-10}$$
4. **Condition C2 — Q2 + Neighbor $\pm 2$ Expansion + CE:**
   $$\text{Q2 Base Union} \rightarrow \text{Add Neighbor Chunks } (\pm 2) \rightarrow \text{Cross-Encoder} \rightarrow \text{Top-10}$$
5. **Condition D — Q2 + Combined Parent + Neighbor $\pm 1$ Expansion + CE:**
   $$\text{Q2 Base Union} \rightarrow \text{Parent + Neighbor } (\pm 1) \text{ Union} \rightarrow \text{Cross-Encoder} \rightarrow \text{Top-10}$$

---

## 3. Primary Master Comparison Table ($N=20$)

| Condition | Base Cand Recall | Exp Cand Recall | Candidate Exp Factor | Final Recall@10 | Evidence Precision@10 | MRR | nDCG@10 | Evidence Completeness | New Gold Recovered | Added Cand Precision | Recovery Efficiency | Candidate Conversion |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Q2 + CE Control** | **75.0%** | **75.0%** | **1.00x** | **60.0%** | **0.200** | **0.417** | **0.546** | **33.3%** | **0** | **N/A** | **N/A** | **80.0%** |
| **Q2 + Parent Exp + CE** | 75.0% | **80.0%** | 2.27x | 50.0% | 0.167 | 0.394 | 0.519 | 16.7% | **1** (`C61`) | 0.37% | 0.37% | 62.5% |
| **Q2 + Neighbor $\pm 1$ + CE** | 75.0% | 75.0% | 1.11x | 55.0% | 0.183 | 0.417 | 0.532 | 16.7% | 0 | 0.00% | 0.00% | 73.3% |
| **Q2 + Neighbor $\pm 2$ + CE** | 75.0% | 75.0% | 1.41x | 55.0% | 0.183 | 0.417 | 0.532 | 16.7% | 0 | 0.00% | 0.00% | 73.3% |
| **Q2 + Combined Exp + CE** | 75.0% | **80.0%** | 2.27x | 50.0% | 0.167 | 0.394 | 0.519 | 16.7% | **1** (`C61`) | 0.37% | 0.37% | 62.5% |

### Secondary Comparison Table ($N=50$)

| Condition | Base Cand Recall | Exp Cand Recall | Candidate Exp Factor | Final Recall@10 | Evidence Precision@10 | MRR | nDCG@10 | Evidence Completeness | New Gold Recovered | Recovery Efficiency | Candidate Conversion |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Q2 + CE Control** | 75.0% | 75.0% | 1.00x | 60.0% | 0.200 | 0.489 | 0.545 | 16.7% | 0 | N/A | 80.0% |
| **Q2 + Parent Exp + CE** | 75.0% | 80.0% | 1.48x | 60.0% | 0.200 | 0.489 | 0.545 | 16.7% | 1 | 0.72% | 75.0% |
| **Q2 + Neighbor $\pm 1$ + CE** | 75.0% | 75.0% | 1.08x | 60.0% | 0.200 | 0.489 | 0.545 | 16.7% | 0 | 0.00% | 80.0% |
| **Q2 + Neighbor $\pm 2$ + CE** | 75.0% | 75.0% | 1.28x | 60.0% | 0.200 | 0.489 | 0.545 | 16.7% | 0 | 0.00% | 80.0% |
| **Q2 + Combined Exp + CE** | 75.0% | 80.0% | 1.48x | 60.0% | 0.200 | 0.489 | 0.545 | 16.7% | 1 | 0.72% | 75.0% |

---

## 4. Per-Question Expansion Analysis ($N=20$)

| Question | Base Cand Recall | Parent Exp Recall | Neighbor $\pm 1$ Recall | CE Top-10 Recall (Control) | CE Top-10 Recall (Parent Exp) | New Gold Recovered | Expansion Provenance & Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Q01** | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | None | All gold chunks (`C4`, `C2`) already present in base pool. |
| **Q02** | 100.0% | 100.0% | 100.0% | 75.0% | 75.0% | None | All gold chunks (`C2`, `C4`, `C10`, `C12`) already present in base pool. |
| **Q03** | 25.0% | **50.0%** | 25.0% | 25.0% | 25.0% | **`C61`** | **Parent Exp recovered `C61`** via seed `C55` (sharing Act 2 Scene 1 parent). CE ranked `C61` #16 (missed Top-10). |
| **Q04** | 100.0% | 100.0% | 100.0% | 100.0% | 66.7% | None | Parent Exp introduced noise; `C6` dropped from Top-10 (#10 $\rightarrow$ #11). |
| **Q05** | 66.7% | 66.7% | 66.7% | 33.3% | 0.0% | None | `C42` remained absent. Parent noise demoted `C12` out of Top-10. |
| **Q06** | 75.0% | 75.0% | 75.0% | 50.0% | 50.0% | None | `C147` remained absent. Base candidates `C105`, `C107` retained in Top-10. |

---

## 5. Critical Evidence Provenance & Tracking

Detailed tracking of the 5 diagnostic unreachable gold chunks:

| Gold Chunk | Target Question | Base Pool Status | Recovered by Parent Exp? | Recovered by Neighbor Exp? | Provenance Source | CE Score (Expanded Pool) | CE Rank | Top-10 Status | Root Cause & Failure Analysis |
| :--- | :--- | :---: | :---: | :---: | :--- | :---: | :---: | :---: | :--- |
| **`C15`** | Q03 (Primary) | Absent | NO | NO | None | — | — | NO | **Semantic & Layout Isolation.** Dialogue in Act 1 Scene 2 describing Syracusan origin had no seed chunks in base pool. |
| **`C29`** | Q03 (Supp) | Absent | NO | NO | None | — | — | NO | **Soliloquy Isolation.** Soliloquy on Ephesus witchcraft shared no parent or neighbor seeds in base pool. |
| **`C61`** | Q03 (Supp) | Absent | **YES** | NO | `source = parent`<br>`seed = C55`<br>`parent_id = Act 2 Scene 1` | -0.8227 | #16 | NO | **Recovered into candidate pool**, but CE ranked it #16 due to competition from 272 expanded candidates. |
| **`C42`** | Q05 (Primary) | Absent | NO | NO | None | — | — | NO | **Dialogue Vocabulary Mismatch.** Money entrustment dialogue in Act 1 Scene 2 had no layout seeds. |
| **`C147`** | Q06 (Supp) | Absent | NO | NO | None | — | — | NO | **Lexical Disconnect.** Legal arrest warrant text in Act 4 Scene 1 shared no parent headers with goldsmith scene. |

---

## 6. Neighbor Context Expansion vs. Neighbor Candidate Expansion

Phase 2/3 evaluated **Neighbor Context Expansion** (blindly concatenating adjacent chunk text to the final context window sent to the LLM).  
Phase 7C evaluated **Neighbor Candidate Expansion** (adding neighboring chunks to the candidate set for Cross-Encoder reranking).

### Comparison Matrix

| Evaluation Dimension | Neighbor Context Expansion (Phase 2/3) | Neighbor Candidate Expansion (Phase 7C) |
| :--- | :--- | :--- |
| **Pipeline Position** | Post-retrieval (LLM context window) | Pre-reranking (Candidate Pool generation) |
| **Candidate Union Impact** | None (does not change candidate recall) | Increases candidate pool size by +11% ($\pm 1$) to +41% ($\pm 2$) |
| **New Gold Evidence Recovered** | 0 new gold chunks | **0 new gold chunks** |
| **Recall Impact** | 0% change | -5.0% Recall@10 drop ($N=20$) due to ranking noise |
| **Context Window Growth** | +100% to +200% context token bloat | +11% to +41% Cross-Encoder pair scoring overhead |
| **Diagnostic Verdict** | Ineffective (causes prompt dilution) | **Ineffective** (fails to recover missing evidence) |

---

## 7. Candidate Quality & Noise Growth Analysis

The objective of candidate expansion is to increase candidate recall without causing uncontrolled candidate explosion.

```
Base Candidate Pool (N=20): 214 Total Candidates across 6 Questions (35.7 / query)
                 │
                 ├── Parent Expansion: +272 New Candidates (+127.1% Growth)
                 │      ├── New Gold Evidence: 1 Chunk (C61)
                 │      ├── New Non-Gold Candidates: 271 Chunks
                 │      └── Noise-to-Signal Ratio: 271:1 (0.37% Added Precision)
                 │
                 └── Neighbor ±1 Expansion: +23 New Candidates (+10.7% Growth)
                        ├── New Gold Evidence: 0 Chunks
                        └── Noise-to-Signal Ratio: Infinity (0.00% Added Precision)
```

### Noise Impact on Cross-Encoder Selection
When Parent Expansion expanded candidate pool size from 35.7 to 81.0 candidates per query:
* Non-relevant section chunks entered the candidate pool with moderate Cross-Encoder scores (-0.5 to +0.2).
* These noise chunks displaced valid supporting evidence (`C12` in Q05, `C6` in Q04), causing **Final Recall@10 to drop from 60.0% to 50.0%**.

---

## 8. Reranker Scaling & Latency Breakdown

| Candidate Expansion Condition | Mean Candidates Scored / Query | Mean CE Latency (ms) | Mean Total Pipeline Latency (ms) | Latency Factor vs. Control |
| :--- | ---: | ---: | ---: | ---: |
| **Q2 + CE Control** | 35.7 pairs | 1,951.2 ms | **3,284.9 ms** | **1.00x** |
| **Q2 + Parent Expansion + CE** | 81.0 pairs | 3,030.6 ms | **4,364.8 ms** | **1.33x** |
| **Q2 + Neighbor $\pm 1$ + CE** | 39.5 pairs | 1,973.4 ms | **3,307.6 ms** | **1.01x** |
| **Q2 + Neighbor $\pm 2$ + CE** | 50.3 pairs | 2,146.8 ms | **3,480.6 ms** | **1.06x** |
| **Q2 + Combined Expansion + CE** | 81.0 pairs | 3,002.9 ms | **4,337.1 ms** | **1.32x** |

---

## 9. Decision Gates Evaluation

* **Gate A — Candidate Recovery (PARTIALLY TRIGGERED):**  
  Parent Expansion recovered `C61` into the candidate pool (expanding candidate recall to 80.0%), but failed to recover `C15`, `C29`, `C42`, or `C147`.
* **Gate B — Candidate Recovery Without Top-10 Improvement (TRIGGERED):**  
  Parent Expansion increased candidate recall from 75.0% to 80.0%, but **Final Recall@10 dropped from 60.0% to 50.0%** because the 2.27x candidate expansion introduced excessive ranking noise that diluted Cross-Encoder precision.
* **Gate C — Structural Expansion Insufficiency (CONFIRMED):**  
  Structural relationships (parent layout sections and local chunk adjacency) are fundamentally **insufficient** to solve candidate generation failure for 4 out of 5 missing gold chunks.
* **Gate D — Strong Candidate + Ranking Improvement (FAILED):**  
  No candidate expansion condition simultaneously improved candidate recall AND final Top-10 recall. **Condition A (Q2 + CE Control) remains the superior architecture.**

---

## 10. Answers to Required Final Analysis Questions

1. **Did candidate expansion recover previously unreachable evidence?**  
   **Partially.** Parent expansion recovered 1 out of 5 missing gold chunks (`C61`). Neighbor expansion recovered 0 chunks.
2. **Which mechanism recovered it?**  
   Parent Expansion (via seed chunk `C55` in Act 2 Scene 1).
3. **How much did candidate recall increase?**  
   Candidate recall increased from **75.0% to 80.0%** (+5.0 percentage points).
4. **How many additional candidates were introduced?**  
   272 new candidate chunks across 6 questions (an average of +45.3 candidates per question).
5. **What was the expansion factor?**  
   **2.27x** expansion factor for Parent Expansion.
6. **Did Cross-Encoder Top-10 recall increase?**  
   **No. It decreased from 60.0% to 50.0%** due to candidate noise dilution.
7. **Which questions benefited?**  
   None. Q03 candidate recall improved (25% $\rightarrow$ 50%), but `C61` was ranked #16 by CE, failing to reach Top-10.
8. **Which evidence remains unreachable?**  
   `C15` (Q03), `C29` (Q03), `C42` (Q05), `C147` (Q06).
9. **Did expansion introduce excessive noise?**  
   **Yes.** Added Candidate Precision was only **0.37%** (271 noise chunks vs. 1 gold chunk).
10. **What was the latency cost?**  
    CE latency increased by **+1,079.4 ms** (+55.3% increase), bringing total pipeline latency to 4.36s.
11. **What is the new candidate-generation ceiling?**  
    **80.0%** with Parent Expansion; **75.0%** with base Q2 expansion.
12. **What is the new final Top-10 ceiling?**  
    **60.0%** (achieved by Q2 + CE Control).
13. **Which candidate-generation mechanism should proceed to the next experiment?**  
    **Neither Parent nor Neighbor expansion should proceed.** Q2 Multi-Query Base Candidate Generation + Cross-Encoder Reranking remains the frozen control baseline.
14. **Is another retrieval mechanism required?**  
    **Yes.** To recover `C15`, `C29`, `C42`, and `C147`, the architecture requires **semantic aspect decomposition, hypothetical document embeddings (HyDE), or dense-lexical vocabulary bridging** rather than structural layout expansion.

---

## 11. Final Architecture Goal & Roadmap

The investigation has established clear boundaries for candidate generation and reranking:

```text
                                USER QUESTION
                                      │
                                      ▼
                            Q2 MULTI-QUERY EXPANSION
                                      │
                                      ▼
                            BASE CANDIDATE GENERATION
                                      │
               ┌──────────────────────┴──────────────────────┐
               │                                             │
      Structural Expansion                           Semantic / Lexical Bridging
    (Parent/Neighbor Exp)                       (Aspect Decomposition / HyDE)
               │                                             │
      [REJECTED: +127% noise,                        [RECOMMENDED FOR PHASE 8]
       Recall drops 60% -> 50%]                              │
                                                             ▼
                                                      CANDIDATE UNION
                                                             │
                                                             ▼
                                                   CROSS-ENCODER RERANKER
                                                             │
                                                             ▼
                                                          TOP-10
```

### Recommendation for Next Phase
Structural candidate expansion is rejected for production pipelines. Future work must focus on **semantic query transformation and dense-lexical bridging** to capture the remaining 20% of unreachable evidence without inflating the candidate pool size.
