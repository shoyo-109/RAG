# Phase 7B — Cross-Encoder Reranking Diagnostic Report

**Corpus:** *The Comedy of Errors* (203 semantic chunks)  
**Date:** September 26, 2026  
**Primary Reranker Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`  
**Candidate Generation Strategy:** Validated Q2 Multi-Query Expansion  

---

## Executive Summary

Phase 7A established that Q2 Multi-Query Expansion effectively expanded candidate evidence recall from **60.0% to 75.0%** at candidate depth $N=20$. However, the standard Reciprocal Rank Fusion (RRF 80/20) mechanism suffered from severe truncation loss, failing to promote any of the newly recovered candidate evidence into the final Top-10 context window (**Final Recall@10 stayed flat at 40.0%**).

Phase 7B evaluated whether replacing RRF with a **Cross-Encoder Reranker** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) on the Q2 candidate union resolves this ranking bottleneck.

### Key Experimental Findings

1. **Substantial Recall Gain:** Q2 + Cross-Encoder reranking increased final **Top-10 Evidence Recall from 40.0% to 60.0%** (+20.0 percentage points absolute gain).
2. **High Reranker Recovery Rate:** Out of the 7 gold evidence chunks present in the candidate pool but missed by RRF Top-10, the Cross-Encoder successfully promoted **5 chunks into Top-10 (71.4% Reranker Recovery Rate)**.
3. **Significant Ranking Metric Boost:** 
   - **nDCG@10** improved from **0.382 to 0.546** (+42.9% relative gain).
   - **MRR** improved from **0.132 to 0.417** (+215.9% relative gain).
   - **Evidence Completeness** doubled from **16.7% to 33.3%**.
   - **Candidate-to-Top-10 Conversion** increased from **53.3% to 80.0%**.
4. **Hard Retrieval Ceiling Identified:** 5 gold chunks (`C15`, `C29`, `C42`, `C61`, `C147`) remained completely absent from the Q2 candidate union at $N=20$ (**25.0% Candidate-Generation Failure**). The Cross-Encoder achieved an **80.0% conversion rate of all available candidate gold chunks** (12 out of 15 available gold chunks placed into Top-10).

---

## 1. Experimental Setup & Frozen Components

To isolate the causal impact of Cross-Encoder reranking, all upstream and downstream pipeline components were kept strictly frozen:

* **Corpus:** *The Comedy of Errors*, 203 semantic chunks (identical chunk IDs, parent IDs, and gold annotations).
* **Embedding Model:** `all-MiniLM-L6-v2` (384-dim dense vectors).
* **BM25 Retriever:** Existing Rank-BM25 implementation on exact chunk text.
* **LLM Generator:** `nvidia/nemotron-3-ultra-550b-a55b` (temp=0.7, top_p=0.95, max_tokens=1024). Generation was frozen during Phase 7B.
* **RRF Parameters:** $k = 60$, Vector Weight = 0.80, BM25 Weight = 0.20.
* **Context Budget:** Top-$K = 10$ chunks.
* **Metric Leakage Control:** The Cross-Encoder evaluated solely `(user_question, candidate_chunk_text)` pairs. Gold labels, claim maps, and chunk IDs were completely concealed from the retrieval/reranking pipeline.

### Cross-Encoder System Specifications
* **Model Name:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
* **Inference Device:** CPU
* **Batch Size:** 32
* **Mean Candidate Pairs Scored per Question:** 35.7 pairs ($N=20$)
* **Mean Reranking Latency:** 1,972.0 ms per question

---

## 2. Experimental Conditions

The experiment evaluated three controlled conditions across candidate pool depths $N=20$ (Primary) and $N=50$ (Secondary):

* **Condition A — Baseline Q0 + RRF (Phase 5/6 Baseline):**
  $$\text{Original Question (Q0)} \rightarrow \text{Semantic top-}N + \text{BM25 top-}N \rightarrow \text{RRF (80/20)} \rightarrow \text{Top-10}$$
* **Condition B — Q2 + RRF (Phase 7A Control):**
  $$\text{Original Question} \rightarrow \text{Q2 Multi-Query Expansion} \rightarrow \text{Candidate Union} \rightarrow \text{RRF (80/20)} \rightarrow \text{Top-10}$$
* **Condition C — Q2 + Cross-Encoder (Phase 7B Target):**
  $$\text{Original Question} \rightarrow \text{Q2 Multi-Query Expansion} \rightarrow \text{Candidate Union} \rightarrow \text{Cross-Encoder Reranker} \rightarrow \text{Top-10}$$

---

## 3. Primary Comparison Results ($N=20$)

| Condition | Candidate Recall@20 | Final Recall@10 | Evidence Precision@10 | Primary Precision@10 | MRR | nDCG@10 | Evidence Completeness | Required Evidence Completeness | Reranker Recovery Rate | Candidate Conversion |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Q0 + RRF (Baseline)** | 60.0% | 40.0% | 0.133 | 0.067 | 0.307 | 0.399 | 16.7% | 40.0% | N/A | 66.7% |
| **Q2 + RRF (Control)** | 75.0% | 40.0% | 0.133 | 0.083 | 0.132 | 0.382 | 16.7% | 40.0% | N/A | 53.3% |
| **Q2 + Cross-Encoder** | **75.0%** | **60.0%** | **0.200** | **0.100** | **0.417** | **0.546** | **33.3%** | **60.0%** | **71.4%** | **80.0%** |

### Secondary Comparison Results ($N=50$)

| Condition | Candidate Recall@50 | Final Recall@10 | Evidence Precision@10 | Primary Precision@10 | MRR | nDCG@10 | Evidence Completeness | Required Evidence Completeness | Reranker Recovery Rate | Candidate Conversion |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Q0 + RRF (Baseline)** | 75.0% | 20.0% | 0.067 | 0.033 | 0.186 | 0.262 | 0.0% | 20.0% | N/A | 26.7% |
| **Q2 + RRF (Control)** | 75.0% | 15.0% | 0.050 | 0.033 | 0.089 | 0.278 | 0.0% | 15.0% | N/A | 20.0% |
| **Q2 + Cross-Encoder** | **75.0%** | **60.0%** | **0.200** | **0.100** | **0.489** | **0.545** | **16.7%** | **60.0%** | **75.0%** | **80.0%** |

---

## 4. Per-Question Diagnostic Breakdown ($N=20$)

| Question | Total Gold Evidence | Gold Evidence Chunks | In Candidate Pool? | RRF Rank | CE Rank | Top-10 RRF? | Top-10 CE? | Detailed Outcome |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Q01** | 2 | C4 (Pri), C2 (Supp) | 2 / 2 | C4: #8<br>C2: #7 | C4: #6<br>C2: #5 | YES (2/2) | YES (2/2) | **Maintained 100% Recall.** Both gold chunks retained in Top-10. |
| **Q02** | 4 | C2 (Pri), C4 (Supp), C10 (Supp), C12 (Supp) | 4 / 4 | C2: #8<br>C4: #21<br>C10: #12<br>C12: #5 | C2: #6<br>C4: #1<br>C10: #2<br>C12: #23 | 50.0% (2/4) | **75.0% (3/4)** | **+25.0% Recall.** CE promoted C4 (#21→#1) & C10 (#12→#2). C12 demoted (#5→#23). |
| **Q03** | 4 | C15 (Pri), C24 (Pri), C29 (Supp), C61 (Supp) | 1 / 4 | C15: Absent<br>C24: #20<br>C29: Absent<br>C61: Absent | C15: Absent<br>C24: #5<br>C29: Absent<br>C61: Absent | 0.0% (0/4) | **25.0% (1/4)** | **+25.0% Recall.** CE promoted C24 (#20→#5). C15/C29/C61 absent from pool. |
| **Q04** | 3 | C4 (Pri), C6 (Supp), C8 (Supp) | 3 / 3 | C4: #4<br>C6: #11<br>C8: #5 | C4: #4<br>C6: #10<br>C8: #2 | 66.7% (2/3) | **100.0% (3/3)** | **+33.3% Recall.** CE promoted C6 (#11→#10) into Top-10, achieving 100% Completeness. |
| **Q05** | 3 | C42 (Pri), C65 (Pri), C12 (Supp) | 2 / 3 | C42: Absent<br>C65: >30<br>C12: #18 | C42: Absent<br>C65: #29<br>C12: #10 | 0.0% (0/3) | **33.3% (1/3)** | **+33.3% Recall.** CE promoted C12 (#18→#10). C42 absent from candidate pool. |
| **Q06** | 4 | C105 (Pri), C107 (Pri), C147 (Supp), C155 (Supp) | 3 / 4 | C105: #9<br>C107: #5<br>C147: Absent<br>C155: #16 | C105: #2<br>C107: #3<br>C147: Absent<br>C155: #29 | 50.0% (2/4) | 50.0% (2/4) | **Recall Neutral.** CE boosted C105 & C107 to Top-3 (#2, #3). C155 remained outside (#29). |

---

## 5. Ranking Recovery Analysis

Every gold evidence chunk present in the candidate pool but missed by RRF Top-10 was tracked to classify the Cross-Encoder's ranking behavior:

| Category | Definition | Gold Chunks | Count |
| :--- | :--- | :--- | :---: |
| **Case A — Promoted into Top-10** | Present in candidate pool, missed by RRF, successfully ranked into CE Top-10 | **C4 (Q02), C10 (Q02), C24 (Q03), C6 (Q04), C12 (Q05)** | **5** |
| **Case B — Rank Improved (Outside Top-10)** | Rank improved by CE, but final rank remained outside Top-10 ($>10$) | None | **0** |
| **Case C — Rank Unchanged / Unranked** | CE score did not meaningfully improve rank relative to pool size | **C65 (Q05)** (RRF >30 $\rightarrow$ CE #29) | **1** |
| **Case D — Ranked Worse / Demoted** | Ranked worse than RRF or demoted out of Top-10 | **C12 (Q02)** (RRF #5 $\rightarrow$ CE #23), **C155 (Q06)** (RRF #16 $\rightarrow$ CE #29) | **2** |

### Key Insights from Ranking Recovery
* **Reranker Recovery Rate:** $\frac{\text{Case A Promoted}}{\text{RRF Misses}} = \frac{5}{7} = \mathbf{71.4\%}$.
* **Recovery Impact:** The Cross-Encoder successfully recovered 5 out of the 7 candidate-present gold chunks that RRF truncated.
* **Trade-Off Analysis:** In Q02, the Cross-Encoder demoted `C12` (Supporting evidence) out of Top-10, but simultaneously promoted `C4` (Supporting) and `C10` (Supporting) into ranks #1 and #2, resulting in a net recall increase for Q02 from 50.0% to 75.0%.

---

## 6. Candidate Ceiling vs. Ranking Ceiling Analysis

Phase 7B explicitly distinguished candidate-generation bottlenecks from ranking/selection bottlenecks:

```
Total Benchmark Gold Evidence: 20 Chunks (100.0%)
            │
            ├── Candidate Generation Stage (Q2 Multi-Query Union @ N=20)
            │      ├── Retrieved into Candidate Pool: 15 Chunks (75.0% Candidate Ceiling)
            │      └── Candidate Generation Failures: 5 Chunks (25.0% Unreachable Ceiling)
            │             └─ C15 (Q03), C29 (Q03), C61 (Q03), C42 (Q05), C147 (Q06)
            │
            └── Final Ranking & Selection Stage (Top-10 Selection)
                   ├── RRF Top-10 Selection: 8 Chunks (40.0% Final Recall) -> Truncation Loss = 35.0%
                   └── Cross-Encoder Top-10 Selection: 12 Chunks (60.0% Final Recall) -> Conversion Rate = 80.0%
```

### Analysis of Unreachable Gold Evidence ($N=20$)
The following 5 gold evidence chunks remained completely absent from the Q2 candidate union:
1. **`C15` (Q03 - Primary):** Syracusan visitor context in Ephesus.
2. **`C29` (Q03 - Supporting):** Antipholus of Syracuse internal monologue on Ephesus witchcraft.
3. **`C61` (Q03 - Supporting):** Adriana's dinner invitation scene.
4. **`C42` (Q05 - Primary):** Syracuse twin entrusts money to Dromio of Syracuse.
5. **`C147` (Q06 - Supporting):** Legal arrest warrant execution text.

**Conclusion:** A Cross-Encoder reranker operates strictly on the candidate union. Because these 5 chunks never entered the candidate pool, the Cross-Encoder hit a **hard theoretical ceiling of 75.0%**. It converted **12 out of 15 available candidate gold chunks (80.0% candidate conversion efficiency)**.

---

## 7. Latency and Performance Breakdown

Reranking introduce a measurable latency overhead. The stage-by-stage latency profile across all six benchmark questions is reported below:

| Retrieval Pipeline Stage | Mean Latency (ms) | Percentage of Total Time |
| :--- | ---: | ---: |
| **Q2 LLM Query Generation** | 1,200.0 ms | 36.3% |
| **Vector & BM25 Retrieval ($N=20$)** | 132.4 ms | 4.0% |
| **RRF Fusion** | < 0.1 ms | 0.0% |
| **Cross-Encoder Inference** | 1,972.0 ms | 59.7% |
| **Total Pipeline Latency** | **3,304.4 ms** | **100.0%** |

### Tradeoff Assessment
* **Quality Improvement:** +20.0 percentage points in Final Recall@10 (40.0% $\rightarrow$ 60.0%), +42.9% relative gain in nDCG@10.
* **Latency Cost:** Added ~1.97s per query for Cross-Encoder scoring of 35.7 candidate pairs on CPU.
* **Verdict:** The latency overhead (~1.97s) is well within acceptable bounds for off-line or agentic multi-hop RAG pipelines, providing significant evidence quality improvements.

---

## 8. Decision Gates Evaluation

* **Gate A — Strong Improvement (PASSED):**  
  $$\text{Q2 + Cross-Encoder Recall@10 (60.0\%)} > \text{Q2 + RRF Recall@10 (40.0\%)}$$  
  The improvement came directly from promoting candidates already present in the Q2 candidate pool (Recovery Rate = 71.4%). **Conclusion: RRF ranking/selection was a genuine primary bottleneck.**

* **Gate B — Candidate Recall Improves but Final Recall Does Not (NOT TRIGGERED):**  
  The Cross-Encoder successfully converted candidate recall into final context recall.

* **Gate C — Per-Question Benefit Analysis:**  
  - **Benefited:** Q02 (+25%), Q03 (+25%), Q04 (+33%), Q05 (+33%).
  - **Neutral:** Q01 (100% maintained), Q06 (50% maintained).
  - **Conclusion:** Both direct multi-hop questions (Q04) and causal/relationship questions (Q02, Q03, Q05) benefited substantially from Cross-Encoder scoring.

* **Gate D — Candidate Generation Ceiling (CONFIRMED):**  
  Gold chunks `C15`, `C29`, `C61`, `C42`, and `C147` remained outside the candidate pool at $N=20$. Cross-Encoder reranking cannot solve candidate-generation failure.

---

## 9. Answers to Required Diagnostic Questions

1. **Did the Cross-Encoder improve final Top-10 evidence recall?**  
   **Yes.** Final Recall@10 increased from **40.0% to 60.0%** (+20.0 percentage points absolute increase).
2. **How many previously truncated gold chunks were recovered?**  
   **5 gold chunks** (`C4`, `C10`, `C24`, `C6`, `C12`) were promoted into Top-10 (Reranker Recovery Rate = **71.4%**).
3. **Did it improve evidence completeness?**  
   **Yes.** Evidence Completeness doubled from **16.7% (1/6 questions) to 33.3% (2/6 questions)**.
4. **Did it improve nDCG/MRR?**  
   **Yes.** **nDCG@10** increased from **0.382 to 0.546** (+42.9%), and **MRR** increased from **0.132 to 0.417** (+215.9%).
5. **Which questions benefited?**  
   Q02, Q03, Q04, and Q05 all achieved higher evidence recall.
6. **Which evidence remained unreachable?**  
   `C15`, `C29`, `C61` (Q03), `C42` (Q05), and `C147` (Q06).
7. **What is the new candidate-generation ceiling?**  
   **75.0%** at candidate depth $N=20$.
8. **What is the new ranking ceiling?**  
   **60.0%** at $N=20$ (80.0% conversion of available candidate pool evidence).
9. **What latency cost did reranking introduce?**  
   An average of **1,972.0 ms** per query (bringing total pipeline latency to ~3.30s).
10. **Is there empirical justification for Phase 7C?**  
    **Yes.** While Cross-Encoder reranking successfully solved the ranking bottleneck (converting 80% of candidates), 25% of required evidence remains missing from the candidate generation stage (`C15`, `C29`, `C61`, `C42`, `C147`). Phase 7C should focus on targeted candidate generation mechanisms (such as Parent-Child Expansion or Neighboring Chunk Contextual Expansion) to capture these missing chunks before reranking.

---

## 10. Summary & Next Steps

Phase 7B proves that **Cross-Encoder reranking resolves the RRF selection bottleneck**, boosting Top-10 evidence recall to **60.0%**. 

To reach $>80\%$ evidence recall, the pipeline must address the remaining candidate-generation failures. We recommend proceeding to **Phase 7C (Parent/Neighbor Context Expansion)** to bring the candidate pool recall closer to 100% prior to Cross-Encoder scoring.
