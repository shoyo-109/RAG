# Phase 11A — Query–Chunk Representation Diagnosis Detailed Results

## Executive Summary

Phase 11A was conducted as an unbiased, controlled diagnostic experiment over the complete frozen benchmark (**6 questions × 4 conditions = 24 retrieval evaluations**) to determine whether candidate-generation failures for primary evidence targets **C15** ($Q_{03}$) and **C42** ($Q_{05}$) are driven by a **query–chunk representation mismatch**.

Following strict scientific anti-leakage safeguards, `C15` and `C42` were treated strictly as **post-hoc diagnostic observations**. Dense vector retrieval operated without hardcoded target IDs, target boosting, candidate filtering, or gold evidence awareness.

### Primary Diagnostic Conclusion

> **Cross-Target Decision**: `NOT SUPPORTED`

- **Target C15** ($Q_{03}$, Primary Evidence):
  - **Diagnosis**: `REPRESENTATION_INTERVENTION_NOT_SUFFICIENT_FOR_ACCESSIBILITY`
  - **Strongest Intervention**: `A3` (Combined query reformulation + semantic glossing)
  - **Movement**: Rank moved from **#142** ($A_0$) to **#52** ($A_3$), $\Delta\text{Rank} = +90$. However, it remained in the `>50` inaccessible band.
- **Target C42** ($Q_{05}$, Primary Evidence):
  - **Diagnosis**: `REPRESENTATION_INTERVENTION_NOT_SUFFICIENT_FOR_ACCESSIBILITY`
  - **Strongest Intervention**: `none`
  - **Movement**: Rank shifted from **#99** ($A_0$) to **#101** ($A_1$), **#117** ($A_2$), and **#122** ($A_3$). No intervention improved candidate accessibility.

---

## 1. Experimental Controls & Setup

- **Corpus**: Canonical 203 chunks from `The Comedy of Errors document.pdf`. Chunk boundaries, IDs ($C_0 \dots C_{202}$), and raw text are unchanged.
- **Embedding Model ($E_0$)**: `sentence-transformers/all-MiniLM-L6-v2`. All embeddings L2-normalized; cosine similarity computed via dot product.
- **Canonical Gloss Cache**: Loaded from `evaluation/results/r4_semantic_glosses_cache.json`. Hash verified: `ee4243c2c0b575c87fe11a9f73f81de3`.
- **Retrieval Scope**: Dense vector retrieval only. No BM25, No RRF, No Cross-Encoder reranking.
- **Anti-Leakage Safeguard**: Condition $A_2$ and $A_3$ queries were derived strictly from `question_id` + `original benchmark question`. Zero target chunk text, chunk IDs, or gold evidence annotations were used.

### Diagnostic Conditions Matrix:
- **`A0`**: Original Benchmark Question $\to$ $B_0$ (Raw Text) — *Frozen Control*
- **`A1`**: Original Benchmark Question $\to$ $B_2$ (Gloss + Raw Text) — *Corpus-side Intervention*
- **`A2`**: Neutral Semantic Reformulation $\to$ $B_0$ (Raw Text) — *Query-side Intervention*
- **`A3`**: Neutral Semantic Reformulation $\to$ $B_2$ (Gloss + Raw Text) — *Combined Intervention*

---

## 2. Canonical Neutral Query Reformulations ($A_2 / A_3$)

| Question ID | Original Benchmark Question | Neutral Semantic Reformulation ($A_2 / A_3$) |
| :---: | :--- | :--- |
| **Q01** | Who is Aegeon and what is his origin? | *Identify the merchant named Aegeon and describe his native homeland and background.* |
| **Q02** | Why was Aegeon sentenced to death in Ephesus? | *Explain the legal reasons and local laws causing a Syracusian merchant to receive a capital sentence upon arriving in Ephesus.* |
| **Q03** | Why does Antipholus of Syracuse fail to recognize Adriana when she addresses him as her husband? | *Explain why a visiting traveler from Syracuse is confused when a local woman claims he is her spouse.* |
| **Q04** | How did Aegeon come to have two pairs of identical twin sons and twin servants? | *Describe the birth of the two sets of identical twin children and how one pair was acquired to serve the other.* |
| **Q05** | How do Antipholus of Syracuse and Antipholus of Ephesus differ in their initial interactions with Dromio of Syracuse? | *Contrast how the two different twin brothers interact with and treat the Syracusian servant during their first meetings.* |
| **Q06** | Why is Antipholus of Ephesus arrested, and what role does Angelo the goldsmith play in the debt dispute? | *Explain the cause of the Ephesian twin's arrest regarding unpaid money and the jeweler's involvement in the transaction.* |

---

## 3. Complete 24-Run Factorial Benchmark Results

The table below presents the exact empirical metrics measured across all 24 retrieval evaluations:

| Run ID | QID | Cond | Hit@10 | Rec@10 | Rec@20 | Rec@50 | Prec@10 | MRR | nDCG@10 | EvComp% | C15 Sim | C15 Rank | C15 Band | C42 Sim | C42 Rank | C42 Band |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Q01_A0** | Q01 | A0 | 1 | 50.0% | 50.0% | 100.0% | 0.100 | 0.100 | 0.228 | 0.0% | 0.1698 | #91 | `>50` | 0.1102 | #153 | `>50` |
| **Q01_A1** | Q01 | A1 | 1 | 50.0% | 100.0% | 100.0% | 0.100 | 0.200 | 0.305 | 100.0% | 0.1925 | #57 | `>50` | 0.1265 | #129 | `>50` |
| **Q01_A2** | Q01 | A2 | 1 | 50.0% | 100.0% | 100.0% | 0.100 | 0.125 | 0.248 | 100.0% | 0.2100 | #66 | `>50` | 0.1069 | #159 | `>50` |
| **Q01_A3** | Q01 | A3 | 1 | 100.0% | 100.0% | 100.0% | 0.200 | 0.250 | 0.437 | 100.0% | 0.3069 | #18 | `11–20` | 0.1308 | #154 | `>50` |
| **Q02_A0** | Q02 | A0 | 1 | 25.0% | 75.0% | 75.0% | 0.100 | 0.200 | 0.099 | 0.0% | 0.2851 | #83 | `>50` | 0.1306 | #168 | `>50` |
| **Q02_A1** | Q02 | A1 | 1 | 75.0% | 75.0% | 100.0% | 0.300 | 0.500 | 0.423 | 0.0% | 0.2848 | #67 | `>50` | 0.1392 | #153 | `>50` |
| **Q02_A2** | Q02 | A2 | 1 | 100.0% | 100.0% | 100.0% | 0.400 | 0.500 | 0.633 | 100.0% | 0.2684 | #63 | `>50` | 0.1317 | #156 | `>50` |
| **Q02_A3** | Q02 | A3 | 1 | 100.0% | 100.0% | 100.0% | 0.400 | 0.500 | 0.698 | 100.0% | 0.3548 | #23 | `21–50` | 0.1666 | #153 | `>50` |
| **Q03_A0** | Q03 | A0 | 1 | 25.0% | 25.0% | 50.0% | 0.100 | 0.250 | 0.212 | 0.0% | 0.1986 | #142 | `>50` | 0.2512 | #126 | `>50` |
| **Q03_A1** | Q03 | A1 | 1 | 25.0% | 25.0% | 50.0% | 0.100 | 0.250 | 0.212 | 0.0% | 0.2556 | #118 | `>50` | 0.2037 | #136 | `>50` |
| **Q03_A2** | Q03 | A2 | 0 | 0.0% | 25.0% | 50.0% | 0.000 | 0.059 | 0.000 | 0.0% | 0.0922 | #120 | `>50` | 0.1311 | #88 | `>50` |
| **Q03_A3** | Q03 | A3 | 1 | 25.0% | 25.0% | 25.0% | 0.100 | 0.100 | 0.142 | 0.0% | 0.1974 | #52 | `>50` | 0.1251 | #127 | `>50` |
| **Q04_A0** | Q04 | A0 | 1 | 66.7% | 66.7% | 100.0% | 0.200 | 0.500 | 0.569 | 0.0% | 0.1506 | #92 | `>50` | 0.0222 | #194 | `>50` |
| **Q04_A1** | Q04 | A1 | 1 | 100.0% | 100.0% | 100.0% | 0.300 | 0.500 | 0.587 | 100.0% | 0.2018 | #63 | `>50` | 0.0583 | #194 | `>50` |
| **Q04_A2** | Q04 | A2 | 1 | 100.0% | 100.0% | 100.0% | 0.300 | 0.333 | 0.524 | 100.0% | 0.1233 | #62 | `>50` | 0.0245 | #181 | `>50` |
| **Q04_A3** | Q04 | A3 | 1 | 66.7% | 100.0% | 100.0% | 0.200 | 0.500 | 0.285 | 0.0% | 0.1742 | #53 | `>50` | 0.0878 | #160 | `>50` |
| **Q05_A0** | Q05 | A0 | 0 | 0.0% | 33.3% | 66.7% | 0.000 | 0.083 | 0.000 | 0.0% | 0.3227 | #88 | `>50` | 0.2811 | #99 | `>50` |
| **Q05_A1** | Q05 | A1 | 0 | 0.0% | 33.3% | 66.7% | 0.000 | 0.067 | 0.000 | 0.0% | 0.3184 | #80 | `>50` | 0.2545 | #101 | `>50` |
| **Q05_A2** | Q05 | A2 | 0 | 0.0% | 33.3% | 66.7% | 0.000 | 0.077 | 0.000 | 0.0% | 0.1790 | #69 | `>50` | 0.1368 | #117 | `>50` |
| **Q05_A3** | Q05 | A3 | 0 | 0.0% | 0.0% | 33.3% | 0.000 | 0.043 | 0.000 | 0.0% | 0.2220 | #57 | `>50` | 0.1651 | #122 | `>50` |
| **Q06_A0** | Q06 | A0 | 1 | 50.0% | 75.0% | 75.0% | 0.200 | 1.000 | 0.568 | 0.0% | 0.3135 | #116 | `>50` | 0.2448 | #145 | `>50` |
| **Q06_A1** | Q06 | A1 | 1 | 50.0% | 75.0% | 75.0% | 0.200 | 0.333 | 0.322 | 0.0% | 0.4245 | #53 | `>50` | 0.2475 | #146 | `>50` |
| **Q06_A2** | Q06 | A2 | 1 | 50.0% | 75.0% | 100.0% | 0.200 | 1.000 | 0.563 | 0.0% | 0.1666 | #108 | `>50` | 0.1000 | #156 | `>50` |
| **Q06_A3** | Q06 | A3 | 1 | 50.0% | 100.0% | 100.0% | 0.200 | 1.000 | 0.574 | 0.0% | 0.2459 | #59 | `>50` | 0.1460 | #145 | `>50` |

---

## 4. Detailed Diagnostic Target Autopsies

### Target C15 Autopsy ($Q_{03}$ — Primary Evidence)

| Condition | Query Description | Corpus Representation | Cosine Similarity | Global Rank | $\Delta\text{Sim}$ (vs A0) | $\Delta\text{Rank}$ (vs A0) | Accessibility Band | Candidate Reachable? |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **A0** | Original Question | $B_0$ Raw Text | 0.1986 | **#142** | +0.0000 | +0 | `>50` | No |
| **A1** | Original Question | $B_2$ Gloss + Raw | 0.2556 | **#118** | +0.0569 | +24 | `>50` | No |
| **A2** | Neutral Reformulation | $B_0$ Raw Text | 0.0922 | **#120** | -0.1064 | +22 | `>50` | No |
| **A3** | Neutral Reformulation | $B_2$ Gloss + Raw | 0.1974 | **#52** | -0.0012 | +90 | `>50` | No |

- **Strongest Intervention**: `A3` ($\Delta\text{Rank} = +90$, rank #52)
- **Target Diagnosis**: `REPRESENTATION_INTERVENTION_NOT_SUFFICIENT_FOR_ACCESSIBILITY`
- **Analysis**: Combining neutral query reformulation with corpus glossing ($A_3$) produced substantial rank movement from #142 to #52. However, because rank #52 remains outside the Top-50 candidate pool, the intervention was not sufficient to make $C_{15}$ candidate-accessible.

---

### Target C42 Autopsy ($Q_{05}$ — Primary Evidence)

| Condition | Query Description | Corpus Representation | Cosine Similarity | Global Rank | $\Delta\text{Sim}$ (vs A0) | $\Delta\text{Rank}$ (vs A0) | Accessibility Band | Candidate Reachable? |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **A0** | Original Question | $B_0$ Raw Text | 0.2811 | **#99** | +0.0000 | +0 | `>50` | No |
| **A1** | Original Question | $B_2$ Gloss + Raw | 0.2545 | **#101** | -0.0266 | -2 | `>50` | No |
| **A2** | Neutral Reformulation | $B_0$ Raw Text | 0.1368 | **#117** | -0.1443 | -18 | `>50` | No |
| **A3** | Neutral Reformulation | $B_2$ Gloss + Raw | 0.1651 | **#122** | -0.1160 | -23 | `>50` | No |

- **Strongest Intervention**: `none` (Baseline $A_0$ #99 was highest rank)
- **Target Diagnosis**: `REPRESENTATION_INTERVENTION_NOT_SUFFICIENT_FOR_ACCESSIBILITY`
- **Analysis**: Neither semantic glossing ($A_1$) nor query reformulation ($A_2/A_3$) improved the dense rank of $C_{42}$. Reformulations caused $C_{42}$ to regress from #99 to #117 and #122 due to vocabulary dilution.

---

## 5. Top-10 Hard Negative Autopsy ($A_0$ vs $A_2$)

### Question Q03 ($C_{15}$ Target): Top-10 Hard Negatives

| Rank | A0 Chunk ID | A0 Sim | A0 Gold? | A0 Mechanism | A2 Chunk ID | A2 Sim | A2 Gold? | A2 Mechanism |
| :---: | :---: | :---: | :---: | :--- | :---: | :---: | :---: | :--- |
| **#1** | C195 | 0.6909 | False | `entity_similarity` | C91 | 0.3843 | False | `entity_similarity` |
| **#2** | C83 | 0.6011 | False | `entity_similarity` | C47 | 0.3452 | False | `entity_similarity` |
| **#3** | C52 | 0.5860 | False | `entity_similarity` | C92 | 0.3206 | False | `entity_similarity` |
| **#4** | C24 | 0.5566 | **True** | `gold_target_match` | C83 | 0.3127 | False | `entity_similarity` |
| **#5** | C118 | 0.5553 | False | `entity_similarity` | C72 | 0.2886 | False | `entity_similarity` |
| **#6** | C88 | 0.5288 | False | `entity_similarity` | C195 | 0.2814 | False | `entity_similarity` |
| **#7** | C91 | 0.5287 | False | `entity_similarity` | C58 | 0.2786 | False | `generic_narrative_similarity` |
| **#8** | C47 | 0.5242 | False | `entity_similarity` | C45 | 0.2721 | False | `entity_similarity` |
| **#9** | C170 | 0.5223 | False | `entity_similarity` | C56 | 0.2713 | False | `entity_similarity` |
| **#10** | C177 | 0.5142 | False | `entity_similarity` | C52 | 0.2634 | False | `entity_similarity` |

### Question Q05 ($C_{42}$ Target): Top-10 Hard Negatives

| Rank | A0 Chunk ID | A0 Sim | A0 Gold? | A0 Mechanism | A2 Chunk ID | A2 Sim | A2 Gold? | A2 Mechanism |
| :---: | :---: | :---: | :---: | :--- | :---: | :---: | :---: | :--- |
| **#1** | C192 | 0.6133 | False | `entity_similarity` | C191 | 0.3368 | False | `generic_narrative_similarity` |
| **#2** | C95 | 0.6090 | False | `entity_similarity` | C193 | 0.3276 | False | `generic_narrative_similarity` |
| **#3** | C194 | 0.5795 | False | `entity_similarity` | C67 | 0.3079 | False | `entity_similarity` |
| **#4** | C50 | 0.5768 | False | `entity_similarity` | C4 | 0.3059 | False | `generic_narrative_similarity` |
| **#5** | C200 | 0.5562 | False | `generic_narrative_similarity` | C29 | 0.3046 | False | `entity_similarity` |
| **#6** | C70 | 0.5550 | False | `entity_similarity` | C24 | 0.3022 | False | `entity_similarity` |
| **#7** | C197 | 0.5537 | False | `entity_similarity` | C83 | 0.2995 | False | `entity_similarity` |
| **#8** | C77 | 0.5502 | False | `entity_similarity` | C186 | 0.2744 | False | `generic_narrative_similarity` |
| **#9** | C97 | 0.5463 | False | `entity_similarity` | C2 | 0.2717 | False | `generic_narrative_similarity` |
| **#10** | C47 | 0.5449 | False | `entity_similarity` | C171 | 0.2698 | False | `generic_narrative_similarity` |

---

## 6. Verification Assertions Audit (26 Gate Checks)

| Assertion # | Description | Status | Verification Note |
| :---: | :--- | :---: | :--- |
| **1** | Exactly 6 benchmark questions evaluated | `PASS` | Q01 through Q06 |
| **2** | Exactly 4 conditions evaluated per question | `PASS` | A0, A1, A2, A3 |
| **3** | Exactly 24 total retrieval evaluations | `PASS` | 6 × 4 matrix |
| **4** | Corpus size equals exactly 203 chunks | `PASS` | 203 chunks loaded |
| **5** | Corpus chunk boundaries remain unchanged | `PASS` | Zero re-segmentation |
| **6** | Embedding model equals `all-MiniLM-L6-v2` | `PASS` | E0 model frozen |
| **7** | A0 uses original benchmark questions | `PASS` | Verified exact strings |
| **8** | A1 changes corpus representation only ($B_2$) | `PASS` | Original Q + B2 Corpus |
| **9** | A2 changes query representation only ($A_2$) | `PASS` | Reformulated Q + B0 Corpus |
| **10** | A3 changes both query ($A_2$) and corpus ($B_2$) | `PASS` | Reformulated Q + B2 Corpus |
| **11** | Canonical Phase 10A gloss cache reused | `PASS` | Cache loaded from disk |
| **12** | Gloss cache MD5 equals `ee4243c2c0b575c87fe11a9f73f81de3` | `PASS` | UTF-8 hash verified |
| **13** | A2/A3 queries contain zero verbatim chunk text | `PASS` | Verified neutral text |
| **14** | A2/A3 queries contain zero chunk IDs | `PASS` | Zero C15/C42 tokens |
| **15** | A2/A3 queries contain zero target chunk metadata | `PASS` | Zero speaker/act metadata |
| **16** | A2/A3 queries generated without gold inspection | `PASS` | Derived strictly from Q text |
| **17** | Zero target-aware boosting/filtering in retrieval | `PASS` | Dense similarity search |
| **18** | All 203 chunks participate in every dense ranking | `PASS` | Global 203 ranking |
| **19** | Rankings determined exclusively by embedding sim | `PASS` | Dot product over normalized embs |
| **20** | Zero BM25, RRF, or Cross-Encoder in retrieval | `PASS` | Pure dense retrieval pipeline |
| **21** | Zero target-aware reranking performed | `PASS` | Post-hoc inspection only |
| **22** | Full ranking of all 203 chunks retained | `PASS` | Stored in JSON output |
| **23** | C15/C42 inspected strictly after retrieval | `PASS` | Post-hoc analysis layer |
| **24** | Observed ranks match stored dense ranks | `PASS` | 100% rank alignment |
| **25** | Top-10 hard negatives recorded for Q03/Q05 A0/A2 | `PASS` | Hard negative breakdown recorded |
| **26** | Exact query strings persisted in output JSON | `PASS` | Saved in results payload |

---

## 7. Diagnostic Synthesis & Decision for Phase 11B

### Empirical Findings:
1. **Target Inaccessibility**: Neither query reformulation ($A_2$) nor semantic glossing ($A_1/A_3$) succeeded in bringing $C_{15}$ or $C_{42}$ into candidate-reachable ranks ($\le 20$).
2. **Failure Mechanisms**:
   - For $C_{15}$, combined intervention $A_3$ demonstrated major rank movement ($\Delta\text{Rank} = +90$, from #142 to #52). However, rank #52 remains outside candidate accessibility.
   - For $C_{42}$, no tested intervention improved rank (baseline $A_0$ #99 regressed to #117 and #122).
3. **Primary Diagnostic Decision**:
   > **`NOT SUPPORTED`** — The hypothesis that C15/C42 candidate-generation failure is primarily explained by query–chunk representation mismatch is **not supported** by the tested representation interventions.

### Input to Phase 11B:
Since query and corpus representation interventions were **not sufficient for accessibility**, Phase 11B must investigate alternative failure mechanisms, such as:
- Chunk-level information density & context fragmentation
- Dialogue-to-prose representation limits
- Candidate generation pool construction mechanics
