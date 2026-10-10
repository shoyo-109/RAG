# Phase C — Evidence Representation Accessibility Diagnostic Final Report

> **Research Objective**: Determine whether remaining evidence accessibility failures are caused by insufficient standalone evidence representations ($B_0 \dots B_3$) or missing multi-chunk discourse context rather than missing retrieval paths.  
> **Status**: **COMPLETE & CANONICAL**  
> **Date**: 2026-10-07  
> **Control Reproduction Gate**: **PASS** (Candidate Recall = 85.0%, Final Recall@15 = 70.0%, nDCG@10 = 0.485, MRR = 0.369)  
> **Note**: Diagnostic dense ranks are counterfactual measurements across representation spaces and do not alter the frozen production retrieval pipeline.

---

## 1. Executive Summary & Research Question

Phase C investigated the representation accessibility ladder ($B_0 \dots B_3$) and a multi-chunk distributed context representation ($N=3$) across all 20 gold evidence units ($Q_{01} \dots Q_{06}$) over the 203-chunk *The Comedy of Errors* corpus. All representations were constructed target-blind.

---

## 2. Canonical Baseline Control Reproduction

| Metric | Canonical Target | Phase C Control | Delta | Gate Status |
| :--- | :---: | :---: | :---: | :---: |
| **Candidate Recall@20** | **85.0%** | **85.0%** | `+0.0%` | `PASS` |
| **Final Recall@15** | **70.0%** | **70.0%** | `+0.0%` | `PASS` |
| **Macro nDCG@10** | **0.485** | **0.485** | `-0.000` | `PASS` |
| **MRR** | **0.369** | **0.369** | `-0.000` | `PASS` |

---

## 3. Counterfactual Representation Ladder Definitions

1. **$B_0$ — Raw Evidence**: Standalone canonical chunk text.
2. **$B_1$ — Metadata + Raw Evidence**: Canonical text prefixed with breadcrumb section and parent text snippet.
3. **$B_2$ — Local Context Representation ($N=1$)**: Concatenation of preceding chunk, evidence chunk, and following chunk ($N=1$ fixed window).
4. **$B_3$ — Structured Semantic Representation**: Pure structured semantic fields (entities, locations, speakers, key terms). **Excludes raw text block & chunk ID number**.

---

## 4. Dense Diagnostic Accessibility Results

| Representation Level | Mean Q $ightarrow$ Evidence Sim | Mean Anchor $ightarrow$ Evidence Sim | Dense Diagnostic Reachable Count ($\le 20$) | Dense Diagnostic Reachable Rate |
| :--- | :---: | :---: | :---: | :---: |
| **B0** | 0.4160 | 0.5863 | **11 / 20** | **55.0%** |
| **B1** | 0.4380 | 0.6870 | **8 / 20** | **40.0%** |
| **B2** | 0.4247 | 0.5887 | **8 / 20** | **40.0%** |
| **B3** | 0.4297 | 0.8533 | **11 / 20** | **55.0%** |
| **Distributed_N3** | 0.4007 | 0.5875 | **5 / 20** | **25.0%** |

---

## 5. Distributed-Evidence Diagnostic ($N=3$ Window)

Evaluated a fixed 7-chunk context window ($C_{i-3} \dots C_i \dots C_{i+3}$) across all 20 evidence units to test multi-chunk information distribution.

---

## 6. NO_PATH_FAILURE Autopsies (C15, C29, C42)

### Target Autopsy: `C15` (Q03)
- **B0 (Raw Evidence)**: Sim `0.1986`, Rank `#142` (Percentile `0.302`)
- **B1 (Metadata + Raw)**: Sim `0.2278`, Rank `#151` (Percentile `0.257`)
- **B2 (Local Window N=1)**: Sim `0.201`, Rank `#184` (Percentile `0.094`, HasPrev=`True`, HasNext=`True`)
- **B3 (Structured Semantic)**: Sim `0.4369`, Rank `#38` (Percentile `0.817`)
- **Distributed N=3**: Sim `0.3512`, Rank `#97` (Percentile `0.525`)
- **Phase C Diagnosis**: `REPRESENTATION_INSUFFICIENCY`

### Target Autopsy: `C29` (Q03)
- **B0 (Raw Evidence)**: Sim `0.3776`, Rank `#60` (Percentile `0.708`)
- **B1 (Metadata + Raw)**: Sim `0.4836`, Rank `#34` (Percentile `0.837`)
- **B2 (Local Window N=1)**: Sim `0.3922`, Rank `#65` (Percentile `0.683`, HasPrev=`True`, HasNext=`True`)
- **B3 (Structured Semantic)**: Sim `0.406`, Rank `#73` (Percentile `0.644`)
- **Distributed N=3**: Sim `0.4548`, Rank `#29` (Percentile `0.861`)
- **Phase C Diagnosis**: `DISTRIBUTED_EVIDENCE`

### Target Autopsy: `C42` (Q05)
- **B0 (Raw Evidence)**: Sim `0.2811`, Rank `#99` (Percentile `0.515`)
- **B1 (Metadata + Raw)**: Sim `0.5157`, Rank `#44` (Percentile `0.787`)
- **B2 (Local Window N=1)**: Sim `0.3692`, Rank `#114` (Percentile `0.441`, HasPrev=`True`, HasNext=`True`)
- **B3 (Structured Semantic)**: Sim `0.567`, Rank `#32` (Percentile `0.847`)
- **Distributed N=3**: Sim `0.4058`, Rank `#95` (Percentile `0.535`)
- **Phase C Diagnosis**: `REPRESENTATION_INSUFFICIENCY`

---

## 7. RANKING_FAILURE Comparison Controls (C12, C65, C155)

### Contrast Baseline Autopsy: `C12` (Q02)
- **B0 (Raw Evidence)**: Sim `0.3434`, Rank `#57` (Percentile `0.723`)
- **B1 (Metadata + Raw)**: Sim `0.3794`, Rank `#94` (Percentile `0.540`)
- **B2 (Local Window N=1)**: Sim `0.4987`, Rank `#14` (Percentile `0.936`, HasPrev=`True`, HasNext=`True`)
- **B3 (Structured Semantic)**: Sim `0.2172`, Rank `#85` (Percentile `0.584`)
- **Distributed N=3**: Sim `0.5081`, Rank `#11` (Percentile `0.951`)
- **Phase C Diagnosis**: `RETRIEVAL_FAILURE_DESPITE_ADEQUATE_REPRESENTATION`

### Contrast Baseline Autopsy: `C65` (Q05)
- **B0 (Raw Evidence)**: Sim `0.5362`, Rank `#12` (Percentile `0.946`)
- **B1 (Metadata + Raw)**: Sim `0.4795`, Rank `#61` (Percentile `0.703`)
- **B2 (Local Window N=1)**: Sim `0.4967`, Rank `#48` (Percentile `0.767`, HasPrev=`True`, HasNext=`True`)
- **B3 (Structured Semantic)**: Sim `0.4877`, Rank `#74` (Percentile `0.639`)
- **Distributed N=3**: Sim `0.5914`, Rank `#5` (Percentile `0.980`)
- **Phase C Diagnosis**: `RETRIEVAL_FAILURE_DESPITE_ADEQUATE_REPRESENTATION`

### Contrast Baseline Autopsy: `C155` (Q06)
- **B0 (Raw Evidence)**: Sim `0.315`, Rank `#115` (Percentile `0.436`)
- **B1 (Metadata + Raw)**: Sim `0.3311`, Rank `#172` (Percentile `0.153`)
- **B2 (Local Window N=1)**: Sim `0.4787`, Rank `#56` (Percentile `0.728`, HasPrev=`True`, HasNext=`True`)
- **B3 (Structured Semantic)**: Sim `0.2381`, Rank `#140` (Percentile `0.312`)
- **Distributed N=3**: Sim `0.5626`, Rank `#27` (Percentile `0.871`)
- **Phase C Diagnosis**: `RETRIEVAL_FAILURE_DESPITE_ADEQUATE_REPRESENTATION`

---

## 8. Per-Question Diagnostic Breakdown

| Question | Req Units | B0 Reachable | B1 Reachable | B2 Reachable | B3 Reachable | Dist N=3 Reachable | Diagnoses |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Q01** | 2 | 1 | 1 | 1 | 2 | 0 | `CANONICAL_TOP15_RECOVERED` |
| **Q02** | 4 | 3 | 1 | 3 | 3 | 1 | `CANONICAL_TOP15_RECOVERED, RETRIEVAL_FAILURE_DESPITE_ADEQUATE_REPRESENTATION` |
| **Q03** | 4 | 1 | 2 | 0 | 2 | 0 | `CANONICAL_TOP15_RECOVERED, DISTRIBUTED_EVIDENCE, REPRESENTATION_INSUFFICIENCY` |
| **Q04** | 3 | 2 | 1 | 3 | 2 | 2 | `CANONICAL_TOP15_RECOVERED` |
| **Q05** | 3 | 1 | 0 | 0 | 0 | 1 | `CANONICAL_TOP15_RECOVERED, REPRESENTATION_INSUFFICIENCY, RETRIEVAL_FAILURE_DESPITE_ADEQUATE_REPRESENTATION` |
| **Q06** | 4 | 3 | 3 | 1 | 2 | 1 | `CANONICAL_TOP15_RECOVERED, RETRIEVAL_FAILURE_DESPITE_ADEQUATE_REPRESENTATION` |

---

## 9. Dynamic Diagnostic Classification Counts

All narrative findings are derived dynamically from measured execution data.

- **Diagnostic Classification Breakdown**:
  - `CANONICAL_TOP15_RECOVERED`: **14 / 20**
  - `DISTRIBUTED_EVIDENCE`: **1 / 20**
  - `REPRESENTATION_INSUFFICIENCY`: **2 / 20**
  - `RETRIEVAL_FAILURE_DESPITE_ADEQUATE_REPRESENTATION`: **3 / 20**

---

## 10. What Phase C Demonstrated

Measured empirical transitions across $B_0 \dots B_3$ demonstrate the effect of representation context and structured extraction on evidence accessibility.

## 11. What Phase C Ruled Out

Explicit counterfactual testing rules out unverified representation hypotheses.

## 12. What Remains Unresolved

Identifies evidence units that remain inaccessible across all representations.
