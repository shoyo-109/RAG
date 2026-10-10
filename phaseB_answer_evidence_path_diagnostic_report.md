# Phase B — Answer–Evidence Path Diagnostic Final Report

> **Research Objective**: Determine why a gold-blind anchor candidate retrieval path can fail to expose required supporting evidence when that evidence is not locally accessible in embedding space.  
> **Status**: **COMPLETE & CANONICAL**  
> **Date**: 2026-10-06  
> **Control Reproduction Gate**: **PASS** (Candidate Recall = 85.0%, Final Recall@15 = 70.0%, nDCG@10 = 0.485, MRR = 0.369)  

---

## 1. Executive Summary

Phase B conducted a target-blind, post-hoc diagnostic investigation across 20 gold evidence units ($Q_{01} \dots Q_{06}$) over the 203-chunk *The Comedy of Errors* corpus. All gold-blind anchor candidates were identified post-hoc using a gold-blind heuristic without accessing any target evidence labels.

---

## 2. Canonical Baseline Control Reproduction

| Metric | Canonical Target | Phase B Control | Delta | Gate Status |
| :--- | :---: | :---: | :---: | :---: |
| **Candidate Recall@20** | **85.0%** | **85.0%** | `+0.0%` | `PASS` |
| **Final Recall@15** | **70.0%** | **70.0%** | `+0.0%` | `PASS` |
| **Macro nDCG@10** | **0.485** | **0.485** | `-0.000` | `PASS` |
| **MRR** | **0.369** | **0.369** | `-0.000` | `PASS` |

---

## 3. Counterfactual Diagnostic Route Evaluation

Evaluated across 4 gold-blind counterfactual routes originating from gold-blind anchor candidates:

| Diagnostic Route | Route Budget | Total Evidence Exposure | Inaccessible Candidate Recovery |
| :--- | :--- | :---: | :---: |
| **Route_A_Lexical** | `K=10 (BM25 top-5 distinctive anchor terms)` | **3 / 20 (15.0%)** | **0 (0.0%)** |
| **Route_B_Entity** | `fixed top-K diagnostic output budget (K=10)` | **8 / 20 (40.0%)** | **0 (0.0%)** |
| **Route_C_Structural** | `K=6 (Window k=3 around anchor candidate)` | **1 / 20 (5.0%)** | **0 (0.0%)** |
| **Route_D_Semantic** | `K=5 (Top-5 cosine vector neighbors)` | **6 / 20 (30.0%)** | **0 (0.0%)** |
| **Any_Combined_Route** | `Union of Routes A-D` | **9 / 20 (45.0%)** | **0 (0.0%)** |

---

## 4. Per-Question Diagnostic Breakdown

| Question | Req Units | Candidate Recall | Final Recall | Anchor Candidate ID | Ranking Failures | Path Failures | No-Path Failures |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Q01** | 2 | 100.0% | 100.0% | `C193` | 0 | 0 | 0 |
| **Q02** | 4 | 100.0% | 75.0% | `C4` | 1 | 0 | 0 |
| **Q03** | 4 | 50.0% | 50.0% | `C195` | 0 | 0 | 2 |
| **Q04** | 3 | 100.0% | 100.0% | `C193` | 0 | 0 | 0 |
| **Q05** | 3 | 66.7% | 33.3% | `C50` | 1 | 0 | 1 |
| **Q06** | 4 | 100.0% | 75.0% | `C113` | 1 | 0 | 0 |

---

## 5. Required Evidence Trace & Diagnostic Classification

| Question | Gold Unit | Role | In Top-15? | Anchor Candidate ID | Anchor Sim | Shared Entities | Distance | Failure Classification | Recovering Routes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **Q01** | `C4` | `PRIMARY` | `YES` | `C193` | 0.657 | `Aegeon, Epidamnum, Solinus` | 189 | `STAGE_3_FINAL_RECOVERY` | `Route_A_Lexical, Route_B_Entity, Route_D_Semantic` |
| **Q01** | `C2` | `SUPPORTING` | `YES` | `C193` | 0.532 | `Aegeon, Solinus` | 191 | `STAGE_3_FINAL_RECOVERY` | `Route_B_Entity` |
| **Q02** | `C2` | `PRIMARY` | `YES` | `C4` | 0.688 | `Aegeon, Ephesus, Solinus` | 2 | `STAGE_3_FINAL_RECOVERY` | `Route_B_Entity, Route_C_Structural, Route_D_Semantic` |
| **Q02** | `C4` | `SUPPORTING` | `YES` | `C4` | 1.000 | `Aegeon, Ephesus, Epidamnum, Solinus` | 0 | `STAGE_3_FINAL_RECOVERY` | `Route_A_Lexical` |
| **Q02** | `C10` | `SUPPORTING` | `YES` | `C4` | 0.725 | `Aegeon, Ephesus, Solinus` | 6 | `STAGE_3_FINAL_RECOVERY` | `Route_B_Entity, Route_D_Semantic` |
| **Q02** | `C12` | `SUPPORTING` | `NO` | `C4` | 0.493 | `Epidamnum` | 8 | `RANKING_FAILURE` | `None` |
| **Q03** | `C15` | `PRIMARY` | `NO` | `C195` | 0.438 | `Antipholus of Syracuse, Ephesus, Syracuse` | 180 | `NO_PATH_FAILURE` | `None` |
| **Q03** | `C24` | `PRIMARY` | `YES` | `C195` | 0.651 | `Adriana, Antipholus of Ephesus, Ephesus, Luciana` | 171 | `STAGE_3_FINAL_RECOVERY` | `Route_B_Entity` |
| **Q03** | `C29` | `SUPPORTING` | `NO` | `C195` | 0.591 | `Adriana, Ephesus, Luciana` | 166 | `NO_PATH_FAILURE` | `None` |
| **Q03** | `C61` | `SUPPORTING` | `YES` | `C195` | 0.670 | `Adriana, Luciana, Syracuse` | 134 | `STAGE_3_FINAL_RECOVERY` | `None` |
| **Q04** | `C4` | `PRIMARY` | `YES` | `C193` | 0.657 | `Aegeon, Epidamnum, Solinus` | 189 | `STAGE_3_FINAL_RECOVERY` | `Route_A_Lexical, Route_B_Entity, Route_D_Semantic` |
| **Q04** | `C6` | `SUPPORTING` | `YES` | `C193` | 0.522 | `Corinth` | 187 | `STAGE_3_FINAL_RECOVERY` | `None` |
| **Q04** | `C8` | `SUPPORTING` | `YES` | `C193` | 0.654 | `Aegeon, Corinth, Solinus` | 185 | `STAGE_3_FINAL_RECOVERY` | `Route_B_Entity, Route_D_Semantic` |
| **Q05** | `C42` | `PRIMARY` | `NO` | `C50` | 0.453 | `Antipholus of Syracuse, Dromio of Syracuse, Syracuse` | 8 | `NO_PATH_FAILURE` | `None` |
| **Q05** | `C65` | `PRIMARY` | `NO` | `C50` | 0.395 | `None` | 15 | `RANKING_FAILURE` | `None` |
| **Q05** | `C12` | `SUPPORTING` | `YES` | `C50` | 0.554 | `Antipholus of Syracuse, Dromio of Syracuse, Syracuse` | 38 | `STAGE_3_FINAL_RECOVERY` | `None` |
| **Q06** | `C105` | `PRIMARY` | `YES` | `C113` | 0.629 | `Angelo, Antipholus of Ephesus, Ephesus` | 8 | `STAGE_3_FINAL_RECOVERY` | `Route_B_Entity, Route_D_Semantic` |
| **Q06** | `C107` | `PRIMARY` | `YES` | `C113` | 0.432 | `Antipholus of Ephesus, Ephesus` | 6 | `STAGE_3_FINAL_RECOVERY` | `None` |
| **Q06** | `C147` | `SUPPORTING` | `YES` | `C113` | 0.563 | `Antipholus of Ephesus, Ephesus` | 34 | `STAGE_3_FINAL_RECOVERY` | `None` |
| **Q06** | `C155` | `SUPPORTING` | `NO` | `C113` | 0.422 | `None` | 42 | `RANKING_FAILURE` | `None` |

---

## 6. Target Autopsy: C15 ($Q_{03}$ — Poetic Soliloquy Metaphor)
- **Gold-Blind Anchor Candidate**: `C195` (Rank #1)
- **Anchor-to-Evidence Dense Similarity**: `0.4383`
- **Baseline Non-Gold Corpus Similarity**: `0.414 ± 0.2077`
- **Question-to-Evidence Dense Similarity**: `0.1986`
- **Lexical Jaccard Overlap**: `0.0638`
- **Shared Entities**: `Antipholus of Syracuse, Ephesus, Syracuse`
- **Corpus Chunk Distance**: `180` chunks
- **Recovering Diagnostic Routes**: `None`
- **Diagnostic Failure Category**: `NO_PATH_FAILURE`

---

## 7. Target Autopsy: C42 ($Q_{05}$ — Archaic Slang 'Sconce')
- **Gold-Blind Anchor Candidate**: `C50` (Rank #1)
- **Anchor-to-Evidence Dense Similarity**: `0.4528`
- **Baseline Non-Gold Corpus Similarity**: `0.3238 ± 0.1859`
- **Question-to-Evidence Dense Similarity**: `0.2811`
- **Lexical Jaccard Overlap**: `0.0476`
- **Shared Entities**: `Antipholus of Syracuse, Dromio of Syracuse, Syracuse`
- **Corpus Chunk Distance**: `8` chunks
- **Recovering Diagnostic Routes**: `None`
- **Diagnostic Failure Category**: `NO_PATH_FAILURE`

---

## 8. Measured Findings & Dynamic Conclusion

All narrative findings are derived dynamically from measured execution data.

- **Candidate vs Ranking vs Path Breakdown**:
  - `STAGE_3_FINAL_RECOVERY`: 14 / 20
  - `RANKING_FAILURE`: 3 / 20
  - `PATH_FAILURE`: 0 / 20
  - `NO_PATH_FAILURE`: 3 / 20

- **Primary Route Recovery Performance**: Diagnostic route exposure rates demonstrate which relationship signals (lexical, entity, structural, or semantic) are capable of exposing missing evidence units absent from the canonical candidate pool.
