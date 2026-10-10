# Phase 11A — Query–Chunk Representation Diagnosis Final Report

## Executive Summary

Phase 11A evaluated whether persistent retrieval bottlenecks for hard targets **C15** and **C42** are attributable to query–chunk representation mismatches. 

Following strict anti-leakage safeguards, dense retrieval was executed across the full frozen benchmark (**6 questions × 4 conditions = 24 runs**) using $E_0 =$ `sentence-transformers/all-MiniLM-L6-v2` over the canonical 203-chunk corpus. `C15` and `C42` were treated strictly as **post-hoc diagnostic observations**.

### Key Diagnostic Outcome

> **Primary Decision**: `NOT SUPPORTED`

- **Target C15** ($Q_{03}$, Primary): Diagnosis: `REPRESENTATION_INTERVENTION_NOT_SUFFICIENT_FOR_ACCESSIBILITY` (Strongest Intervention: `A3`)
- **Target C42** ($Q_{05}$, Primary): Diagnosis: `REPRESENTATION_INTERVENTION_NOT_SUFFICIENT_FOR_ACCESSIBILITY` (Strongest Intervention: `none`)

---

## 1. Experimental Controls & Setup

- **Corpus**: Canonical 203 chunks (`The Comedy of Errors document.pdf`)
- **Embedding Model**: $E_0 =$ `sentence-transformers/all-MiniLM-L6-v2` (L2 Normalized Cosine Sim via Dot Product)
- **Gloss Cache MD5**: `ee4243c2c0b575c87fe11a9f73f81de3`
- **Evaluations**: 24 runs (6 Questions × 4 Conditions: $A_0, A_1, A_2, A_3$)
- **Anti-Leakage Safeguard**: Zero target boosting, filtering, or chunk-text query injection.

---

## 2. Benchmark Aggregate Retrieval Metrics (Mean over Q01–Q06)

| Condition | Query | Corpus | Hit@10 | Rec@10 | Rec@20 | Rec@50 | Prec@10 | MRR | nDCG@10 | EvComp% | ReqComp% |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A0** | Original Question | B0 (Raw Text) | 83.3% | 36.1% | 54.2% | 77.8% | 0.117 | 0.356 | 0.279 | 0.0% | 33.3%
| **A1** | Original Question | B2 (Gloss + Raw) | 83.3% | 50.0% | 68.1% | 81.9% | 0.167 | 0.308 | 0.308 | 33.3% | 50.0%
| **A2** | Neutral Reformulation | B0 (Raw Text) | 66.7% | 50.0% | 72.2% | 86.1% | 0.167 | 0.349 | 0.328 | 50.0% | 50.0%
| **A3** | Neutral Reformulation | B2 (Gloss + Raw) | 83.3% | 56.9% | 70.8% | 76.4% | 0.183 | 0.399 | 0.356 | 33.3% | 33.3%

---

## 3. Diagnostic Target Autopsies (Post-Hoc Analysis)

### Target C15 ($Q_{{03}}$ — Primary Evidence)

| Condition | Query Type | Corpus Rep | Cosine Similarity | Global Rank | ΔSim (vs A0) | ΔRank (vs A0) | Accessibility Band | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **A0** | Original | B0 Raw | 0.1986 | #142 | +0.0000 | +0 | `>50` | Inaccessible |
| **A1** | Original | B2 Gloss | 0.2556 | #118 | +0.0569 | +24 | `>50` | Inaccessible |
| **A2** | A2 Reformulation | B0 Raw | 0.0922 | #120 | -0.1064 | +22 | `>50` | Inaccessible |
| **A3** | A2 Reformulation | B2 Gloss | 0.1974 | #52 | -0.0012 | +90 | `>50` | Inaccessible |

- **Strongest Intervention**: `A3`
- **Target Diagnosis**: `REPRESENTATION_INTERVENTION_NOT_SUFFICIENT_FOR_ACCESSIBILITY`

### Target C42 ($Q_{05}$ — Primary Evidence)

| Condition | Query Type | Corpus Rep | Cosine Similarity | Global Rank | ΔSim (vs A0) | ΔRank (vs A0) | Accessibility Band | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **A0** | Original | B0 Raw | 0.2811 | #99 | +0.0000 | +0 | `>50` | Inaccessible |
| **A1** | Original | B2 Gloss | 0.2545 | #101 | -0.0266 | -2 | `>50` | Inaccessible |
| **A2** | A2 Reformulation | B0 Raw | 0.1368 | #117 | -0.1443 | -18 | `>50` | Inaccessible |
| **A3** | A2 Reformulation | B2 Gloss | 0.1651 | #122 | -0.1160 | -23 | `>50` | Inaccessible |

- **Strongest Intervention**: `none`
- **Target Diagnosis**: `REPRESENTATION_INTERVENTION_NOT_SUFFICIENT_FOR_ACCESSIBILITY`

---

## 4. Top-10 Hard Negative Breakdown ($A_0$ vs $A_2$)

### $Q_{03}$ Top Hard Negatives (A0 vs A2)

| Rank | A0 Chunk | A0 Sim | A0 Gold? | A0 Mechanism | A2 Chunk | A2 Sim | A2 Gold? | A2 Mechanism |
| :---: | :---: | :---: | :---: | :--- | :---: | :---: | :---: | :--- |
| #1 | C195 | 0.6909 | False | entity_similarity | C91 | 0.3843 | False | entity_similarity |
| #2 | C83 | 0.6011 | False | entity_similarity | C47 | 0.3452 | False | entity_similarity |
| #3 | C52 | 0.5860 | False | entity_similarity | C92 | 0.3206 | False | entity_similarity |
| #4 | C24 | 0.5566 | True | gold_target_match | C83 | 0.3127 | False | entity_similarity |
| #5 | C118 | 0.5553 | False | entity_similarity | C72 | 0.2886 | False | entity_similarity |
| #6 | C88 | 0.5288 | False | entity_similarity | C195 | 0.2814 | False | entity_similarity |
| #7 | C91 | 0.5287 | False | entity_similarity | C58 | 0.2786 | False | generic_narrative_similarity |
| #8 | C47 | 0.5242 | False | entity_similarity | C45 | 0.2721 | False | entity_similarity |
| #9 | C170 | 0.5223 | False | entity_similarity | C56 | 0.2713 | False | entity_similarity |
| #10 | C177 | 0.5142 | False | entity_similarity | C52 | 0.2634 | False | entity_similarity |

### $Q_{{05}}$ Top Hard Negatives (A0 vs A2)

| Rank | A0 Chunk | A0 Sim | A0 Gold? | A0 Mechanism | A2 Chunk | A2 Sim | A2 Gold? | A2 Mechanism |
| :---: | :---: | :---: | :---: | :--- | :---: | :---: | :---: | :--- |
| #1 | C192 | 0.6133 | False | entity_similarity | C191 | 0.3368 | False | generic_narrative_similarity |
| #2 | C95 | 0.6090 | False | entity_similarity | C193 | 0.3276 | False | generic_narrative_similarity |
| #3 | C194 | 0.5795 | False | entity_similarity | C67 | 0.3079 | False | entity_similarity |
| #4 | C50 | 0.5768 | False | entity_similarity | C4 | 0.3059 | False | generic_narrative_similarity |
| #5 | C200 | 0.5562 | False | generic_narrative_similarity | C29 | 0.3046 | False | entity_similarity |
| #6 | C70 | 0.5550 | False | entity_similarity | C24 | 0.3022 | False | entity_similarity |
| #7 | C197 | 0.5537 | False | entity_similarity | C83 | 0.2995 | False | entity_similarity |
| #8 | C77 | 0.5502 | False | entity_similarity | C186 | 0.2744 | False | generic_narrative_similarity |
| #9 | C97 | 0.5463 | False | entity_similarity | C2 | 0.2717 | False | generic_narrative_similarity |
| #10 | C47 | 0.5449 | False | entity_similarity | C171 | 0.2698 | False | generic_narrative_similarity |

---

## 5. Phase 11A Decision & Next Experiment

> **Diagnostic Question**: Is the candidate-generation failure of hard targets C15 and C42 primarily explained by query–chunk representation mismatch?
> 
> **Decision**: `NOT SUPPORTED`

### Justification & Key Empirical Observations:
1. **Target Movement Analysis**:
   - `C15`: A0 Rank = `#142`, Best Rank = `#52`
   - `C42`: A0 Rank = `#99`, Best Rank = `#99`
2. **Representation Accessibility**:
   - Tested interventions (glosses, neutral semantic reformulations) demonstrate empirical impact on rank movements.
   - If ranks remain $>50$, the outcome states strictly: `REPRESENTATION_INTERVENTION_NOT_SUFFICIENT_FOR_ACCESSIBILITY`.

---

## 6. Verification Assertions Audit

- [x] All 26 programmatic validation assertions passed cleanly.
- [x] Gloss Cache MD5 matches canonical hash (`ee4243c2c0b575c87fe11a9f73f81de3`).
- [x] Exactly 24 runs executed with zero data leakage.
