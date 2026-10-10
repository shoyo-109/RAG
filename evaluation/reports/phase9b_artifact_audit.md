# Phase 9B — Artifact Integrity Audit & Reconciliation Report

**Corpus:** *The Comedy of Errors* (203 semantic chunks)  
**Audit Date:** September 27, 2026  
**Status:** **PASSED & FROZEN** — All numerical discrepancies reconciled into canonical JSON.

---

## 1. Phase 9B Artifact Inventory

| Artifact File | Artifact Type | Generation Source | Timestamp | Represented Experiment | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `evaluation/results/phase9b_representation_ablation.json` | Machine-Readable JSON | `run_phase9b_representation_ablation.py` | 2026-09-26T20:35:19Z | Raw empirical measurements for R0–R4 | **Source of Truth** |
| `evaluation/results/r4_semantic_glosses_cache.json` | JSON Cache | Nemotron LLM batch generator | 2026-09-26T20:31:09Z | Neutral 1-2 sentence R4 glosses | **Verified Cache** |
| `evaluation/reports/phase9b_representation_ablation.md` | Markdown Report | Generated from script run | 2026-09-26T20:35:32Z | Human-readable Phase 9B summary | Audited & Superseded |
| `Phase_9B_Retrieval_Representation_Ablation_Report.md` | Root Markdown Report | Copy of `phase9b_representation_ablation.md` | 2026-09-26T20:35:43Z | Root copy for user inspection | Audited & Superseded |
| `evaluation/results/phase9b_canonical_results.json` | Canonical Source of Truth | Reconciled from raw JSON | 2026-09-27T04:38:39Z | Reconciled canonical benchmark dataset | **FROZEN CANONICAL JSON** |
| `evaluation/reports/phase9b_representation_ablation_canonical.md` | Canonical Markdown Report | Generated from canonical JSON | 2026-09-27T04:38:50Z | Programmatically generated report | **CANONICAL REPORT** |

---

## 2. Canonical Experiment Identity

* **Corpus:** *The Comedy of Errors*, 203 semantic chunks (`C0` to `C202`).
* **Retrieval Architecture:**
  - Q2 Multi-Query Expansion (exact frozen sub-queries from Phase 7A/7B/8)
  - Candidate Union Depth $N=20$
  - Cross-Encoder Reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
  - Context Budget: Top-$K = 10$ chunks
* **Dense Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors)
* **Target Failed Evidence Chunks:** `C15`, `C29`, `C42`, `C61`, `C147`
* **Representation Conditions:**
  - `R0` = Raw original chunk text (Control Baseline)
  - `R1` = Local Context-Enriched (`[PREVIOUS]` + `[TARGET]` + `[NEXT]`)
  - `R2` = Metadata-Enriched (`Act`, `Scene`, `Speakers`, `Document Type`, `Entities`)
  - `R3` = Context + Metadata (R2 header + R1 text)
  - `R4` = Neutral LLM Semantic Gloss + Original Text (Diagnostic Probe)

---

## 3. Discrepancy Audit & Reconciliation Table

| Metric / Target | Initial Markdown Report Value | Canonical Raw JSON Value (`phase9b_canonical_results.json`) | Discrepancy Cause | Reconciliation Status |
| :--- | :---: | :---: | :--- | :--- |
| **`C15` R0 Dense Similarity** | `0.3124` | **`0.1986`** | `DERIVED-METRIC BUG` (Prose text used un-normalized dot product estimate instead of exact unit-normalized cosine similarity). | **Reconciled to `0.1986`** |
| **`C61` R0 CE Rank** | `#14` | **`#15`** | `FORMATTING / DERIVED-METRIC BUG` (0-index vs 1-index boundary in early script printer). | **Reconciled to `#15`** |
| **`C147` R0 Pipeline Top-10 Status** | `NO` (in prose summary) | **`YES`** (in full pipeline) | `REPORTING BUG` (Early text table compared raw Q2 union without BM25; full pipeline R0 includes Q03 subquery #5 rank, surfacing C147 to CE rank #3). | **Reconciled to `YES`** |
| **R0 Pipeline MRR** | `0.417` | **`0.368`** | `DERIVED-METRIC BUG` (Phase 8 reference used macro average; Phase 9B raw evaluation computes exact reciprocal rank across all 6 cases). | **Reconciled to `0.368`** |
| **R0 Pipeline nDCG@10** | `0.485` | **`0.485`** | `ROUNDING` (Exact value `0.48499621...`). | **Reconciled to `0.485`** |
| **R4 Pipeline Final Rec@10** | `65.0%` | **`65.0%`** | `EXACT MATCH` | **Verified (`65.0%`)** |
| **R4 Pipeline nDCG@10** | `0.604` | **`0.604`** | `EXACT MATCH` | **Verified (`0.604`)** |

---

## 4. Phase 9B Integrity Gate Status

* [x] All numerical discrepancies between raw output, JSON, and Markdown identified and reconciled.
* [x] Canonical JSON created at `evaluation/results/phase9b_canonical_results.json`.
* [x] Canonical Markdown report generated at `evaluation/reports/phase9b_representation_ablation_canonical.md`.
* [x] Gold evidence annotations confirmed 100% unchanged.
* [x] Experiment configuration frozen.
* [x] Target-level values and aggregate metrics strictly aligned.
* [x] Phase 9B Integrity Gate **PASSED**. Proceeding to Phase 9D.
