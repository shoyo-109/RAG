# Phase B — Answer–Evidence Path Diagnostic Experimental Plan (Final Audited & Corrected)

## Status
- **Phase**: Phase B (Answer–Evidence Path Diagnostic)
- **Preceding Phases**: Phase 10C (Canonical Baseline Control), Phase 11C (Candidate-Depth Diagnostic), Phase A (Semantic Neighborhood Diagnostic)
- **Follow-up Constraint**: Do **NOT** automatically launch Phase C or Phase 12 after completion.

---

## 1. Research Question & Objectives

The central research question is:

> **Why can a gold-blind anchor candidate retrieval path fail to expose the evidence needed to justify that answer when the required evidence is not locally accessible in embedding space?**

More specifically, this phase investigates:
1. Does the retrieved gold-blind anchor candidate contain enough semantic/lexical information to establish a retrieval path toward the required evidence?
2. Are gold-blind anchor candidates and required evidence chunks connected through identifiable corpus relationships (lexical, semantic, entity, reference, structural) that the current retriever does not exploit?
3. What is the root cause of retrieval failure across gold evidence units: `STAGE_3_FINAL_RECOVERY`, `RANKING_FAILURE`, `PATH_FAILURE`, or `NO_PATH_FAILURE`?
4. Can these relationships be detected **post-hoc** without injecting gold evidence into retrieval?

---

## 2. Experimental Constraints & Gold-Blind Rule

1. **Zero Target-Aware Injections**: No gold chunk IDs, no gold evidence text, no target-aware queries, no HyDE, no LLM query expansion are introduced into the retrieval pipeline.
2. **Post-Hoc Gold-Blind Anchor Candidate Selection**: Anchor candidates are identified **only after** canonical retrieval completes using a deterministic composite answerability heuristic ($S_{\text{anchor}} = 0.4 \cdot S_{\text{ce}} + 0.3 \cdot S_{\text{dense}} + 0.2 \cdot S_{\text{lex}} + 0.1 \cdot S_{\text{rank}}$) over the canonical Top-15. *Zero access to `gold_map` or gold IDs.*
3. **Gold-Blind Diagnostic Routes**: Counterfactual diagnostic routes are constructed exclusively using information present in the anchor candidate itself and the corpus metadata/text. Gold annotations are only used **afterward** to evaluate route recovery rate.
4. **Frozen Production Control**: Production pipeline (embedding model `all-MiniLM-L6-v2`, BM25 retriever, RRF, Cross-Encoder `ms-marco-MiniLM-L-6-v2`, Top-K=15) remains 100% frozen.

---

## 3. Canonical Control Reproduction & 4-Metric Gating

The script will reproduce the canonical Phase 11C / 10C control over the 203-chunk *The Comedy of Errors* corpus across benchmark questions Q01–Q06 (20 gold evidence units) and gate on:

| Control Metric | Canonical Baseline Target | Gate Tolerance |
| :--- | :---: | :---: |
| **Candidate Recall@20** | **85.0%** | $\pm 3.0\%$ (82.0% – 88.0%) |
| **Final Recall@15** | **70.0%** | $\pm 3.0\%$ (67.0% – 73.0%) |
| **Macro nDCG@10** | **0.485** | $\pm 0.030$ (0.455 – 0.515) |
| **MRR** | **0.369** | $\pm 0.030$ (0.339 – 0.399) |

*If any control metric falls outside tolerance, execution immediately terminates. In generated reports, control metric deltas and PASS/FAIL statuses are computed dynamically against canonical targets.*

---

## 4. Observability & Relationship Signals

For every pair `(gold-blind anchor candidate → required gold evidence unit)`, 6 independent relationship signals will be measured:
1. **Lexical Overlap**: Jaccard similarity, shared content words, shared entity names.
2. **Dense Semantic Similarity**: Cosine similarity ($\text{Anchor Candidate} \rightarrow \text{Evidence}$, $\text{Question} \rightarrow \text{Anchor Candidate}$, $\text{Question} \rightarrow \text{Evidence}$) and comparison against non-gold baseline corpus mean/std.
3. **Entity / Character Continuity**: Shared named entities from deterministic `KNOWN_ENTITIES` list.
4. **Surface Pronoun Overlap**: Co-occurrence of discourse pronouns (`he`, `she`, `they`, `this`, `that`).
5. **Structural / Narrative Proximity**: Chunk index distance ($\Delta C$), same act/scene section metadata.
6. **BM25 Relationship**: Anchor candidate-derived lexical retrieval exposure.

---

## 5. Counterfactual Diagnostic Routes

From each retrieved gold-blind anchor candidate, evaluate 4 counterfactual routes:
- **Route A (Lexical Anchor Candidate Route)**: Retrieve using top-5 distinctive anchor terms (TF-IDF weighted, deterministic sort) via BM25 (Budget $K=10$).
- **Route B (Entity Route)**: Ranked entity overlap score ($S_{\text{ent}}$) across 203 corpus chunks (fixed top-K diagnostic output budget $K=10$).
- **Route C (Structural Proximity Route)**: Neighboring chunks within window $k \le 3$ (Budget $K=6$).
- **Route D (Semantic Vector Route)**: Top-5 nearest vector neighbors to anchor candidate embedding (Budget $K=5$).

---

## 6. Precedence Order for Failure Classification

Each of the 20 required gold evidence units will be classified with strict precedence:
1. `STAGE_3_FINAL_RECOVERY`: In canonical Final Top-15.
2. `RANKING_FAILURE`: In Candidate Pool@20, but demoted during Cross-Encoder reranking.
3. `PATH_FAILURE`: Absent from Candidate Pool@20, but exposed by at least one diagnostic route.
4. `NO_PATH_FAILURE`: Absent from Candidate Pool@20 and unexposed by all tested diagnostic routes.

---

## 7. Target Autopsies (C15 & C42)

Perform post-hoc diagnostic autopsies for established failure chunks:
- **C15** ($Q_{03}$ — Poetic Soliloquy Metaphor): Trace anchor candidate distance, semantic similarity, and route accessibility.
- **C42** ($Q_{05}$ — Archaic Slang 'Sconce'): Trace entity overlap, lexical similarity, and route accessibility.

---

## 8. Section 21 Pre-Execution Validation Checklist

```text
[PASS] No gold IDs used in anchor candidate selection
[PASS] No gold text used in anchor candidate selection
[PASS] No C15/C42 target logic in retrieval
[PASS] Route A bounded (K=10)
[PASS] Route B output bounded (K=10)
[PASS] Route C bounded (K=6)
[PASS] Route D bounded (K=5)
[PASS] No rank-derived fake BM25 scores
[PASS] All route scores/ranks recorded
[PASS] Four-metric control gate active
[PASS] Per-question metrics implemented
[PASS] Dynamic conclusions & control report deltas implemented
```

---

## 9. Output Artifacts

- `phaseB_results.json`
- `phaseB_answer_evidence_trace.csv`
- `phaseB_route_results.csv`
- `phaseB_question_metrics.csv`
- `phaseB_answer_evidence_path_diagnostic_report.md`
