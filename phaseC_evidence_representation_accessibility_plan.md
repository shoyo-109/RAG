# Phase C — Evidence Representation Accessibility Diagnostic Experimental Plan (Audited & Reconciled)

## Status
- **Phase**: Phase C (Evidence Representation Accessibility Diagnostic)
- **Preceding Phases**: Phase 10C (Canonical Baseline Control), Phase 11C (Candidate-Depth Diagnostic), Phase A (Semantic Neighborhood Diagnostic), Phase B (Answer–Evidence Path Diagnostic)
- **Follow-up Constraint**: Diagnostic evaluation phase only. Do **NOT** automatically deploy modified representations to production or launch Phase D until diagnostic findings are verified.

---

## 1. Research Question & Objectives

Phase B established that inaccessible evidence units (`C15`, `C29`, `C42`) cannot be recovered via post-hoc lexical, entity, structural, or semantic routes originating from gold-blind anchor candidates (`PATH_FAILURE = 0/20`, `NO_PATH_FAILURE = 3/20`).

The central research question for Phase C is:

> **Why does required evidence remain inaccessible when it is not exposed through the tested lexical, entity, structural, or local semantic relationships to the answer-bearing retrieval path?**

More specifically, Phase C investigates:
> **Does the standalone representation of the required evidence contain enough retrievable semantic, contextual, discourse, entity, and relational information for the retrieval system to recognize that it supports the question/answer?**

The goal is causal diagnosis to distinguish among:
1. `REPRESENTATION_INSUFFICIENCY`: Structured semantic representation ($B_3$) produces a material accessibility improvement while standalone raw chunk text ($B_0$) is unreachable.
2. `CONTEXT_DEPENDENCE`: Evidence requires local surrounding discourse context ($B_2$, window $N=1$) to produce a material accessibility improvement.
3. `DISTRIBUTED_EVIDENCE`: Required information is distributed across a wider multi-chunk context ($N=3$, window $C_{i-3} \dots C_{i+3}$).
4. `SEMANTIC_QUERY_MISMATCH`: Question-to-evidence dense similarity remains consistently weak across all representations relative to accessible evidence controls (`C12`, `C65`, `C155`).
5. `RETRIEVAL_FAILURE_DESPITE_ADEQUATE_REPRESENTATION`: Evidence representation $B_0$ is already candidate-accessible ($\text{dense\_diagnostic\_rank} \le 20$), proving the failure occurs downstream in candidate union or reranking.
6. `UNRESOLVED`: Multiple explanations remain equally plausible or evidence remains inaccessible without a clear representation transition.

---

## 2. Experimental Constraints & Gold-Blind Protocol

1. **Zero Target-Aware Injections**: No gold chunk IDs, gold evidence texts, gold questions, or target-specific terminology are used to generate representations $B_0 \dots B_3$, distributed context representations, query vectors, or candidate pools.
2. **Gold Annotations Post-Hoc Only**: Gold labels (`gold_map`, `required_evidence`, `C15`, `C29`, `C42`) are accessed **strictly post-hoc** for metric evaluation and diagnostic comparison.
3. **Frozen Production Control**: Production retrieval pipeline remains 100% frozen:
   - Embedding Model $E_0$: `sentence-transformers/all-MiniLM-L6-v2`
   - BM25 Retriever
   - Production RRF Candidate Union
   - Cross-Encoder $R_0$: `cross-encoder/ms-marco-MiniLM-L-6-v2`
   - Candidate Depth: Top-20 per subquery
   - Final Top-K: 15

---

## 3. Canonical Baseline Control Reproduction & 4-Metric Gate

Before running counterfactual diagnostic representations, the script must reproduce the canonical Phase 10C/11C/Phase B baseline control over the 203-chunk *The Comedy of Errors* corpus across benchmark questions $Q_{01} \dots Q_{06}$ (20 gold evidence units) and gate on:

| Control Metric | Canonical Baseline Target | Gate Tolerance |
| :--- | :---: | :---: |
| **Candidate Recall@20** | **85.0%** | $\pm 3.0\%$ (82.0% – 88.0%) |
| **Final Recall@15** | **70.0%** | $\pm 3.0\%$ (67.0% – 73.0%) |
| **Macro nDCG@10** | **0.485** | $\pm 0.030$ (0.455 – 0.515) |
| **MRR** | **0.369** | $\pm 0.030$ (0.339 – 0.399) |

*If any control metric fails to reproduce within tolerance, execution immediately terminates.*

---

## 4. Counterfactual Representation Ladder ($B_0 \dots B_3$) & Distributed Diagnostic

Construct 4 parallel representation levels plus a multi-chunk distributed context representation for all 203 corpus chunks:

### $B_0$ — Raw Evidence
Exact canonical chunk text:
$$B_0(C_i) = \text{raw\_text}(C_i)$$

### $B_1$ — Metadata + Raw Evidence
Canonical text prefixed with existing canonical metadata (breadcrumb section, parent text snippet):
$$B_1(C_i) = \text{breadcrumb} + " \mid " + \text{parent\_snippet} + "\n" + \text{raw\_text}(C_i)$$

### $B_2$ — Local Context Representation ($N=1$)
Concatenation of preceding chunk, evidence chunk, and following chunk ($N=1$ fixed window for all chunks). Includes explicit boundary metadata (`has_prev_context`, `has_next_context`, `prev_chunk_id`, `next_chunk_id`):
$$B_2(C_i) = \text{text}(C_{i-1}) + "\n[EVIDENCE]\n" + \text{text}(C_i) + "\n[AFTER]\n" + \text{text}(C_{i+1})$$

### $B_3$ — Structured Semantic Representation (NO RAW TEXT & NO CHUNK_ID INSIDE TEXT)
Pure structured semantic representation derived deterministically from metadata and text. **Does NOT reproduce the raw text block and excludes arbitrary chunk ID numbers from semantic text**:
```text
SECTION: {breadcrumb_path}
ENTITIES: {extracted_entities_str}
LOCATIONS: {locations_str}
SPEAKERS: {speakers_str}
ACTION_TERMS: {action_verbs_str}
RELATION_TERMS: {relation_phrases_str}
EVENT_TERMS: {event_nouns_str}
TEMPORAL_TERMS: {temporal_terms_str}
SEMANTIC_RELATIONS: {semantic_relations_str}
```

### Distributed Context Diagnostic ($N=3$)
Fixed 7-chunk context window ($C_{i-3} \dots C_i \dots C_{i+3}$) to test whether required information is distributed across a broader discourse span. Records `distributed_start_idx`, `distributed_end_idx`, `distributed_window_size`, `distributed_has_left_context`, `distributed_has_right_context`.

---

## 5. Normalized Accessibility Percentile & Dual-Condition Material Improvement

Because $B_0, B_1, B_2, B_3$ and Distributed_$N3$ inhabit different representation spaces, raw rank differences across spaces are not directly comparable.

### Normalized Accessibility Percentile Formula
For dense diagnostic rank $R \in [1, 203]$:
$$\text{Percentile} = 1.0 - \frac{R - 1}{203 - 1}$$
*(Rank #1 = 1.000; Rank #203 = 0.000).*

### Dual-Condition "Material Improvement" Threshold
A representation transition ($B_{k-1} \rightarrow B_k$) is a **material accessibility improvement** iff **BOTH** conditions hold:
1. **Condition A**: Either `prev_rank > 20 AND curr_rank <= 20` OR `curr_percentile - prev_percentile >= 0.10` ($\ge 10$ percentage point increase).
2. **Condition B**: `curr_q_similarity - prev_q_similarity >= 0.05`.

Raw rank deltas (`rank_delta_descriptive_*`) are recorded for descriptive logging only and do not participate in causal classification.

---

## 6. Diagnostic Classification Precedence

```text
IF evidence in canonical Final Top-15:
    CANONICAL_TOP15_RECOVERED

ELIF evidence in canonical Candidate Pool@20:
    RETRIEVAL_FAILURE_DESPITE_ADEQUATE_REPRESENTATION

ELIF B0 dense_diagnostic_rank <= 20:
    RETRIEVAL_FAILURE_DESPITE_ADEQUATE_REPRESENTATION

ELIF B2 produces a material improvement AND B3 does NOT independently produce a material improvement:
    CONTEXT_DEPENDENCE

ELIF B3 produces a material improvement AND B2 does NOT independently produce a material improvement:
    REPRESENTATION_INSUFFICIENCY

ELIF Distributed_N3 produces a material improvement relative to B0:
    DISTRIBUTED_EVIDENCE

ELIF all representations remain weak relative to accessible evidence controls (C12, C65, C155) AND no material transition occurs:
    SEMANTIC_QUERY_MISMATCH

ELSE:
    UNRESOLVED
```

---

## 7. Autopsy Suites & Output Artifacts

- **NO_PATH_FAILURE Autopsies**: `C15`, `C29`, `C42`
- **RANKING_FAILURE Controls**: `C12`, `C65`, `C155`

### Output Files
- `phaseC_results.json`
- `phaseC_evidence_trace.csv`
- `phaseC_representation_metrics.csv`
- `phaseC_question_metrics.csv`
- `phaseC_evidence_representation_accessibility_report.md`
- `phaseC_evidence_representation_accessibility_plan.md`
- `evaluation/run_phaseC_evidence_representation_accessibility.py`
