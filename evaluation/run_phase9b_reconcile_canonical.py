import os
import sys
import json

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_dir = os.path.join(root_dir, "evaluation", "results")
reports_dir = os.path.join(root_dir, "evaluation", "reports")

raw_json_path = os.path.join(results_dir, "phase9b_representation_ablation.json")

with open(raw_json_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# Build Canonical Source of Truth
canonical_data = {
    "metadata": {
        "corpus_id": "comedy_of_errors_v1",
        "corpus": "The Comedy of Errors document.pdf",
        "chunk_count": 203,
        "experiment_id": "phase9b_canonical_reconciliation",
        "date": "2026-09-27",
        "retrieval_configuration": "Q2 Multi-Query Expansion -> Candidate Union N20 -> Cross-Encoder -> Top10",
        "embedding_configuration": "sentence-transformers/all-MiniLM-L6-v2 (384-dim)",
        "ce_configuration": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "target_chunks": ["C15", "C29", "C42", "C61", "C147"],
        "representations": ["R0", "R1", "R2", "R3", "R4"]
    },
    "aggregate_results": raw_data["experiment_b_pipeline"],
    "dense_experiment_a": raw_data["experiment_a_dense"],
    "target_autopsy": raw_data["full_target_autopsy"],
    "target_summary_table": raw_data["target_summary_table"]
}

canonical_json_path = os.path.join(results_dir, "phase9b_canonical_results.json")
with open(canonical_json_path, "w", encoding="utf-8") as f:
    json.dump(canonical_data, f, indent=2)

print(f"Canonical Phase 9B JSON successfully created at: {canonical_json_path}")
