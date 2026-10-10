import os
import sys
import json
import numpy as np
import pandas as pd

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
json_path = os.path.join(root_dir, "phase10b_candidate_depth_results.json")
autopsy_path = os.path.join(root_dir, "phase10b_target_autopsy.json")

with open(json_path, "r", encoding="utf-8") as f:
    results = json.load(f)

print("=== CHECKING B1 N=20 ACROSS K IN [5, 10, 15, 20] ===")
for r in results["condition_results"]:
    if r["config"] == "B1" and r["candidate_depth_N"] == 20:
        print(f"K={r['selection_budget_K']:2d} | CandRec: {r['candidate_recall_N']*100:.1f}% | FinalRec: {r['final_recall_K']*100:.1f}% | EvPrec: {r['evidence_precision_K']:.3f} | nDCG: {r['ndcg_K']:.3f}")

with open(autopsy_path, "r", encoding="utf-8") as f:
    autopsy = json.load(f)

print("\n=== TARGET AUTOPSY TRACES AT N=20, K=10 ===")
df_auto = pd.DataFrame(autopsy)
subset = df_auto[(df_auto["candidate_depth_N"] == 20) & (df_auto["selection_budget_K"] == 10)]
for idx, row in subset.iterrows():
    print(f"Config: {row['config']} | Target: {row['target_chunk']:4s} | DenseRank: {row['dense_rank']:3d} | InCand: {row['in_candidate_pool']} | CandRank: {str(row['candidate_rank']):4s} | CERank: {str(row['CE_rank']):4s} | FinalTopK: {row['in_final_TopK']} | Stage: {row['failure_stage']}")
