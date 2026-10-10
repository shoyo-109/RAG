import json

with open('evaluation/experiments/results/phase2_experiments_dump.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Total experiments completed: {len(data)}\n")
print(f"| {'Configuration':35s} | {'Rec@10':7s} | {'EvP@10':7s} | {'PriP@10':7s} | {'MRR':6s} | {'nDCG':6s} | {'Comp':6s} | {'ReqComp':7s} | {'Redun':6s} |")
print('|' + '-'*37 + '|' + '-'*9 + '|' + '-'*9 + '|' + '-'*9 + '|' + '-'*8 + '|' + '-'*8 + '|' + '-'*8 + '|' + '-'*9 + '|' + '-'*8 + '|')

for exp_id, exp_data in data.items():
    s = exp_data['summary']
    r10 = f"{s['recall_10']*100:.1f}%"
    ev10 = f"{s['ev_p_10']*100:.1f}%"
    pri10 = f"{s['pri_p_10']*100:.1f}%"
    mrr = f"{s['mrr']:.3f}"
    ndcg = f"{s['ndcg_10']:.3f}"
    comp = f"{s['completeness']*100:.1f}%"
    req_comp = f"{s['req_completeness']*100:.1f}%"
    red = f"{s['redundancy']*100:.1f}%"
    print(f"| {exp_id:35s} | {r10:7s} | {ev10:7s} | {pri10:7s} | {mrr:6s} | {ndcg:6s} | {comp:6s} | {req_comp:7s} | {red:6s} |")
