"""Additional read-only denominator, saved-history and report-field checks."""
import json,math,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent.parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from experiments.completion_development_verified_20261031 import archive_io
read=lambda p:json.loads(archive_io.disk_path(p).read_text(encoding='utf-8'))
r=read(HERE/'post_review_checks.json');scores=read(HERE/'SCORES.json')['rows']
key=lambda j:(j['case'],j['seed'],j['method'])
ix={key(x['job']):x for x in scores};issues=[]
def check(ok,label):
    if not ok:issues.append(label)
old=ROOT/'experiments/new_task_validation_20261030'
old_scores=read(old/'SCORES.json')['rows']
check(len(old_scores)==len({key(x['job']) for x in old_scores})==32,'old_full_unique_score_denominator')
check(all(x['status']=='complete' and type(x['both_accepted']) is bool and len(x['clean_nrmse'])==len(x['observed_nrmse'])==2 and all(math.isfinite(v) and v>=0 for v in x['clean_nrmse']+x['observed_nrmse']) for x in old_scores),'all_old_rows_complete_finite')
history_rows=0;timing_paths=0
for cell in r['systems']:
    job=cell['job'];tag=f"{job['case']}_s{job['seed']}_{job['method']}";folder=HERE/'runs'/tag
    model=read(folder/'model.json');bundle=read(folder/'delivery.json')
    check(np.allclose(cell['train_nrmse'],bundle['train_nrmse'],rtol=1e-9,atol=1e-11) and np.allclose(cell['validation_nrmse'],bundle['validation_nrmse'],rtol=1e-9,atol=1e-11),tag+':bundle_errors')
    check(math.isclose(float(np.mean(cell['clean_nrmse'])),ix[key(job)]['mean_clean_nrmse'],rel_tol=1e-12,abs_tol=1e-14),tag+':score_mean_field')
    events=[json.loads(line) for line in (folder/'events.jsonl').read_text(encoding='utf-8').splitlines()]
    clean=[{k:v for k,v in e.items() if k not in ['output','monotonic']} for e in events]
    check(clean==model['history'],tag+':events_equal_saved_history');history_rows+=len(events)
    if job['method']=='base':
        prior=read(old/'runs'/f"{job['case']}_s{job['seed']}_repair"/'model.json')
        change=next(x for x in r['control_identity'] if x['job']==job)
        for d in change['changes']['model.json']:
            p=d['path'].split('/');index=int(p[2]);timing_paths+=1
            check(p[1]=='history' and model['history'][index]['phase']==prior['history'][index]['phase']=='phase_budget',tag+':classified_phase_budget_path')
        for m in [model,prior]:
            for epoch in range(4):
                steps=[x['step'] for x in m['history'] if x['phase']=='gp' and x['step'] in [2*epoch,2*epoch+1]]
                check(steps==[2*epoch,2*epoch+1],tag+':distinct_ordered_gp_steps')

family=r['family_discovery'];fields=next(iter(family.values())).keys()
totals={k:sum(x[k] for x in family.values()) for k in fields}
check(totals['new_evaluations']==totals['valid']==166 and totals['protected']==totals['validation_retained']==54,'candidate_supplement_166_54')
check(totals['compositional_linear_subproblems']==404 and totals['implicit_linear_subproblems']==78953 and totals['linear_subproblems']==311102,'linear_work_totals')
check(totals['compositional_returned_roots']==119 and totals['implicit_returned_roots']==138 and totals['compositional_valid_systems']==84 and totals['compositional_validation_retained']==35,'compositional_root_system_totals')
result=dict(status='passed' if not issues else 'findings',issues=issues,main_check_receipt_sha256=archive_io.sha(HERE/'post_review_checks.json'),old_score_rows=32,saved_history_rows_compared=history_rows,classified_phase_budget_paths_checked=timing_paths,discovery_totals=totals)
(HERE/'post_review_validation/extra_checks.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
