"""Reuse raw-graph verification, but preserve six-cell and archived-control counts."""
from pathlib import Path
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];PREVIOUS=HERE.parent/'slump_physical_20261026'
sys.path.insert(0,str(ROOT))
from experiments.phase_budget_20261027 import study
from experiments.slump_physical_20261026 import evaluate as original,data
from experiments.native_api_20261026.verify import require
original.HERE=HERE
original.study.HERE=HERE
read,write,sha=data.read,data.write,data.sha


def verify():
    if (HERE/'VERIFICATION.json').exists():raise RuntimeError('Keep existing verification.')
    f,m=original.guard();s=read(HERE/'SCORES.json');original.study.check_files(f['controls'])
    require(s['models_sha256']==sha(HERE/'ALL_MODELS_FROZEN.json') and m['frozen_unix']<=s['scored_unix'],'Score freeze chain drift.')
    rows=[]
    for job,score in zip(f['jobs'],s['rows']):
        require(time.time()<original.study.budget(),'Verification budget expired.')
        try:
            require(score['job']==job and score['status']=='complete','Incomplete cell: '+score['status'])
            row=original.verify_cell(job,score,f)
            model=read(HERE/'runs'/original.study.name(job)/'model.json')
            require(model['config']['time_allocation']=='phased','Wrong timing strategy.')
            allocations=[r for r in model['history'] if r['phase']=='phase_budget']
            require(bool(allocations),'No actual phase budget events.')
            require(all(r['neural_until_seconds']<=r['gp_until_seconds']<=r['exploration_until_seconds']==39 for r in allocations),'Incorrect reserved phase boundaries.')
            row['phase_allocations']=allocations
        except Exception as e:row=dict(job=job,verified=False,error=repr(e))
        rows.append(row);print(original.study.name(job),'verified' if row['verified'] else row['error'],flush=True)
    previous=read(PREVIOUS/'SCORES.json');index={(r['job']['seed'],r['job']['method']):r for r in previous['rows']};pairs=[]
    for score in s['rows']:
        job=score['job'];old=index[job['seed'],job['method']];p=dict(seed=job['seed'],initialization=job['method'],available=score['status']==old['status']=='complete')
        if p['available']:
            p.update(global_nrmse=old['nrmse'],phased_nrmse=score['nrmse'],protected=(np.asarray(score['nrmse'])<=np.asarray(old['nrmse'])*1.1+1e-8).tolist(),
                global_mechanism=old['mechanism']['passed'],phased_mechanism=score['mechanism']['passed'],
                global_both_accepted=old['both_accepted'],phased_both_accepted=score['both_accepted'],
                global_wall_seconds=old['resources']['wall_seconds'],phased_wall_seconds=score['resources']['wall_seconds'])
        pairs.append(p)
    complete=[r for r in s['rows'] if r['status']=='complete']
    summary=dict(expected=6,complete=len(complete),verified=sum(r['verified'] for r in rows),new_searches=6,historical_controls=9,
        mechanism_pass=sum(r['mechanism']['passed'] for r in complete),both_accepted=sum(r['both_accepted'] for r in complete),
        protected_outputs=sum(sum(p.get('protected',[])) for p in pairs),expected_outputs=18,pairs=pairs,
        production_default_changed=False,scope='Timing mechanism repair on historical data; not blind validation or evidence of engineering accuracy.')
    write(HERE/'SUMMARY.json',summary)
    write(HERE/'VERIFICATION.json',dict(status='passed' if all(r['verified'] for r in rows) else 'failed',expected_systems=6,verified_systems=sum(r['verified'] for r in rows),
        rows=rows,elapsed_seconds=time.time()-f['budget']['start_unix'],freeze_sha256=sha(HERE/'FREEZE.json'),models_sha256=sha(HERE/'ALL_MODELS_FROZEN.json'),scores_sha256=sha(HERE/'SCORES.json')))


if __name__=='__main__':original.evaluate();verify()
