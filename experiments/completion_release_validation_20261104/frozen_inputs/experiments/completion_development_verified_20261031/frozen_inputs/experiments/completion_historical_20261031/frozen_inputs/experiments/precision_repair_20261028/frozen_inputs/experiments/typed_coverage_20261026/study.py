"""Sequential bounded supervisor, source/model seals, then uniform scoring."""
import argparse
import importlib.metadata
from pathlib import Path
import platform
import subprocess
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from experiments.second_round_20261026.cases import previous
from experiments.typed_coverage_20261026 import cases,worker,delivery
from experiments.guarded_selection_20261026 import selector
legacy=previous('study',cases=cases,worker=worker.base)
legacy.HERE,legacy.ROOT=HERE,ROOT
read,write,sha=worker.read,worker.write,worker.sha
METHODS=('prefix','balanced','independent')
LIMIT=2760


def matrix():return [dict(case=c,seed=s,method=m,seconds=60) for c in cases.CASES for s in cases.SEEDS for m in METHODS]
def budget():return read(HERE/'BUDGET.json')['start_unix']+LIMIT
legacy.budget=budget
def name(job):return f"{job['case']}_s{job['seed']}_{job['method']}"


def verify_files(files,root=ROOT):
    for n,h in files.items():
        p=(root/n).resolve()
        if not p.is_relative_to(root.resolve()) or sha(p)!=h:raise RuntimeError('Frozen file drift: '+n)


def freeze():
    if (HERE/'FREEZE.json').exists() or (HERE/'BUDGET.json').exists():raise RuntimeError('Already started.')
    b=dict(start_unix=time.time(),limit_seconds=LIMIT,sequential_workers=1,threads=1,system_cap_seconds=60,
        independent_output_cap_seconds=30,verification_reserve_seconds=600,python=sys.version,platform=platform.platform(),
        packages={n:importlib.metadata.version(n) for n in ('numpy','scipy','torch','sympy','scikit-learn','psutil')})
    write(HERE/'BUDGET.json',b)
    files=list((ROOT/'src').rglob('*.py'))+list(HERE.glob('*.py'))+list((ROOT/'tests').glob('*.py'))
    for directory in ('first_round','second_round_20261026','guarded_selection_20261026','shared_discovery_20261026'):
        files+=list((HERE.parent/directory).glob('*.py'))
    files += [HERE/n for n in ('PROTOCOL.md','PREFLIGHT.json','BUDGET.json','calibration/PROTOCOL.json','calibration/VERIFICATION.json','GRAMMAR_PREFLIGHT.json')]
    files += [ROOT/n for n in ('pyproject.toml','MANIFEST.in','docs/environment.txt','experiments/first_round/FREEZE.json')]
    write(HERE/'FREEZE.json',dict(jobs=matrix(),config=read(HERE.parent/'first_round/FREEZE.json')['config'],policy=selector.POLICY,
        specs={c:cases.design(c) for c in cases.CASES},budget=b,frozen_unix=time.time(),test_labels_accessed=False,
        files={p.relative_to(ROOT).as_posix():sha(p) for p in files},base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()))


def checkpoint(folder):
    write(folder/'CELL_FROZEN.json',dict(frozen_unix=time.time(),files={p.name:sha(p) for p in folder.iterdir() if p.is_file() and p.name!='CELL_FROZEN.json'}))


def train():
    f=read(HERE/'FREEZE.json');verify_files(f['files'])
    if (HERE/'ALL_MODELS_FROZEN.json').exists():
        verify_files(read(HERE/'ALL_MODELS_FROZEN.json')['files'],HERE);return
    if not (HERE/'TRAINING_STARTED.json').exists():write(HERE/'TRAINING_STARTED.json',dict(start_unix=time.time()))
    for job in f['jobs']:
        verify_files(f['files']);folder=HERE/'runs'/name(job)
        if (folder/'CELL_FROZEN.json').exists():verify_files(read(folder/'CELL_FROZEN.json')['files'],folder);continue
        if (folder/'job.json').exists():raise RuntimeError('Unsealed cell requires incident review: '+str(folder))
        if time.time()+job['seconds']>budget()-600:
            folder.mkdir(parents=True,exist_ok=True);write(folder/'job.json',job);write(folder/'config.json',f['config'])
            write(folder/'resources.json',dict(status='not_started_budget',job=job))
        else:legacy.run_job(job,f['config'],'runs')
        checkpoint(folder)
    write(HERE/'ALL_MODELS_FROZEN.json',dict(frozen_unix=time.time(),test_labels_accessed=False,freeze_sha256=sha(HERE/'FREEZE.json'),
        files={p.relative_to(HERE).as_posix():sha(p) for p in (HERE/'runs').rglob('*') if p.is_file()}))


def mechanism(folder):
    report=legacy.mechanism(folder)
    report['full_mechanism_pass']=all(r['policy_delta']>0 and r['neural_valid']>0 and r['gp_new_evaluations']>0 and
        r['nsga_generations']>=2 and r['nsga_new_evaluations']>0 for r in report['outputs'])
    return report


def evaluate():
    from dimensr_mimo import DimenSRMIMORegressor
    from dimensr_mimo._single import DimenSRRegressor
    if (HERE/'SCORES.json').exists():raise RuntimeError('Already scored.')
    f,m=read(HERE/'FREEZE.json'),read(HERE/'ALL_MODELS_FROZEN.json')
    verify_files(f['files']);verify_files(m['files'],HERE)
    if m['freeze_sha256']!=sha(HERE/'FREEZE.json'):raise RuntimeError('Freeze mismatch.')
    rows=[]
    for job in f['jobs']:
        if time.time()>=budget():raise RuntimeError('Scoring budget expired.')
        folder=HERE/'runs'/name(job);row=dict(job=job,resources=read(folder/'resources.json'))
        if row['resources']['status']=='complete':
            try:
                result=read(folder/'result.json');spec=cases.design(job['case']);X,Y=cases.generate(job['case'],'test',job['seed'])
                _,U=cases.training_data(job['case'],'train',job['seed']);single=job['method']=='independent'
                cls=DimenSRRegressor if single else DimenSRMIMORegressor;models=[cls.load(folder/n) for n in result['models']]
                pred=np.column_stack([z.predict(X) for z in models]) if single else models[0].predict(X)
                if not np.isfinite(pred).all():raise ValueError('Nonfinite held-out predictions.')
                e=np.sqrt(np.mean((pred-Y)**2,axis=0))/U.std(axis=0);bundle=read(folder/'delivery.json')
                row.update(status='complete',nrmse=e.tolist(),mean_nrmse=float(e.mean()),structure=bundle['structure'],
                    test_accepted=delivery.accepted(e,bundle['structure'],spec['acceptance']),development_accepted=bundle['development_accepted'],
                    mechanism=mechanism(folder),domain_pass=all(a['status']=='enclosed' for a in bundle['domain_audits']))
            except Exception as e:row.update(status='scoring_failed',error=repr(e))
        else:row['status']=row['resources']['status']
        rows.append(row)
    write(HERE/'SCORES.json',dict(rows=rows,scored_unix=time.time(),all_models_freeze_sha256=sha(HERE/'ALL_MODELS_FROZEN.json')))


def run():
    train();evaluate()
    from experiments.typed_coverage_20261026.verify import verify
    verify()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('freeze','train','evaluate','run'));globals()[p.parse_args().command]()
