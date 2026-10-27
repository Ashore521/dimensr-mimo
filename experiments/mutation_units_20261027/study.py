"""Reuse the established bounded supervisor; freeze all models before scoring."""
import argparse
import importlib.metadata
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from dimensr_mimo import SearchConfig,audit
from dimensr_mimo.generation import SystemGrammar
from experiments.second_round_20261026.cases import previous
from experiments.compound_search_20261027 import cases,worker
from experiments.native_api_20261026.run import read,write,sha,arrays_hash

legacy=previous('study',cases=cases,worker=worker.base);legacy.HERE,legacy.ROOT=HERE,ROOT
METHODS=('dimensional','compound');LIMIT=1200


def budget():return read(HERE/'BUDGET.json')['start_unix']+LIMIT
legacy.budget=budget


def name(job):return f"{job['case']}_s{job['seed']}_{job['method']}"
def matrix():
    return [dict(case=c,seed=s,method=m,seconds=60) for c in ('polynomial_n0','polynomial_n1')
            for s in (701,702) for m in (('dimensional','compound') if s==701 else ('compound','dimensional'))]



def verify_files(files,root=ROOT):
    for n,h in files.items():
        p=(root/n).resolve()
        if not p.is_relative_to(root.resolve()) or sha(p)!=h:raise RuntimeError('Frozen file drift: '+n)


def configuration():
    cfg=read(HERE.parent/'first_round/FREEZE.json')['config']
    return dict(cfg,operators=cfg['operators']+['exp'],reference_budget='expanded',time_allocation='phased')


def preflight():
    if (HERE/'PREFLIGHT.json').exists():raise RuntimeError('Keep existing preflight.')
    cfg=SearchConfig(**configuration());rows=[]
    for case in cases.CASES:
        spec=cases.design(case)
        grammar=SystemGrammar(tuple(map(cases.unit,spec['features'])),tuple(map(cases.unit,spec['targets'])),tuple(map(cases.unit,spec['constants'])),cfg)
        for seed in cases.seeds(case):
            for split in ('train','validation'):
                X,Y=cases.training_data(case,split,seed)
                assert np.isfinite(X).all() and np.isfinite(Y).all() and np.all(Y.std(axis=0)>0)
                assert len(X)==(96 if split=='train' else 48)
        rows.append(dict(case=case,grammar_outputs=len(grammar.plain),seeds=cases.seeds(case)))
    write(HERE/'PREFLIGHT.json',dict(status='passed',test_labels_accessed=False,rows=rows))


def freeze():
    if (HERE/'FREEZE.json').exists() or (HERE/'BUDGET.json').exists():raise RuntimeError('Already frozen.')
    assert read(HERE/'PREFLIGHT.json')['status']=='passed'
    files=list((ROOT/'src').rglob('*.py'))+list(HERE.glob('*.py'))+list((ROOT/'tests').glob('*.py'))
    for directory in ('first_round','second_round_20261026','typed_coverage_20261026','native_api_20261026','noise_families_20261026','compound_search_20261027'):
        files+=list((HERE.parent/directory).glob('*.py'))
    files += [HERE/'PROTOCOL.md',HERE/'PREFLIGHT.json',ROOT/'pyproject.toml']
    for p in (ROOT/'src').rglob('*.py'):
        dest=HERE/'source'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    b=dict(start_unix=time.time(),limit_seconds=LIMIT,sequential_workers=1,threads=1,system_cap_seconds=60,
        fit_seconds=52,verification_reserve_seconds=600,
        versions={n:importlib.metadata.version(n) for n in ('numpy','scipy','torch','sympy','scikit-learn','psutil')})
    write(HERE/'BUDGET.json',b)
    data={f'{c}_s{s}':arrays_hash(*cases.training_data(c,'train',s),*cases.training_data(c,'validation',s)) for c in cases.CASES for s in cases.seeds(c)}
    write(HERE/'FREEZE.json',dict(frozen_unix=time.time(),jobs=matrix(),config=configuration(),specs={c:cases.design(c) for c in cases.CASES},
        data_sha256=data,files={p.relative_to(ROOT).as_posix():sha(p) for p in files},test_labels_accessed=False,budget=b,
        base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
        paired_test_protection=dict(relative=.10,absolute=1e-8),adoption_rule='Repair gate: all eight focused systems technically verified, including the four previously failed cells. Compound recommendation remains separate and requires all eight paired outputs protected.'))


def train():
    f=read(HERE/'FREEZE.json');verify_files(f['files'])
    if (HERE/'ALL_MODELS_FROZEN.json').exists():verify_files(read(HERE/'ALL_MODELS_FROZEN.json')['files'],HERE);return
    if not (HERE/'TRAINING_STARTED.json').exists():write(HERE/'TRAINING_STARTED.json',dict(start_unix=time.time()))
    for job in f['jobs']:
        verify_files(f['files']);folder=HERE/'runs'/name(job)
        if (folder/'CELL_FROZEN.json').exists():verify_files(read(folder/'CELL_FROZEN.json')['files'],folder);continue
        if (folder/'job.json').exists():raise RuntimeError('Unsealed cell requires incident review: '+str(folder))
        if time.time()+job['seconds']>budget()-600:
            folder.mkdir(parents=True,exist_ok=True);write(folder/'job.json',job);write(folder/'config.json',f['config'])
            write(folder/'resources.json',dict(status='not_started_budget',job=job))
        else:legacy.run_job(job,f['config'],'runs')
        write(folder/'CELL_FROZEN.json',dict(frozen_unix=time.time(),files={p.name:sha(p) for p in folder.iterdir() if p.is_file() and p.name!='CELL_FROZEN.json'}))
    write(HERE/'ALL_MODELS_FROZEN.json',dict(frozen_unix=time.time(),test_labels_accessed=False,freeze_sha256=sha(HERE/'FREEZE.json'),
        files={p.relative_to(HERE).as_posix():sha(p) for p in (HERE/'runs').rglob('*') if p.is_file()}))


def mechanism(folder):
    report=legacy.mechanism(folder)
    report['full_mechanism_pass']=all(r['policy_delta']>0 and r['neural_valid']>0 and r['gp_new_evaluations']>0 and
        r['nsga_generations']>=2 and r['nsga_new_evaluations']>0 for r in report['outputs'])
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('preflight','freeze','train','run'))
    command=p.parse_args().command
    if command=='run':
        preflight();freeze();train()
        from experiments.mutation_units_20261027 import evaluate
        evaluate.evaluate();evaluate.verify()
    else:globals()[command]()
