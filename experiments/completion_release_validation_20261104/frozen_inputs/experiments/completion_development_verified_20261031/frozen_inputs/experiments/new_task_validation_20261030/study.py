"""Freeze and supervise the locked 16-pair new-task matrix without retries."""
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from dimensr_mimo import SearchConfig
from experiments.new_task_validation_20261030 import cases, worker
from experiments.precision_repair_20261028 import study as prior
from experiments.second_round_20261026.cases import previous
from experiments.native_api_20261026.run import read, write, sha, arrays_hash

METHODS = ('base', 'repair')
LIMIT = 2880
PRIOR = ROOT/'experiments/precision_repair_20261028'
supervisor = previous('study', cases=cases, worker=worker)
supervisor.HERE, supervisor.ROOT = HERE, ROOT
name, mechanism = prior.name, prior.mechanism


def verify_files(files, root=None):
    root = ROOT if root is None else root
    for n,h in files.items():
        path = (root/n).resolve()
        if not path.is_relative_to(root.resolve()) or sha(path)!=h:
            raise RuntimeError('Frozen file drift: '+n)


def check_candidate_lock():
    lock = read(HERE/'CANDIDATE_LOCK.json')
    actual = {p.relative_to(ROOT).as_posix() for p in (ROOT/'src').rglob('*.py')}
    if len(lock['source_files'])!=40 or set(lock['source_files'])!=actual:
        raise RuntimeError('Candidate source manifest drift.')
    verify_files(lock['source_files'])
    if sha(PRIOR/'SUMMARY.json')!=lock['source_summary_sha256']:
        raise RuntimeError('Historical candidate summary drift.')


def budget():
    return read(HERE/'BUDGET.json')['start_unix']+LIMIT


supervisor.budget = budget


def matrix():
    return [dict(case=c,seed=s,method=m,seconds=60) for c in cases.CASES
            for s in cases.seeds(c) for m in (METHODS if s%2 else METHODS[::-1])]


def configuration():
    if read(PRIOR/'ALL_MODELS_FROZEN.json')['freeze_sha256']!=sha(PRIOR/'FREEZE.json'):
        raise RuntimeError('Historical frozen configuration drift.')
    return read(PRIOR/'FREEZE.json')['config']


def preflight():
    check_candidate_lock()
    if (HERE/'PREFLIGHT.json').exists():
        raise RuntimeError('Keep existing preflight.')
    assert len(matrix())==32 and len({name(j) for j in matrix()})==32
    for method in METHODS:
        for case in cases.CASES:
            SearchConfig(**dict(configuration(),precision_repair=method=='repair',final_selection='precision',
                acceptance=cases.design(case)['acceptance'])).validate()
    rows = []
    for case in cases.CASES:
        for seed in cases.seeds(case):
            X,Y = cases.training_data(case,'train',seed)
            V,Z = cases.training_data(case,'validation',seed)
            assert X.shape==Y.shape==(96,2) and V.shape==Z.shape==(48,2)
            assert all(np.isfinite(a).all() for a in (X,Y,V,Z)) and np.all(Y.std(axis=0)>0)
            rows.append(dict(case=case,seed=seed,data_sha256=arrays_hash(X,Y,V,Z)))
    write(HERE/'PREFLIGHT.json',dict(status='passed',rows=rows,test_labels_accessed=False))


def freeze_inputs():
    # The audited predecessor's complete dependency closure includes test fixtures.
    files = [ROOT/n for n in read(PRIOR/'FREEZE.json')['files']]
    files += [PRIOR/n for n in ('FREEZE.json','SUMMARY.json','ALL_MODELS_FROZEN.json','SCORES.json',
        'VERIFICATION.json','POST_RUN_REVIEW.md','HARNESS_REVIEW.md','FROZEN_INPUT_ARCHIVE.json','post_review_checks.json')]
    files += [PRIOR/n for n in read(PRIOR/'ALL_MODELS_FROZEN.json')['files']]
    files += list((ROOT/'src').rglob('*.py'))+list((ROOT/'tests').glob('*.py'))+list(HERE.glob('*.py'))
    files += list(HERE.glob('*.md'))+list((HERE/'validation').glob('*.log'))
    files += list((HERE/'history_sources').rglob('*'))
    files += [HERE/n for n in ('PLAN.md','PROTOCOL.md','PRE_RUN_REVIEW.md','TEST_VALIDATION.json','PREFLIGHT.json',
        'CANDIDATE_LOCK.json','NOVELTY_AUDIT.json','ORACLE_AUDIT.json','DESIGN_AUDIT.json')]
    files += [ROOT/'pyproject.toml']
    return sorted(set(p for p in files if not p.is_dir()))


def freeze():
    check_candidate_lock()
    if (HERE/'FREEZE.json').exists() or (HERE/'BUDGET.json').exists():
        raise RuntimeError('Already frozen.')
    assert read(HERE/'PREFLIGHT.json')['status']=='passed'
    files = freeze_inputs()
    # Check required inputs before leaving any partial source-copy checkpoint.
    for path in files:
        if not path.is_file():
            raise RuntimeError('Missing freeze input: '+str(path))
    for path in (ROOT/'src').rglob('*.py'):
        dest = HERE/'source'/path.relative_to(ROOT)
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,dest)
        files.append(dest)
    b = dict(start_unix=time.time(),limit_seconds=LIMIT,sequential_workers=1,threads=1,
        system_cap_seconds=60,fit_seconds=52,verification_reserve_seconds=600,
        versions={n:importlib.metadata.version(n) for n in ('numpy','scipy','torch','sympy','scikit-learn','psutil')})
    write(HERE/'BUDGET.json',b)
    files.append(HERE/'BUDGET.json')
    write(HERE/'FREEZE.json',dict(frozen_unix=time.time(),jobs=matrix(),config=configuration(),
        specs={c:cases.design(c) for c in cases.CASES},
        data_sha256={f'{c}_s{s}':arrays_hash(*cases.training_data(c,'train',s),*cases.training_data(c,'validation',s))
                     for c in cases.CASES for s in cases.seeds(c)},
        files={p.relative_to(ROOT).as_posix():sha(p) for p in files},budget=b,test_labels_accessed=False,
        base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
        paired_test_protection=dict(relative=.1,absolute=1e-8)))


def train():
    check_candidate_lock()
    f = read(HERE/'FREEZE.json')
    verify_files(f['files'])
    if (HERE/'ALL_MODELS_FROZEN.json').exists():
        verify_files(read(HERE/'ALL_MODELS_FROZEN.json')['files'],HERE)
        return
    if not (HERE/'TRAINING_STARTED.json').exists():
        write(HERE/'TRAINING_STARTED.json',dict(start_unix=time.time()))
    for job in f['jobs']:
        check_candidate_lock()
        verify_files(f['files'])
        folder = HERE/'runs'/name(job)
        if (folder/'CELL_FROZEN.json').exists():
            verify_files(read(folder/'CELL_FROZEN.json')['files'],folder)
            continue
        if (folder/'job.json').exists():
            raise RuntimeError('Unsealed cell requires incident review: '+name(job))
        if time.time()+job['seconds']>budget()-600:
            folder.mkdir(parents=True,exist_ok=True)
            write(folder/'job.json',job)
            write(folder/'config.json',f['config'])
            write(folder/'resources.json',dict(status='not_started_budget',job=job))
        else:
            supervisor.run_job(job,f['config'],'runs')
        write(folder/'CELL_FROZEN.json',dict(frozen_unix=time.time(),files={p.name:sha(p)
            for p in folder.iterdir() if p.is_file() and p.name!='CELL_FROZEN.json'}))
    write(HERE/'ALL_MODELS_FROZEN.json',dict(frozen_unix=time.time(),test_labels_accessed=False,
        freeze_sha256=sha(HERE/'FREEZE.json'),files={p.relative_to(HERE).as_posix():sha(p)
        for p in (HERE/'runs').rglob('*') if p.is_file()}))


if __name__=='__main__':
    command = sys.argv[1]
    if command=='run':
        preflight(); freeze(); train()
        from experiments.new_task_validation_20261030.evaluate import run
        run()
    elif command in ('preflight','freeze','train'):
        globals()[command]()
    else:
        raise ValueError('Use preflight, freeze, train or run.')
