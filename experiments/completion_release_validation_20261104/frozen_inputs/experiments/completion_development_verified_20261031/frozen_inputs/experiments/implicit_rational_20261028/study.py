"""Freeze, sequentially supervise, and seal the complete protected-closure matrix."""
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
from experiments.compound_search_20261027 import cases, study as prior
from experiments.second_round_20261026.cases import previous
from experiments.native_api_20261026.run import read, write, sha, arrays_hash
from experiments.implicit_rational_20261028 import worker

METHODS = ('compound', 'implicit')
LIMIT = 2880
supervisor = previous('study', cases=cases, worker=worker)
supervisor.HERE, supervisor.ROOT = HERE, ROOT
verify_files, name = prior.verify_files, prior.name


def mechanism(folder):
    report=prior.mechanism(folder)
    events=[json.loads(line) for line in (folder/'events.jsonl').read_text(encoding='utf-8').splitlines()]
    generations={r['step'] for r in events if r['phase']=='nsga2' and r['new_evaluations']>0}
    report['nsga_generations_with_new_evaluations']=len(generations)
    report['full_mechanism_pass']=report['full_mechanism_pass'] and len(generations)>=2
    return report


def budget():
    return read(HERE/'BUDGET.json')['start_unix'] + LIMIT


supervisor.budget = budget


def matrix():
    return [dict(case=c, seed=s, method=m, seconds=60) for c in cases.CASES
            for s in cases.seeds(c) for m in (METHODS if s % 2 else METHODS[::-1])]


def configuration():
    return dict(prior.configuration(), initialization='compound', sharing_strategy='closure', wall_seconds=52)


def preflight():
    if (HERE/'PREFLIGHT.json').exists():
        raise RuntimeError('Keep existing preflight.')
    assert len(matrix()) == 34 and len({name(j) for j in matrix()}) == 34
    for method in METHODS:
        for case in cases.CASES:
            SearchConfig(**dict(configuration(), initialization=method, final_selection='precision', acceptance=cases.design(case)['acceptance'])).validate()
    rows = []
    for case in cases.CASES:
        for seed in cases.seeds(case):
            X, Y = cases.training_data(case, 'train', seed)
            V, Z = cases.training_data(case, 'validation', seed)
            assert X.shape == (96, 2) and V.shape == (48, 2)
            assert all(np.isfinite(a).all() for a in (X, Y, V, Z)) and np.all(Y.std(axis=0) > 0)
            rows.append(dict(case=case, seed=seed, data_sha256=arrays_hash(X, Y, V, Z)))
    write(HERE/'PREFLIGHT.json', dict(status='passed', rows=rows, test_labels_accessed=False))


def freeze():
    if (HERE/'FREEZE.json').exists() or (HERE/'BUDGET.json').exists():
        raise RuntimeError('Already frozen.')
    assert read(HERE/'PREFLIGHT.json')['status'] == 'passed'
    files = list((ROOT/'src').rglob('*.py')) + list((ROOT/'tests').glob('*.py')) + list(HERE.glob('*.py'))
    files += list((HERE/'validation').glob('*.log')) + [HERE/'TEST_VALIDATION.json']
    for directory in ('first_round', 'second_round_20261026', 'typed_coverage_20261026', 'native_api_20261026',
                      'noise_families_20261026', 'compound_search_20261027', 'matched_sharing_20261028',
                      'protected_closure_20261028', 'balanced_closure_20261028'):
        files += list((HERE.parent/directory).glob('*.py'))
    files += [HERE/'PROTOCOL.md', HERE/'PLAN.md', HERE/'PRE_RUN_REVIEW.md', HERE/'TRAINING_DIAGNOSIS.md', HERE/'PREFLIGHT.json', ROOT/'pyproject.toml',
              HERE.parent/'physical_pool_20261027/source/src/dimensr_mimo/genetics.py',
              HERE.parent/'balanced_closure_20261028/source/src/dimensr_mimo/structure.py',
              ROOT/'examples/physical_coordinates.py', HERE.parent/'first_round/FREEZE.json']
    for p in (ROOT/'src').rglob('*.py'):
        dest = HERE/'source'/p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(p, dest)
    b = dict(start_unix=time.time(), limit_seconds=LIMIT, sequential_workers=1, threads=1,
        system_cap_seconds=60, fit_seconds=52, verification_reserve_seconds=600,
        versions={n:importlib.metadata.version(n) for n in ('numpy', 'scipy', 'torch', 'sympy', 'scikit-learn', 'psutil')})
    write(HERE/'BUDGET.json', b)
    files.append(HERE/'BUDGET.json')
    write(HERE/'FREEZE.json', dict(frozen_unix=time.time(), jobs=matrix(), config=configuration(),
        specs={c:cases.design(c) for c in cases.CASES},
        data_sha256={f'{c}_s{s}':arrays_hash(*cases.training_data(c, 'train', s), *cases.training_data(c, 'validation', s))
                     for c in cases.CASES for s in cases.seeds(c)},
        files={p.relative_to(ROOT).as_posix():sha(p) for p in files}, budget=b, test_labels_accessed=False,
        base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip(),
        paired_test_protection=dict(relative=.1, absolute=1e-8)))


def train():
    f = read(HERE/'FREEZE.json'); verify_files(f['files'])
    if (HERE/'ALL_MODELS_FROZEN.json').exists():
        verify_files(read(HERE/'ALL_MODELS_FROZEN.json')['files'], HERE); return
    if not (HERE/'TRAINING_STARTED.json').exists():
        write(HERE/'TRAINING_STARTED.json', dict(start_unix=time.time()))
    for job in f['jobs']:
        verify_files(f['files']); folder = HERE/'runs'/name(job)
        if (folder/'CELL_FROZEN.json').exists():
            verify_files(read(folder/'CELL_FROZEN.json')['files'], folder); continue
        if (folder/'job.json').exists():
            raise RuntimeError('Unsealed cell requires incident review: ' + name(job))
        if time.time() + job['seconds'] > budget() - 600:
            folder.mkdir(parents=True, exist_ok=True)
            write(folder/'job.json', job); write(folder/'config.json', f['config'])
            write(folder/'resources.json', dict(status='not_started_budget', job=job))
        else:
            supervisor.run_job(job, f['config'], 'runs')
        write(folder/'CELL_FROZEN.json', dict(frozen_unix=time.time(), files={p.name:sha(p)
            for p in folder.iterdir() if p.is_file() and p.name != 'CELL_FROZEN.json'}))
    write(HERE/'ALL_MODELS_FROZEN.json', dict(frozen_unix=time.time(), test_labels_accessed=False,
        freeze_sha256=sha(HERE/'FREEZE.json'), files={p.relative_to(HERE).as_posix():sha(p)
        for p in (HERE/'runs').rglob('*') if p.is_file()}))


if __name__ == '__main__':
    command = sys.argv[1]
    if command == 'run':
        preflight(); freeze(); train()
        from experiments.implicit_rational_20261028.evaluate import evaluate, verify
        evaluate(); verify()
    elif command in ('preflight', 'freeze', 'train'):
        globals()[command]()
    else:
        raise ValueError('Use preflight, freeze, train or run.')
