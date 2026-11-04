"""Frozen ten-system training-only diagnostic using the established supervisor."""
import importlib.metadata
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from experiments.matched_sharing_20261028 import study as prior
from experiments.compound_search_20261027 import cases
from experiments.second_round_20261026.cases import previous
from experiments.native_api_20261026.run import read, write, sha, arrays_hash
from experiments.search_lineage_20261028 import worker

PAIRS = (('damped_n0', 701), ('damped_n1', 702), ('rational_n0', 702),
         ('rational_variant', 711), ('polynomial_variant', 711))
LIMIT = 1200
supervisor = previous('study', cases=cases, worker=worker)
supervisor.HERE, supervisor.ROOT = HERE, ROOT
supervisor.budget = lambda: read(HERE/'BUDGET.json')['start_unix']+LIMIT


def matrix():
    return [dict(case=c, seed=s, method=m, seconds=60) for c, s in PAIRS
            for m in (('random', 'matched') if s % 2 else ('matched', 'random'))]


def freeze():
    if (HERE/'FREEZE.json').exists() or (HERE/'BUDGET.json').exists():
        raise RuntimeError('Retain the existing freeze.')
    check = read(HERE/'SELFCHECK.json')
    assert check['status'] == 'passed'
    assert read(HERE/'SUPERVISOR_SELFCHECK.json')['status'] == 'passed'
    for name, expected in check['files'].items():
        assert sha(HERE/name) == expected, 'Selfcheck source drift: '+name
    files = list((ROOT/'src').rglob('*.py')) + list(HERE.glob('*.py'))
    for directory in ('first_round', 'second_round_20261026', 'typed_coverage_20261026', 'native_api_20261026',
                      'noise_families_20261026', 'compound_search_20261027', 'matched_sharing_20261028'):
        files += list((HERE.parent/directory).glob('*.py'))
    files += [HERE/'PROTOCOL.md', HERE/'SELFCHECK.json', HERE/'SUPERVISOR_SELFCHECK.json', HERE/'PRE_RUN_REVIEW.md', ROOT/'pyproject.toml',
              ROOT/'examples/physical_coordinates.py', HERE.parent/'first_round/FREEZE.json']
    for p in (ROOT/'src').rglob('*.py'):
        assert p.read_bytes() == subprocess.check_output(['git', 'show', '652b05884269c7563b101e0a14a9b15f749bbc6f:'+p.relative_to(ROOT).as_posix()])
        dest = HERE/'source'/p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
    data = {f'{c}_s{s}':arrays_hash(*cases.training_data(c, 'train', s), *cases.training_data(c, 'validation', s)) for c,s in PAIRS}
    budget = dict(start_unix=time.time(), limit_seconds=LIMIT, sequential_workers=1, threads=1,
        system_cap_seconds=60, fit_seconds=52, verification_reserve_seconds=300,
        versions={n:importlib.metadata.version(n) for n in ('numpy', 'scipy', 'torch', 'sympy', 'scikit-learn', 'psutil')})
    write(HERE/'BUDGET.json', budget)
    write(HERE/'FREEZE.json', dict(frozen_unix=time.time(), jobs=matrix(), config=prior.configuration(),
        data_sha256=data, files={p.relative_to(ROOT).as_posix():sha(p) for p in files}, budget=budget,
        test_data_accessed=False, base_commit='652b05884269c7563b101e0a14a9b15f749bbc6f'))


def train():
    frozen = read(HERE/'FREEZE.json')
    prior.verify_files(frozen['files'])
    if (HERE/'ALL_MODELS_FROZEN.json').exists():
        prior.verify_files(read(HERE/'ALL_MODELS_FROZEN.json')['files'], HERE)
        return
    for job in frozen['jobs']:
        prior.verify_files(frozen['files'])
        folder = HERE/'runs'/prior.name(job)
        if (folder/'CELL_FROZEN.json').exists():
            prior.verify_files(read(folder/'CELL_FROZEN.json')['files'], folder)
            continue
        if (folder/'job.json').exists():
            raise RuntimeError('Interrupted cell requires incident review: '+folder.name)
        if time.time()+60 > supervisor.budget()-300:
            folder.mkdir(parents=True, exist_ok=True)
            write(folder/'job.json', job)
            write(folder/'resources.json', dict(status='not_started_budget', job=job))
        else:
            supervisor.run_job(job, frozen['config'], 'runs')
        write(folder/'CELL_FROZEN.json', dict(frozen_unix=time.time(), files={p.name:sha(p)
            for p in folder.iterdir() if p.is_file() and p.name != 'CELL_FROZEN.json'}))
    write(HERE/'ALL_MODELS_FROZEN.json', dict(frozen_unix=time.time(), freeze_sha256=sha(HERE/'FREEZE.json'),
        test_data_accessed=False, files={p.relative_to(HERE).as_posix():sha(p) for p in (HERE/'runs').rglob('*') if p.is_file()}))


if __name__ == '__main__':
    if sys.argv[1] == 'run':
        freeze()
        train()
        from experiments.search_lineage_20261028.verify import verify
        verify()
    elif sys.argv[1] == 'train':
        train()
    else:
        raise ValueError('Use run or train.')
