"""Sequential, externally bounded reselection, with all models sealed before scoring."""
import importlib.metadata
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT))
from dimensr_mimo.audit import data_digest
from experiments.second_round_20261026.cases import previous
from experiments.physical_pool_20261027 import data,worker
read,write,sha=data.read,data.write,data.sha
LIMIT=2400;RESERVE=720
supervisor=previous('study',cases=data.synthetic,worker=worker)
supervisor.HERE=HERE;supervisor.ROOT=ROOT


def budget():return read(HERE/'BUDGET.json')['start_unix']+LIMIT
supervisor.budget=budget


def verify_files(files,root=ROOT):
    for name,expected in files.items():
        path=(root/name).resolve()
        if not path.is_relative_to(root.resolve()) or sha(path)!=expected:
            raise ValueError('Frozen file drift: '+name)


def freeze():
    if (HERE/'FREEZE.json').exists() or (HERE/'BUDGET.json').exists():raise RuntimeError('Keep existing freeze.')
    jobs=data.jobs();assert len(jobs)==24 and len({data.name(j) for j in jobs})==24
    sources={j['source']:sha(ROOT/j['source']) for j in jobs}
    assert sum(read(ROOT/j['source'])['n_outputs'] for j in jobs)==54
    files=list((ROOT/'src').rglob('*.py'))+list((ROOT/'tests').glob('*.py'))+list(HERE.glob('*.py'))
    for directory in ('first_round','second_round_20261026','typed_coverage_20261026','native_api_20261026',
                      'noise_families_20261026','compound_search_20261027','slump_physical_20261026'):
        files+=list((HERE.parent/directory).glob('*.py'))
    files += [ROOT/'examples/physical_coordinates.py',ROOT/'pyproject.toml',HERE/'PROTOCOL.md',
        ROOT/'experiments/slump_physical_20261026/DATA_MANIFEST.json',ROOT/'experiments/reference_budget_20261027/FREEZE.json']
    for path in (ROOT/'src').rglob('*.py'):
        dest=HERE/'source'/path.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
    metadata={data.name(j):dict(spec=data.spec(j),data_sha256={key:data_digest(*data.arrays(j,split))
        for key,split in (('training','train'),('validation','validation'))}) for j in jobs}
    b=dict(start_unix=time.time(),limit_seconds=LIMIT,workers=1,threads=1,per_pool_seconds=60,
        verification_reserve_seconds=RESERVE,searches=0,
        versions={n:importlib.metadata.version(n) for n in ('numpy','scipy','torch','sympy','scikit-learn','psutil')})
    write(HERE/'BUDGET.json',b)
    write(HERE/'FREEZE.json',dict(frozen_unix=time.time(),budget=b,jobs=jobs,metadata=metadata,sources=sources,
        files={p.relative_to(ROOT).as_posix():sha(p) for p in files},test_data_used=False,
        base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()))


def train():
    frozen=read(HERE/'FREEZE.json');verify_files(frozen['files']);verify_files(frozen['sources'])
    if (HERE/'ALL_MODELS_FROZEN.json').exists():
        verify_files(read(HERE/'ALL_MODELS_FROZEN.json')['files'],HERE);return
    if not (HERE/'TRAINING_STARTED.json').exists():
        write(HERE/'TRAINING_STARTED.json',dict(start_unix=time.time(),operation='reselection_only',searches=0))
    for job in frozen['jobs']:
        verify_files(frozen['files']);verify_files(frozen['sources']);folder=HERE/'runs'/data.name(job)
        if (folder/'CELL_FROZEN.json').exists():
            verify_files(read(folder/'CELL_FROZEN.json')['files'],folder);continue
        if (folder/'job.json').exists():raise RuntimeError('Unsealed cell requires incident review: '+data.name(job))
        if time.time()+job['seconds']>budget()-RESERVE:
            folder.mkdir(parents=True,exist_ok=True);write(folder/'job.json',job)
            write(folder/'resources.json',dict(status='not_started_budget',job=job))
        else:supervisor.run_job(job,dict(searches=0,refitting=False),'runs')
        write(folder/'CELL_FROZEN.json',dict(frozen_unix=time.time(),files={p.name:sha(p) for p in folder.iterdir() if p.is_file() and p.name!='CELL_FROZEN.json'}))
    write(HERE/'ALL_MODELS_FROZEN.json',dict(frozen_unix=time.time(),test_data_used=False,freeze_sha256=sha(HERE/'FREEZE.json'),
        files={p.relative_to(HERE).as_posix():sha(p) for p in (HERE/'runs').rglob('*') if p.is_file()}))


if __name__=='__main__':
    command=sys.argv[1]
    if command=='run':
        freeze();train()
        from experiments.physical_pool_20261027.evaluate import evaluate
        evaluate()
    elif command=='freeze':freeze()
    elif command=='train':train()
    else:raise ValueError('Use freeze, train or run.')
