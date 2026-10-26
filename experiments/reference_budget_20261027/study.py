"""Six same-budget searches with expanded reference costs and archived controls."""
import importlib.metadata
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];PREVIOUS=HERE.parent/'grammar_cost_20261027'
sys.path.insert(0,str(ROOT))
from experiments.phase_budget_20261027 import study as phase
from experiments.slump_physical_20261026 import data
supervisor=phase.supervisor;supervisor.HERE=HERE
read,write,sha=data.read,data.write,data.sha


def freeze():
    if (HERE/'FREEZE.json').exists():raise RuntimeError('Already frozen.')
    previous=read(PREVIOUS/'FREEZE.json');jobs=previous['jobs'];config=dict(previous['config'],reference_budget='expanded')
    shutil.copyfile(PREVIOUS/'DATA_MANIFEST.json',HERE/'DATA_MANIFEST.json');controls={}
    for directory in (PREVIOUS,HERE.parent/'phase_budget_20261027',HERE.parent/'slump_physical_20261026'):
        for p in sorted((directory/'runs').glob('*/model*.json')):controls[p.relative_to(ROOT).as_posix()]=sha(p)
        for name in ('SCORES.json','VERIFICATION.json','ALL_MODELS_FROZEN.json','FREEZE.json'):
            p=directory/name;controls[p.relative_to(ROOT).as_posix()]=sha(p)
    write(HERE/'CONTROLS.json',dict(files=controls,new_searches=6,paired_token_reference_controls=6,earlier_controls=15))
    files=list((ROOT/'src').rglob('*.py'))+list(HERE.glob('*.py'))+list((ROOT/'tests').glob('*.py'))
    for directory in ('first_round','second_round_20261026','typed_coverage_20261026','native_api_20261026','slump_physical_20261026','phase_budget_20261027','grammar_cost_20261027'):
        files+=list((HERE.parent/directory).glob('*.py'))
    files += [HERE/n for n in ('PLAN.md','PROTOCOL.md','CONTROLS.json','DATA_MANIFEST.json')]
    files += [ROOT/'pyproject.toml',ROOT/'MANIFEST.in']
    for p in (ROOT/'src').rglob('*.py'):
        dest=HERE/'source'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    b=dict(start_unix=time.time(),limit_seconds=1200,system_seconds=60,fit_seconds=52,verification_reserve_seconds=600,
        workers=1,threads=1,versions={n:importlib.metadata.version(n) for n in ('numpy','scipy','torch','sympy','scikit-learn','psutil')})
    write(HERE/'BUDGET.json',b)
    write(HERE/'FREEZE.json',dict(frozen_unix=time.time(),jobs=jobs,config=config,budget=b,
        search_limits=previous['search_limits'],physical_limits=previous['physical_limits'],constant_units=previous['constant_units'],
        data_sha256=previous['data_sha256'],files={p.relative_to(ROOT).as_posix():sha(p) for p in files},
        controls=controls,test_labels_accessed=False,base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()))


if __name__=='__main__':
    if len(sys.argv)!=2:raise ValueError('Use freeze or train.')
    if sys.argv[1]=='freeze':freeze()
    elif sys.argv[1]=='train':supervisor.train()
    else:raise ValueError('Unknown stage.')
