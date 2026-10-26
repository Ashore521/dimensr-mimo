"""Frozen 2x2 decomposition of seed-family alignment and unit composition."""
import argparse
import importlib.metadata
import importlib.util
from pathlib import Path
import platform
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from experiments.typed_coverage_20261026 import cases
from experiments.unit_composition_20261026 import worker as implementation
from experiments.guarded_selection_20261026 import selector
module=importlib.util.spec_from_file_location('_unit_composition_supervisor',HERE.parent/'typed_coverage_20261026/study.py')
common=importlib.util.module_from_spec(module);module.loader.exec_module(common)
worker=implementation.previous
read,write,sha=worker.read,worker.write,worker.sha
METHODS=implementation.ARMS
LIMIT=3480
common.HERE=HERE;common.METHODS=METHODS;common.LIMIT=LIMIT;common.legacy.HERE=HERE
matrix,budget,name,verify_files=common.matrix,common.budget,common.name,common.verify_files
train,evaluate,mechanism=common.train,common.evaluate,common.mechanism


def freeze():
    if (HERE/'FREEZE.json').exists() or (HERE/'BUDGET.json').exists():raise RuntimeError('Already started.')
    calibration=read(HERE/'calibration/SUMMARY.json')
    if not all(r['status']=='complete' and r['mechanism_pass'] for r in calibration['rows']):raise RuntimeError('Calibration incomplete.')
    b=dict(start_unix=time.time(),limit_seconds=LIMIT,sequential_workers=1,threads=1,system_cap_seconds=60,verification_reserve_seconds=600,
        python=sys.version,platform=platform.platform(),packages={n:importlib.metadata.version(n) for n in ('numpy','scipy','torch','sympy','scikit-learn','psutil')})
    write(HERE/'BUDGET.json',b)
    files=list((ROOT/'src').rglob('*.py'))+list(HERE.glob('*.py'))+list((ROOT/'tests').glob('*.py'))
    for directory in ('first_round','second_round_20261026','guarded_selection_20261026','shared_discovery_20261026','typed_coverage_20261026'):
        files+=list((HERE.parent/directory).glob('*.py'))
    files += [HERE/n for n in ('PROTOCOL.md','PREFLIGHT.json','BUDGET.json','PROVENANCE.json','calibration/PROTOCOL.json','calibration/SUMMARY.json','calibration/VERIFICATION.json')]
    files += [ROOT/n for n in ('pyproject.toml','MANIFEST.in','docs/environment.txt','experiments/first_round/FREEZE.json','experiments/typed_coverage_20261026/FREEZE.json')]
    write(HERE/'FREEZE.json',dict(jobs=matrix(),config=read(HERE.parent/'first_round/FREEZE.json')['config'],policy=selector.POLICY,
        specs={c:cases.design(c) for c in cases.CASES},budget=b,frozen_unix=time.time(),test_labels_accessed=False,
        files={p.relative_to(ROOT).as_posix():sha(p) for p in files},base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
        evidence_scope='Known development regression; previous test scores already observed.'))


def run():
    train();evaluate()
    from experiments.unit_composition_20261026.verify import verify
    verify()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('freeze','train','evaluate','run'));globals()[p.parse_args().command]()
