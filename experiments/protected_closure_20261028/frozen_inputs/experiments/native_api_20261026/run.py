"""Frozen bounded API equivalence and physical-coordinate integration checks."""
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
import psutil
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'src'))
from dimensr_mimo import DimenSRMIMORegressor,SearchConfig
from experiments.typed_coverage_20261026 import cases
from examples import physical_coordinates as example

JOBS=[('products','legacy'),('dimensional_frequency','legacy'),('dimensional_frequency','normalized'),
      ('partial','normalized'),('high_dimensional','normalized'),('affine_lengths','physical')]


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def arrays_hash(*arrays):return hashlib.sha256(b''.join(np.asarray(a,dtype='<f8').tobytes() for a in arrays)).hexdigest()


def config(case,method):
    if method=='physical':return example.configuration()
    cfg=SearchConfig(**read(HERE.parent/'first_round/FREEZE.json')['config']);cfg.random_state=601;cfg.wall_seconds=52
    if method=='normalized':
        cfg.initialization='dimensional';cfg.final_selection='precision';cfg.acceptance=cases.design(case)['acceptance']
    return cfg


def reference_check():
    rows=[]
    for case,method in JOBS:
        if method!='normalized':continue
        baseline=read(HERE/'baseline_corrected'/f'{case}_{method}/model.json')
        prior=read(HERE.parent/'unit_composition_20261026/runs'/f'{case}_s601_normalized/searched_model.json')
        checks={k:baseline[k]==prior[k] for k in ('search_fingerprint','candidates')}
        checks.update({k:baseline['diagnostics'][k]==prior['diagnostics'][k] for k in ('frozen_candidate_sha256','operations','changed','novel','successful')})
        rows.append(dict(case=case,checks=checks,passed=all(checks.values())))
    return rows


def cell(folder,case,method):
    torch.set_num_threads(1)
    def event(row):
        with (folder/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
    try:
        if method=='physical':
            m,physical=example.fit_example(folder,event);X,Y,V,Z=example.data()
            predictions={name:arrays_hash(physical.predict(a)) for name,a in [('train',X),('validation',V)]}
        else:
            spec=cases.design(case);X,Y=cases.training_data(case,'train',601);V,Z=cases.training_data(case,'validation',601)
            m=DimenSRMIMORegressor(config(case,method),spec['features'],spec['targets'],spec['constants']).fit(
                X,Y,X_val=V,Y_val=Z,input_bounds=spec['bounds'],callback=event)
            m.save(folder/'model.json');predictions={name:arrays_hash(m.predict(a)) for name,a in [('train',X),('validation',V)]}
        write(folder/'result.json',dict(status='complete',case=case,method=method,search_fingerprint=m.search_fingerprint_,
            pool_sha256=m.diagnostics_['frozen_candidate_sha256'],prediction_hashes=predictions,
            deadline_reached=m.diagnostics_['deadline_reached'],test_scores_computed=False))
    except Exception as e:
        (folder/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
        write(folder/'result.json',dict(status='failed',case=case,method=method,error=repr(e),test_scores_computed=False))
        raise


def run():
    if (HERE/'FREEZE.json').exists():raise RuntimeError('Frozen batch already exists; retain failures.')
    references=reference_check();write(HERE/'REFERENCE_VERIFICATION.json',references)
    if not all(r['passed'] for r in references):raise RuntimeError('Corrected reference differs from published run.')
    copied={}
    for p in sorted((ROOT/'src').rglob('*.py')):
        dest=HERE/'source'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
        copied[p.relative_to(ROOT).as_posix()]=sha(dest)
    jobs=[]
    for case,method in JOBS:
        arrays=example.data() if method=='physical' else (*cases.training_data(case,'train',601),*cases.training_data(case,'validation',601))
        jobs.append(dict(case=case,method=method,config=asdict(config(case,method)),data_sha256=arrays_hash(*arrays)))
    files=[Path(__file__),HERE/'verify.py',ROOT/'examples/physical_coordinates.py',ROOT/'pyproject.toml',
        HERE.parent/'typed_coverage_20261026/cases.py',HERE.parent/'precision_delivery_20261026/selector.py',
        HERE.parent/'guarded_selection_20261026/selector.py',HERE/'PROTOCOL.md',HERE/'PROVENANCE.json']
    files.extend(HERE/('baseline' if method=='legacy' else 'baseline_corrected')/f'{case}_{method}/model.json'
                 for case,method in JOBS if method!='physical')
    write(HERE/'FREEZE.json',dict(frozen_unix=time.time(),jobs=jobs,source_sha256=copied,
        files={p.relative_to(ROOT).as_posix():sha(p) for p in files},cell_cap_seconds=60,total_cap_seconds=400,
        physical_limits=example.PHYSICAL_LIMITS,thread_count=1,workers=1,test_scores_computed=False,
        versions={p:importlib.metadata.version(p) for p in ('numpy','scipy','torch','sympy','scikit-learn','psutil')},
        base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()))
    start=time.monotonic();rows=[];env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUTF8='1')
    for case,method in JOBS:
        folder=HERE/'runs'/f'{case}_{method}';folder.mkdir(parents=True);began=time.monotonic();reason=None;peak=0
        with (folder/'stdout.log').open('w',encoding='utf-8') as log:
            p=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'cell',str(folder),case,method],stdout=log,stderr=subprocess.STDOUT,env=env)
            while p.poll() is None:
                try:
                    parent=psutil.Process(p.pid);tree=[parent]+parent.children(recursive=True)
                    peak=max(peak,sum(x.memory_info().rss for x in tree if x.is_running()))
                    if time.monotonic()-began>=60 or time.monotonic()-start>=400:
                        reason='timeout'
                        for child in reversed(tree):
                            try:child.kill()
                            except psutil.NoSuchProcess:pass
                        break
                except psutil.NoSuchProcess:pass
                time.sleep(.2)
            p.wait()
        status=reason or (read(folder/'result.json')['status'] if (folder/'result.json').exists() else 'failed')
        row=dict(case=case,method=method,status=status,wall_seconds=time.monotonic()-began,peak_rss_bytes=peak,returncode=p.returncode)
        write(folder/'resources.json',row);rows.append(row);print(row,flush=True)
    write(HERE/'RUN_SUMMARY.json',dict(rows=rows,wall_seconds=time.monotonic()-start))
    files={p.relative_to(HERE).as_posix():sha(p) for p in sorted((HERE/'runs').rglob('*')) if p.is_file()}
    write(HERE/'ALL_MODELS_FROZEN.json',dict(frozen_unix=time.time(),freeze_sha256=sha(HERE/'FREEZE.json'),files=files,test_scores_computed=False))


if __name__=='__main__':
    run() if len(sys.argv)==1 else cell(Path(sys.argv[2]),sys.argv[3],sys.argv[4])
