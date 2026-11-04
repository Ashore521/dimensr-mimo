"""Pre-integration fingerprints on known development inputs; no test scoring."""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import psutil
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(HERE/'baseline/source/src'))
from dimensr_mimo import DimenSRMIMORegressor,SearchConfig
from experiments.typed_coverage_20261026 import cases
from experiments.typed_coverage_20261026.worker import read,write,sha,prediction_hashes
from experiments.unit_composition_20261026.worker import intervention
from experiments.typed_coverage_20261026.observe import Observation

JOBS=[('dimensional_frequency','normalized'),('partial','normalized'),('high_dimensional','normalized')]


def cell(folder,case,method):
    torch.set_num_threads(1);spec=cases.design(case)
    X,Y=cases.training_data(case,'train',601);V,Z=cases.training_data(case,'validation',601)
    cfg=SearchConfig(**read(HERE.parent/'first_round/FREEZE.json')['config']);cfg.random_state=601;cfg.wall_seconds=52
    def fit(callback=None):return DimenSRMIMORegressor(cfg,spec['features'],spec['targets'],spec['constants']).fit(X,Y,X_val=V,Y_val=Z,input_bounds=spec['bounds'],callback=callback)
    if method=='legacy':m=fit()
    else:
        with intervention('normalized',spec),Observation(folder/'trace.jsonl',cfg,spec,True) as observed:m=fit(observed.event)
    m.save(folder/'model.json')
    write(folder/'result.json',dict(case=case,method=method,config=asdict(cfg),search_fingerprint=m.search_fingerprint_,
        pool_sha256=m.diagnostics_['frozen_candidate_sha256'],prediction_hashes=prediction_hashes(m,X,V),
        deadline_reached=m.diagnostics_['deadline_reached'],test_scores_computed=False))


def run():
    out=HERE/'baseline_corrected'
    if out.exists():raise RuntimeError('Baseline already exists.')
    out.mkdir(parents=True);source_root=HERE/'baseline/source';files=list((source_root/'src').rglob('*.py'));copies={}
    for p in files:
        dest=out/'source'/p.relative_to(source_root);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest);copies[p.relative_to(source_root).as_posix()]=sha(dest)
    write(out/'PROTOCOL.json',dict(jobs=JOBS,cell_cap_seconds=60,total_limit_seconds=320,source_sha256=copies,source_origin='baseline/source/src frozen before native API edits',
        base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
        script_sha256=sha(Path(__file__)),test_scores_computed=False))
    start=time.monotonic();rows=[];env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUTF8='1')
    for case,method in JOBS:
        folder=out/f'{case}_{method}';folder.mkdir();began=time.monotonic();reason=None;peak=0
        with (folder/'stdout.log').open('w',encoding='utf-8') as log:
            p=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'cell',str(folder),case,method],stdout=log,stderr=subprocess.STDOUT,env=env)
            while p.poll() is None:
                try:
                    parent=psutil.Process(p.pid);tree=[parent]+parent.children(recursive=True);peak=max(peak,sum(x.memory_info().rss for x in tree if x.is_running()))
                    if time.monotonic()-began>=60 or time.monotonic()-start>=320:
                        reason='timeout'
                        for child in reversed(tree):
                            try:child.kill()
                            except psutil.NoSuchProcess:pass
                        break
                except psutil.NoSuchProcess:pass
                time.sleep(.2)
            p.wait()
        row=dict(case=case,method=method,status=reason or ('complete' if p.returncode==0 and (folder/'result.json').exists() else 'failed'),wall_seconds=time.monotonic()-began,peak_rss_bytes=peak)
        write(folder/'resources.json',row);rows.append(row);print(row,flush=True)
    write(out/'SUMMARY.json',dict(rows=rows,wall_seconds=time.monotonic()-start))


if __name__=='__main__':
    run() if len(sys.argv)==1 else cell(Path(sys.argv[2]),sys.argv[3],sys.argv[4])
