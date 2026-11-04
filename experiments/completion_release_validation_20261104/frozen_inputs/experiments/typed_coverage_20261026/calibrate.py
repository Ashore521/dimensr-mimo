"""Four bounded artificial-data fits: runtime coverage, no test scoring."""
import argparse
from dataclasses import asdict
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import psutil
import torch
from dimensr_mimo import DimenSRMIMORegressor,SearchConfig
from experiments.typed_coverage_20261026.observe import Observation
from experiments.typed_coverage_20261026 import audit


def write(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')
def read(path):return json.loads(path.read_text())


def cell(folder,p,balanced):
    torch.set_num_threads(1)
    cfg=SearchConfig(**read(HERE.parent/'first_round/FREEZE.json')['config']);cfg.random_state=991;cfg.wall_seconds=55
    zero=[0]*7;spec=dict(features=[zero]*p,targets=[zero]*2,constants=[zero],bounds=[[-.8,.8]]*p)
    rng=np.random.default_rng(991);X=rng.uniform(-.8,.8,(96,p))
    Y=np.column_stack((X.sum(axis=1)/np.sqrt(p),X[:,0]-X[:,-1]))
    with Observation(folder/'trace.jsonl',cfg,spec,balanced) as observed:
        model=DimenSRMIMORegressor(cfg,spec['features'],spec['targets'],spec['constants']).fit(
            X[:72],Y[:72],X_val=X[72:],Y_val=Y[72:],input_bounds=spec['bounds'],callback=observed.event)
    model.save(folder/'model.json');loaded=DimenSRMIMORegressor.load(folder/'model.json')
    np.testing.assert_array_equal(model.predict(X),loaded.predict(X))
    units=audit.units(model.best_.system,spec['features'],spec['targets'],spec['constants'])
    events=model.history_
    mechanism=dict(neural_parameter_change=sum(e.get('parameter_delta',0) for e in events),
        valid_neural=sum(e.get('neural_valid',0) for e in events),gp_evaluations=sum(e.get('new_evaluations',0) for e in events if e['phase']=='gp'),
        nsga_generations=sum(e['phase']=='nsga2' for e in events),nsga_evaluations=sum(e.get('new_evaluations',0) for e in events if e['phase']=='nsga2'))
    write(folder/'result.json',dict(status='complete',inputs=p,balanced=balanced,config=asdict(cfg),observation=observed.summary(),
        units=units,complexity=audit.complexity(model.best_.system),mechanism=mechanism,reload_equal=True,test_scores_computed=False))


def run():
    out=HERE/'calibration'
    if out.exists():raise RuntimeError('Calibration already exists; no automatic retries.')
    out.mkdir();start=time.time();deadline=start+360
    files=list((ROOT/'src').rglob('*.py'))+list(HERE.glob('*.py'))+[HERE.parent/'shared_discovery_20261026/seeds.py',HERE.parent/'first_round/FREEZE.json']
    write(out/'PROTOCOL.json',dict(start_unix=start,limit_seconds=360,system_cap_seconds=60,jobs=[dict(inputs=p,balanced=b) for p in (8,11) for b in (False,True)],
        source_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        purpose='Artificial affine targets; actual initialization coverage, unit checks and resource completion only; no test data or performance selection.'))
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUTF8='1');rows=[]
    for p in (8,11):
        for balanced in (False,True):
            folder=out/f'p{p}_{"balanced" if balanced else "prefix"}';folder.mkdir();began=time.monotonic();peak=0
            with (folder/'stdout.log').open('w',encoding='utf-8') as log:
                proc=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'cell','--folder',str(folder),'--inputs',str(p),'--balanced',str(int(balanced))],stdout=log,stderr=subprocess.STDOUT,env=env)
                reason=None
                while proc.poll() is None:
                    try:
                        parent=psutil.Process(proc.pid);tree=[parent]+parent.children(recursive=True)
                        peak=max(peak,sum(c.memory_info().rss for c in tree if c.is_running()))
                        if time.monotonic()-began>=60 or time.time()>=deadline:
                            reason='timeout'
                            for child in reversed(tree):
                                try:child.kill()
                                except psutil.NoSuchProcess:pass
                            break
                    except psutil.NoSuchProcess:pass
                    time.sleep(.2)
                proc.wait()
            resource=dict(inputs=p,balanced=balanced,status=reason or ('complete' if proc.returncode==0 and (folder/'result.json').exists() else 'failed'),wall_seconds=time.monotonic()-began,peak_rss_bytes=peak,returncode=proc.returncode)
            write(folder/'resources.json',resource);rows.append(resource);print(json.dumps(resource),flush=True)
    write(out/'SUMMARY.json',dict(rows=rows,total_seconds=time.time()-start,test_scores_computed=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('run','cell'));p.add_argument('--folder');p.add_argument('--inputs',type=int);p.add_argument('--balanced',type=int)
    args=p.parse_args()
    run() if args.command=='run' else cell(Path(args.folder),args.inputs,bool(args.balanced))
