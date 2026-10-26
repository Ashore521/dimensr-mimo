"""Artificial-response, no-test runtime calibration for the combined intervention."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import psutil
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from dimensr_mimo import DimenSRMIMORegressor,SearchConfig
from experiments.typed_coverage_20261026.observe import Observation
from experiments.unit_composition_20261026.worker import intervention,read,write,sha


def cell(folder,p,method):
    torch.set_num_threads(1)
    cfg=SearchConfig(**read(HERE.parent/'first_round/FREEZE.json')['config']);cfg.random_state=992;cfg.wall_seconds=52
    L=[1,0,0,0,0,0,0];V=[1,0,-1,0,0,0,0];K=[-1,0,0,0,0,0,0];zero=[0]*7
    spec=dict(features=[L]*p,targets=[L,V] if p==2 else [zero,zero],constants=[zero,K,L,V],bounds=[[-.8,.8]]*p)
    X=np.random.default_rng(992).uniform(-.8,.8,(96,p));Y=np.column_stack((X.sum(axis=1)/np.sqrt(p),X[:,0]-X[:,-1]))
    with intervention(method,spec),Observation(folder/'trace.jsonl',cfg,spec,True) as observed:
        m=DimenSRMIMORegressor(cfg,spec['features'],spec['targets'],spec['constants']).fit(X[:72],Y[:72],X_val=X[72:],Y_val=Y[72:],input_bounds=spec['bounds'],callback=observed.event)
    m.save(folder/'model.json');loaded=DimenSRMIMORegressor.load(folder/'model.json');np.testing.assert_array_equal(m.predict(X),loaded.predict(X))
    evidence=dict(policy_delta=sum(e.get('parameter_delta',0) for e in m.history_),neural_valid=sum(e.get('neural_valid',0) for e in m.history_),
        gp_new=sum(e.get('new_evaluations',0) for e in m.history_ if e['phase']=='gp'),nsga_generations=sum(e['phase']=='nsga2' for e in m.history_),
        nsga_new=sum(e.get('new_evaluations',0) for e in m.history_ if e['phase']=='nsga2'))
    write(folder/'result.json',dict(status='complete',spec=spec,method=method,observation=observed.summary(),mechanism=evidence,
        mechanism_pass=all(v>0 for v in evidence.values()) and evidence['nsga_generations']>=2,reload_equal=True,test_scores_computed=False))


def run():
    out=HERE/'calibration'
    if out.exists():raise RuntimeError('Calibration already exists; never auto-retry.')
    out.mkdir();start=time.time();files=list((ROOT/'src').rglob('*.py'))+list(HERE.glob('*.py'))+list((HERE.parent/'typed_coverage_20261026').glob('*.py'))
    files+=[HERE.parent/'shared_discovery_20261026/seeds.py',HERE.parent/'first_round/FREEZE.json']
    write(out/'PROTOCOL.json',dict(start_unix=start,limit_seconds=300,cell_cap_seconds=60,source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in files},
        jobs=[dict(inputs=p,method=m) for p in (2,11) for m in ('balanced','combined')],test_scores_computed=False))
    rows=[];env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUTF8='1')
    for p in (2,11):
        for method in ('balanced','combined'):
            folder=out/f'p{p}_{method}';folder.mkdir();began=time.monotonic();peak=0;reason=None
            with (folder/'stdout.log').open('w',encoding='utf-8') as log:
                proc=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'cell','--folder',str(folder),'--inputs',str(p),'--method',method],stdout=log,stderr=subprocess.STDOUT,env=env)
                while proc.poll() is None:
                    try:
                        parent=psutil.Process(proc.pid);tree=[parent]+parent.children(recursive=True);peak=max(peak,sum(x.memory_info().rss for x in tree if x.is_running()))
                        if time.monotonic()-began>=60 or time.time()-start>=300:
                            reason='timeout'
                            for child in reversed(tree):
                                try:child.kill()
                                except psutil.NoSuchProcess:pass
                            break
                    except psutil.NoSuchProcess:pass
                    time.sleep(.2)
                proc.wait()
            result=read(folder/'result.json') if (folder/'result.json').exists() else {}
            row=dict(inputs=p,method=method,status=reason or ('complete' if proc.returncode==0 and result.get('status')=='complete' else 'failed'),
                mechanism_pass=result.get('mechanism_pass',False),wall_seconds=time.monotonic()-began,peak_rss_bytes=peak,returncode=proc.returncode)
            write(folder/'resources.json',row);rows.append(row);print(row,flush=True)
    write(out/'SUMMARY.json',dict(rows=rows,wall_seconds=time.time()-start,test_scores_computed=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('run','cell'));p.add_argument('--folder');p.add_argument('--inputs',type=int);p.add_argument('--method');a=p.parse_args()
    run() if a.command=='run' else cell(Path(a.folder),a.inputs,a.method)
