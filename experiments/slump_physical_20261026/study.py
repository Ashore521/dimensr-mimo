"""Sequential three-output supervisor with explicit 20-second output deadlines."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import psutil

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from dimensr_mimo import AffineCoordinates,SearchConfig,audit
from dimensr_mimo.graph import System,Vertex
from dimensr_mimo.physical import restore_graph
from dimensr_mimo._single.units import unit
from experiments.slump_physical_20261026 import data,worker
from experiments.native_api_20261026.run import arrays_hash
read,write,sha=data.read,data.write,data.sha


def name(job):return f"slump_s{job['seed']}_{job['method']}"
def matrix():return [dict(case='concrete_slump',seed=s,method=m,seconds=60) for s in (801,802,803) for m in ('legacy','dimensional','independent')]
def budget():return read(HERE/'BUDGET.json')['start_unix']+1200


def configuration():
    f=read(HERE.parent/'noise_families_20261026/FREEZE.json');return f['config']


def check_files(files,root=ROOT):
    for n,h in files.items():
        p=(root/n).resolve()
        if not p.is_relative_to(root.resolve()) or sha(p)!=h:raise ValueError('Frozen file drift: '+n)


def preflight():
    if (HERE/'PREFLIGHT.json').exists():raise RuntimeError('Preflight already exists.')
    m=read(HERE/'DATA_MANIFEST.json');X,Y=data.load('train');V,Z=data.load('val');coords=AffineCoordinates.from_training(X,Y)
    assert coords.to_dict()==m['coordinates'];bounds=np.asarray(data.BOUNDS)
    assert all(np.all(A>=bounds[:,0]) and np.all(A<=bounds[:,1]) for A in (X,V))
    for A in (Y,Z):np.testing.assert_allclose(coords.restore_outputs(coords.outputs(A)),A,rtol=1e-14,atol=1e-14)
    catalog=worker.constant_units();targets=list(map(unit,m['target_units']))
    nodes=tuple(Vertex('c'+str(next(i for i,c in enumerate(catalog) if unit(c)==u)),units=u,parameter=j) for j,u in enumerate(targets))
    system=System(nodes,(0,1,2));physical,values,units,roles=restore_graph(system,[0.,0.,0.],m['feature_units'],m['target_units'],catalog,coords)
    audit.units(physical,m['feature_units'],m['target_units'],units)
    np.testing.assert_array_equal(physical.evaluate(X,values),np.broadcast_to(coords.y_offset,Y.shape))
    assert all(a['status']=='enclosed' for a in physical.audit(values,data.BOUNDS))
    SearchConfig(**configuration()).validate()
    write(HERE/'PREFLIGHT.json',dict(status='passed',test_labels_accessed=False,rows={'train':len(X),'val':len(V)},
        data_sha256=arrays_hash(X,Y,V,Z),constant_units=catalog,coordinate_roundtrip='allclose 1e-14',
        raw_physical_constant_graph_checked=True,exact_input_duplicates=m['exact_input_duplicates']))


def freeze():
    if (HERE/'FREEZE.json').exists():raise RuntimeError('Already frozen.')
    assert read(HERE/'PREFLIGHT.json')['status']=='passed'
    files=list((ROOT/'src').rglob('*.py'))+list(HERE.glob('*.py'))
    for directory in ('first_round','second_round_20261026','typed_coverage_20261026','native_api_20261026'):
        files+=list((HERE.parent/directory).glob('*.py'))
    files += [HERE/n for n in ('PROTOCOL.md','DATA_MANIFEST.json','PREFLIGHT.json')]
    files += [HERE.parent/'noise_families_20261026/FREEZE.json',ROOT/'pyproject.toml']
    for p in (ROOT/'src').rglob('*.py'):
        dest=HERE/'source'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    b=dict(start_unix=time.time(),limit_seconds=1200,system_seconds=60,independent_output_seconds=20,verification_reserve_seconds=600,
        workers=1,threads=1,versions={n:importlib.metadata.version(n) for n in ('numpy','scipy','torch','sympy','scikit-learn','psutil')})
    write(HERE/'BUDGET.json',b)
    write(HERE/'FREEZE.json',dict(frozen_unix=time.time(),jobs=matrix(),config=configuration(),budget=b,
        search_limits=worker.SEARCH_LIMITS,physical_limits=worker.PHYSICAL_LIMITS,constant_units=worker.constant_units(),
        files={p.relative_to(ROOT).as_posix():sha(p) for p in files},test_labels_accessed=False,
        data_sha256=read(HERE/'PREFLIGHT.json')['data_sha256'],
        base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()))


def run_job(job,config):
    folder=HERE/'runs'/name(job);folder.mkdir(parents=True);write(folder/'job.json',job);write(folder/'config.json',config)
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUTF8='1')
    started=time.monotonic();peak=0;reason=None;cpu={};returncode=None
    with (folder/'stdout.log').open('w',encoding='utf-8') as log:
        p=subprocess.Popen([sys.executable,'-B',str(HERE/'worker.py'),str(folder),str(folder/'job.json'),str(folder/'config.json')],stdout=log,stderr=subprocess.STDOUT,env=env)
        while p.poll() is None:
            try:
                parent=psutil.Process(p.pid);tree=[parent]+parent.children(recursive=True);rss=0
                for child in tree:
                    try:
                        rss+=child.memory_info().rss;t=child.cpu_times();cpu[child.pid]=t.user+t.system
                    except psutil.NoSuchProcess:pass
                peak=max(peak,rss)
                if job['method']=='independent' and (folder/'events.jsonl').exists():
                    deadline=None
                    for line in (folder/'events.jsonl').read_text(encoding='utf-8').splitlines():
                        try:row=json.loads(line)
                        except json.JSONDecodeError:continue
                        if row['phase']=='output_start':deadline=row['monotonic']+row['allowance_seconds']
                        if row['phase']=='output_complete':deadline=None
                    if deadline is not None and time.monotonic()>=deadline:reason='per_output_timeout'
                if time.monotonic()-started>=60:reason='system_timeout'
                if time.time()>=budget()-600:reason='global_fit_budget'
                if reason:
                    for child in reversed(tree):
                        try:child.kill()
                        except psutil.NoSuchProcess:pass
                    break
            except psutil.NoSuchProcess:pass
            time.sleep(.2)
        returncode=p.wait()
    result=read(folder/'result.json') if (folder/'result.json').exists() else {}
    resource=dict(status=reason or result.get('status','worker_failed'),job=job,returncode=returncode,wall_seconds=time.monotonic()-started,
        peak_rss_bytes=peak,process_tree_sampled_cpu_seconds=sum(cpu.values()))
    write(folder/'resources.json',resource);print(name(job),resource['status'],round(resource['wall_seconds'],2),flush=True)


def train():
    f=read(HERE/'FREEZE.json');check_files(f['files'])
    if (HERE/'ALL_MODELS_FROZEN.json').exists():check_files(read(HERE/'ALL_MODELS_FROZEN.json')['files'],HERE);return
    if not (HERE/'TRAINING_STARTED.json').exists():write(HERE/'TRAINING_STARTED.json',dict(start_unix=time.time()))
    for job in f['jobs']:
        folder=HERE/'runs'/name(job);check_files(f['files'])
        if (folder/'CELL_FROZEN.json').exists():check_files(read(folder/'CELL_FROZEN.json')['files'],folder);continue
        if folder.exists():raise RuntimeError('Unsealed cell requires incident review.')
        if time.time()+60>budget()-600:
            folder.mkdir(parents=True);write(folder/'job.json',job);write(folder/'resources.json',dict(status='not_started_budget',job=job))
        else:run_job(job,f['config'])
        write(folder/'CELL_FROZEN.json',dict(frozen_unix=time.time(),files={p.name:sha(p) for p in folder.iterdir() if p.is_file() and p.name!='CELL_FROZEN.json'}))
    write(HERE/'ALL_MODELS_FROZEN.json',dict(frozen_unix=time.time(),freeze_sha256=sha(HERE/'FREEZE.json'),test_labels_accessed=False,
        files={p.relative_to(HERE).as_posix():sha(p) for p in (HERE/'runs').rglob('*') if p.is_file()}))


def mechanism(folder,result):
    events=[json.loads(line) for line in (folder/'events.jsonl').read_text(encoding='utf-8').splitlines()]
    single=result['job']['method']=='independent';reports=[]
    for j in (range(3) if single else [None]):
        rows=[r for r in events if r['output']==j];d=result['diagnostics'][j if single else 0]
        evidence=dict(output=j,policy_delta=sum(r.get('parameter_delta',0) for r in rows),neural_valid=sum(r.get('neural_valid',0) for r in rows),
            gp_new_evaluations=d['instrumentation'].get('evaluation_gp_new',0) if single else sum(r['new_evaluations'] for r in rows if r['phase']=='gp'),
            nsga2_generations=sum(r['phase']=='nsga2' for r in rows),nsga2_new_evaluations=sum(r.get('new_evaluations',0) for r in rows if r['phase']=='nsga2'),
            rademacher_calls=d['instrumentation'].get('local_rademacher_calls',0) if single else d.get('rademacher_systems',0))
        evidence['passed']=all(v>0 for k,v in evidence.items() if k!='output') and evidence['nsga2_generations']>=2
        reports.append(evidence)
    return dict(passed=all(r['passed'] for r in reports),outputs=reports)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('preflight','freeze','train'));globals()[p.parse_args().command]()
