"""Fit all models and freeze choices before any external test arrays are loaded."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from dataclasses import asdict,replace
from datetime import datetime,timezone
import gzip
import importlib.metadata
import json
import os
import subprocess
import sys
import time

import joblib
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from dimensr_mimo import SearchConfig,MIMOSymbolicRegressor
from dimensr_mimo._vendor.units import ZERO
from common import HERE,ROOT,SEALED,DATASETS,METHODS,SYMBOLIC,THREAD_ENV,transfer,shared,v3,sha,save,jobs,name,check_freeze,baseline_predict

BASE_SECONDS,CONTROL_SECONDS,EXTRA_SECONDS=20.,25.,5.

def development(dataset):
    meta=json.loads((HERE/'data/manifest.json').read_text())[dataset]
    with np.load(HERE/'data'/f'{dataset}.npz',allow_pickle=False) as arrays:
        split={k:arrays[k] for k in ('X_train','Y_train','X_val','Y_val')}
    return split,meta

def fit_job(job,out):
    check_freeze();dataset,seed=job;split,meta=development(dataset)
    X,Y=split['X_train'],split['Y_train'];units=(ZERO,)*X.shape[1];targets=(ZERO,)*Y.shape[1]
    bounds=np.array(meta['bounds']);scales=np.array(meta['target_scale']);coefficients=(ZERO,)
    cfg=SearchConfig(**dict(v3.BASE_CONFIG,search_seconds=BASE_SECONDS,random_state=seed))
    cpu,wall=time.process_time(),time.monotonic()
    models,pools,diagnostics,choices,archives={}, {}, {}, {}, {}
    with threadpool_limits(limits=1):
        for label,seconds in (('independent20',BASE_SECONDS),('independent25',CONTROL_SECONDS)):
            pools[label],diagnostics[label]=transfer.independent(X,Y,bounds,units,targets,coefficients,replace(cfg,search_seconds=seconds))
        for rule in ('random','semantic'):
            label=rule+'25'
            pools[label],diagnostics[label]=transfer.transfer(pools['independent20'],X,Y,bounds,targets,coefficients,cfg,seconds=EXTRA_SECONDS,rule=rule)
        independent_cpu=time.process_time()-cpu
        joint=MIMOSymbolicRegressor(replace(cfg,search_seconds=CONTROL_SECONDS),feature_names=meta['feature_names'],target_names=meta['target_names'])
        start=time.process_time()
        joint.fit(X,Y,X_val=split['X_val'],Y_val=split['Y_val'],input_bounds=bounds)
        joint_fit_cpu=time.process_time()-start
        models['joint25']=joint;diagnostics['joint25']=joint.diagnostics_
        for label,pool in pools.items():
            start=time.process_time()
            archives[label]=shared.validate_archives(pool,split['X_val'],split['Y_val'],scales,bounds,targets,cfg.interval_max_boxes)
            indices=shared.select_indices(archives[label],'compact')
            candidate=shared.combine(archives[label],indices)
            models[label]=shared.inference_model(candidate,replace(cfg,search_seconds=BASE_SECONDS if label=='independent20' else CONTROL_SECONDS),
                bounds,scales,meta['feature_names'],meta['target_names'],units,targets,coefficients)
            search_cpu=diagnostics[label]['search_cpu_seconds']
            if label in ('random25','semantic25'):search_cpu+=diagnostics['independent20']['search_cpu_seconds']
            choices[label]=dict(indices=indices,training_cpu_seconds=search_cpu,selection_cpu_seconds=time.process_time()-start,
                selected_transfer_heads=sum(i>=len(archives['independent20'][j]) for j,i in enumerate(indices)) if label in ('random25','semantic25') else 0)
        choices['joint25']=dict(training_cpu_seconds=joint.diagnostics_['search_cpu_seconds'],
            fit_cpu_seconds=joint_fit_cpu,selected_transfer_heads=0)
        normalized=(Y-np.array(meta['target_mean']))/scales
        for label,estimator in (('mean',DummyRegressor(strategy='mean')),('ridge',Ridge(alpha=1.)),
            ('extra_trees',ExtraTreesRegressor(n_estimators=200,min_samples_leaf=2,n_jobs=1,random_state=seed))):
            start=time.process_time();estimator.fit(X,normalized)
            models[label]=dict(estimator=estimator,offset=np.array(meta['target_mean']),scale=scales)
            choices[label]=dict(training_cpu_seconds=time.process_time()-start,selected_transfer_heads=0)
        for label in METHODS:
            model=models[label];symbolic=label in SYMBOLIC
            file=label+('.json' if symbolic else '.joblib');path=out/file
            if symbolic:
                model.save(path);restored=MIMOSymbolicRegressor.load(path)
                prediction=model.predict(split['X_val'])
                np.testing.assert_array_equal(restored.predict(split['X_val']),prediction)
                audit=model.audit_domain()
                if audit['status']!='enclosed':raise ValueError('Selected model failed training-domain audit')
            else:
                joblib.dump(model,path,compress=3);restored=joblib.load(path)
                prediction=baseline_predict(model,split['X_val'])
                np.testing.assert_array_equal(baseline_predict(restored,split['X_val']),prediction);audit=None
            choices[label].update(model=file,model_sha256=sha(path),complexity=model.model_['complexity'] if symbolic else None,
                validation=v3.metrics(split['Y_val'],prediction,scales),training_domain=audit)
        with (out/'archives.json.gz').open('xb') as stream:
            stream.write(gzip.compress(json.dumps(archives,allow_nan=False).encode(),mtime=0))
        save(out/'CHOICES.json',choices)
    return dict(dataset=dataset,seed=seed,status='ok',config=asdict(cfg),choices=choices,diagnostics=diagnostics,
        independent_and_transfer_cpu_seconds=independent_cpu,total_cpu_seconds=time.process_time()-cpu,
        wall_seconds=time.monotonic()-wall,archives_sha256=sha(out/'archives.json.gz'),choices_sha256=sha(out/'CHOICES.json'))

def freeze():
    paths=list((ROOT/'src/dimensr_mimo').rglob('*.py'))+list(HERE.glob('*.py'))
    paths += [HERE/'PROTOCOL.md',HERE/'data/manifest.json',ROOT/'tests/test_external_transfer.py',
        ROOT/'experiments/semantic_transfer_v1/transfer.py',ROOT/'experiments/shared_private_v1/hybrid.py',
        ROOT/'experiments/budget_capacity_v1/curves.py',ROOT/'experiments/real_blind_v3/runner.py']
    paths += list((HERE/'data').glob('*.npz'))+list(SEALED.glob('*.npz'))
    save(HERE/'FREEZE.json',dict(files={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)},
        jobs=[name(j) for j in jobs()],created_at_utc=datetime.now(timezone.utc).isoformat(),
        algorithm_commit='01eefe18cfb46310aff39f834298a311f8eae653',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        packages={p:importlib.metadata.version(p) for p in ('numpy','scipy','sympy','scikit-learn','threadpoolctl','joblib')}))

def freeze_models():
    check_freeze();records={}
    for job in jobs():
        out=HERE/'runs'/name(job);r=json.loads((out/'result.json').read_text())
        if r['status']!='ok' or r['freeze_sha256']!=sha(HERE/'FREEZE.json'):raise ValueError('Incomplete or inconsistent training grid')
        assert set(r['choices'])==set(METHODS)
        assert sha(out/'CHOICES.json')==r['choices_sha256']
        assert sha(out/'archives.json.gz')==r['archives_sha256']
        for choice in r['choices'].values():assert sha(out/choice['model'])==choice['model_sha256']
        records[name(job)]=dict(result_sha256=sha(out/'result.json'),choices_sha256=r['choices_sha256'],models={m:c['model_sha256'] for m,c in r['choices'].items()})
    save(HERE/'MODELS_FROZEN.json',dict(freeze_sha256=sha(HERE/'FREEZE.json'),created_at_utc=datetime.now(timezone.utc).isoformat(),records=records))

def launch(job):
    out=HERE/'runs'/name(job)
    if out.exists():
        if not (out/'result.json').exists():raise FileExistsError(f'Incomplete run preserved: {out}')
        r=json.loads((out/'result.json').read_text())
        if r['freeze_sha256']!=sha(HERE/'FREEZE.json'):raise ValueError('Different freeze')
        return r['status']
    out.mkdir(parents=True)
    save(out/'started.json',dict(job=job,time_utc=datetime.now(timezone.utc).isoformat(),freeze_sha256=sha(HERE/'FREEZE.json')))
    with (out/'process.log').open('x',encoding='utf-8') as log:
        try:
            child=subprocess.run([sys.executable,str(HERE/'study.py'),'job',*map(str,job)],stdout=log,stderr=subprocess.STDOUT,
                timeout=300,env=dict(os.environ,**THREAD_ENV,PYTHONUTF8='1',PYTHONHASHSEED='0'))
            r=json.loads((out/'child.json').read_text()) if (out/'child.json').exists() else dict(status='failed',error='No child result')
            if child.returncode:r.update(status='failed',returncode=child.returncode)
        except subprocess.TimeoutExpired:r=dict(status='timeout',error='300-second wall timeout')
        except OSError as exc:r=dict(status='failed',error=str(exc))
    r.update(dataset=job[0],seed=job[1],freeze_sha256=sha(HERE/'FREEZE.json'));save(out/'result.json',r)
    return r['status']

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('freeze','batch','job'))
    p.add_argument('identity',nargs='*');p.add_argument('--workers',type=int,default=4);args=p.parse_args()
    if args.command=='freeze':freeze()
    elif args.command=='job':
        job=args.identity[0],int(args.identity[1]);assert job in jobs()
        try:r=fit_job(job,HERE/'runs'/name(job))
        except Exception as exc:
            import traceback;traceback.print_exc();r=dict(status='failed',error=f'{type(exc).__name__}: {exc}')
        save(HERE/'runs'/name(job)/'child.json',r);return int(r['status']!='ok')
    else:
        check_freeze()
        if args.workers<1:p.error('workers must be positive')
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            pending={pool.submit(launch,j):j for j in jobs()};statuses=[]
            for future in as_completed(pending):
                status=future.result();statuses.append(status);print(name(pending[future]),status,flush=True)
        print(f'Finished {len(statuses)} cells; failures={sum(s!="ok" for s in statuses)}',flush=True)
        if any(s!='ok' for s in statuses):return 1
        freeze_models();print('All 120 model choices frozen; no test arrays loaded.',flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
