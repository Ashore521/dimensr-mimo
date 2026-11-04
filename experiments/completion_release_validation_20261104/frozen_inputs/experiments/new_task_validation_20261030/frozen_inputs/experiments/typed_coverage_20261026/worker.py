"""One bounded dimensional search; train and validation only."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from dimensr_mimo import DimenSRMIMORegressor,SearchConfig
from dimensr_mimo._single import DimenSRRegressor
from experiments.second_round_20261026.cases import previous
from experiments.guarded_selection_20261026 import selector
from experiments.typed_coverage_20261026 import cases,delivery
from experiments.typed_coverage_20261026.observe import Observation

base=previous('worker',cases=cases)
base.generate=cases.training_data
write=base.write
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def prediction_hashes(m,X,V):
    return {n:hashlib.sha256(m.predict(A).astype('<f8').tobytes()).hexdigest() for n,A in [('train',X),('validation',V)]}


def run(folder,job,config):
    if job['method'] not in ('prefix','balanced','independent') or (folder/'result.json').exists():
        raise RuntimeError('Unknown arm or existing result.')
    torch.set_num_threads(1);started=time.monotonic();cpu=time.process_time()
    result=dict(job=job,status='failed',models=[],diagnostics=[],prediction_hashes=[],audits=[])
    try:
        X,Y=cases.training_data(job['case'],'train',job['seed']);V,Z=cases.training_data(job['case'],'validation',job['seed'])
        spec=cases.design(job['case']);single=job['method']=='independent'
        if single:
            base.run(folder,job,config);result=read(folder/'result.json')
            if result['status']!='complete':return
        else:
            cfg=SearchConfig(**config);cfg.random_state=job['seed'];cfg.wall_seconds=job['seconds']-8
            with Observation(folder/'trace.jsonl',cfg,spec,job['method']=='balanced') as observed:
                def event(e):
                    observed.event(e)
                    with (folder/'events.jsonl').open('a',encoding='utf-8') as f:
                        f.write(json.dumps(dict(e,output=None,worker_elapsed=time.monotonic()-started,monotonic=time.monotonic()),allow_nan=False)+'\n')
                m=DimenSRMIMORegressor(cfg,spec['features'],spec['targets'],spec['constants']).fit(
                    X,Y,X_val=V,Y_val=Z,input_bounds=spec['bounds'],callback=event)
            m.save(folder/'searched_model.json');source=read(folder/'searched_model.json')
            write(folder/'model.json',selector.derive(source,'guarded'))
            write(folder/'pool.json',delivery.pool(source,spec))
            result.update(models=['model.json'],observation=observed.summary(),effective_config=asdict(cfg),search_fingerprint=m.search_fingerprint_)
        cls=DimenSRRegressor if single else DimenSRMIMORegressor
        models=[cls.load(folder/n) for n in result['models']]
        result['prediction_hashes']=[prediction_hashes(m,X,V) for m in models]
        result['audits']=[m.audit_domain(input_bounds=spec['bounds']) for m in models]
        if not single:result['diagnostics']=[models[0].diagnostics_]
        system,constants=delivery.graph(models,single)
        bundle=delivery.record(system,constants,spec,X,Y,V,Z)
        expected=np.column_stack([m.predict(X) for m in models]) if single else models[0].predict(X)
        np.testing.assert_array_equal(expected,system.evaluate(X,constants))
        bundle['formulas']=[str(m.expression_) for m in models] if single else models[0].formulas_
        bundle['domain_audits']=result['audits'];bundle['sharing']=system.sharing
        write(folder/'delivery.json',bundle)
        result.update(status='complete',reload_graph_predictions_equal=True,test_scores_computed=False,
            development_accepted=bundle['development_accepted'])
    except Exception as e:
        result.update(status='failed',error=repr(e));(folder/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
    result.update(worker_wall_seconds=time.monotonic()-started,worker_cpu_seconds=time.process_time()-cpu)
    write(folder/'result.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder');p.add_argument('job');p.add_argument('config');a=p.parse_args()
    run(Path(a.folder),read(Path(a.job)),read(Path(a.config)))
