"""Compare precision repair after the same complete compound MIMO search."""
from dataclasses import asdict
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from dimensr_mimo import DimenSRMIMORegressor, SearchConfig, PhysicalModel
from experiments.new_task_validation_20261030 import cases
from experiments.typed_coverage_20261026 import delivery
from experiments.native_api_20261026.run import read, write, arrays_hash


def run(folder,job,config):
    if job['method'] not in ('base','repair'):
        raise ValueError('Outside the frozen precision-repair methods.')
    from experiments.new_task_validation_20261030.study import check_candidate_lock
    check_candidate_lock()
    torch.set_num_threads(1);start=time.monotonic();cpu=time.process_time()
    result=dict(job=job,status='failed',models=[],diagnostics=[],prediction_hashes=[],audits=[],test_scores_computed=False)
    try:
        X,Y=cases.training_data(job['case'],'train',job['seed']);V,Z=cases.training_data(job['case'],'validation',job['seed'])
        spec=cases.design(job['case'])
        cfg=SearchConfig(**dict(config,random_state=job['seed'],initialization='compound',sharing_strategy='closure',precision_repair=job['method']=='repair',
            final_selection='precision',acceptance=spec['acceptance']))
        def event(row):
            import json
            with (folder/'events.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(dict(row,output=None,monotonic=time.monotonic()),allow_nan=False)+'\n')
        model=DimenSRMIMORegressor(cfg,spec['features'],spec['targets'],spec['constants'],spec['feature_names']).fit(
            X,Y,X_val=V,Y_val=Z,input_bounds=spec['bounds'],callback=event)
        model.save(folder/'model.json')
        result.update(models=['model.json'],diagnostics=[model.diagnostics_],effective_config=asdict(cfg),
            search_fingerprint=model.search_fingerprint_,application_acceptance=model.post_selection_)
        physical=PhysicalModel.from_estimator(model,input_bounds=spec['bounds'],physical_limits=spec['acceptance'],
            training_data=(X,Y),validation_data=(V,Z));physical.save(folder/'physical.json')
        result['prediction_hashes']=[{n:arrays_hash(model.predict(A)) for n,A in [('train',X),('validation',V)]}]
        result['audits']=[model.audit_domain(spec['bounds'])]
        models=[DimenSRMIMORegressor.load(folder/'model.json')]
        system,constants=delivery.graph(models,False)
        bundle=delivery.record(system,constants,spec,X,Y,V,Z)
        np.testing.assert_array_equal(model.predict(X),system.evaluate(X,constants))
        bundle.update(formulas=system.formulas(constants,spec['feature_names']),sharing=system.sharing,domain_audits=result['audits'])
        write(folder/'delivery.json',bundle)
        write(folder/'pool.json',delivery.pool(read(folder/'model.json'),spec))
        result.update(status='complete',data_sha256=arrays_hash(X,Y,V,Z),development_accepted=bundle['development_accepted'])
    except Exception as error:
        result.update(status='failed',error=repr(error));(folder/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
    result.update(worker_wall_seconds=time.monotonic()-start,worker_cpu_seconds=time.process_time()-cpu)
    write(folder/'result.json',result)


if __name__=='__main__':
    run(Path(sys.argv[1]),read(sys.argv[2]),read(sys.argv[3]))
