"""Fit observed training/validation data only; no latent-target or test selection."""
from dataclasses import asdict
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from dimensr_mimo import DimenSRMIMORegressor,SearchConfig,PhysicalModel
from dimensr_mimo._single import DimenSRRegressor
from experiments.second_round_20261026.cases import previous
from experiments.compound_search_20261027 import cases
from experiments.typed_coverage_20261026 import delivery
from experiments.native_api_20261026.run import read,write,sha,arrays_hash

base=previous('worker',cases=cases);base.generate=cases.training_data


def run(folder,job,config):
    torch.set_num_threads(1);start=time.monotonic();cpu=time.process_time()
    result=dict(job=job,status='failed',models=[],diagnostics=[],prediction_hashes=[],audits=[],test_scores_computed=False)
    try:
        X,Y=cases.training_data(job['case'],'train',job['seed']);V,Z=cases.training_data(job['case'],'validation',job['seed'])
        spec=cases.design(job['case']);single=job['method']=='independent'
        if single:
            base.run(folder,job,config);result=read(folder/'result.json');result['test_scores_computed']=False
            if result['status']!='complete':return
        else:
            cfg=SearchConfig(**config);cfg.random_state=job['seed'];cfg.wall_seconds=52
            cfg.initialization='dimensional' if job['method']=='dimensional' else 'compound'
            if job['method']=='atoms':
                from dimensr_mimo import structure
                structure._compound_products=lambda *args: None
            cfg.final_selection='precision';cfg.acceptance=spec['acceptance']
            def event(row):
                with (folder/'events.jsonl').open('a',encoding='utf-8') as f:
                    import json
                    f.write(json.dumps(dict(row,output=None,monotonic=time.monotonic()),allow_nan=False)+'\n')
            m=DimenSRMIMORegressor(cfg,spec['features'],spec['targets'],spec['constants'],spec['feature_names']).fit(
                X,Y,X_val=V,Y_val=Z,input_bounds=spec['bounds'],callback=event)
            m.save(folder/'model.json');result.update(models=['model.json'],diagnostics=[m.diagnostics_],effective_config=asdict(cfg),
                search_fingerprint=m.search_fingerprint_,application_acceptance=m.post_selection_)
            physical=PhysicalModel.from_estimator(m,input_bounds=spec['bounds'],physical_limits=spec['acceptance'],
                training_data=(X,Y),validation_data=(V,Z));physical.save(folder/'physical.json')
            result['prediction_hashes']=[{n:arrays_hash(m.predict(A)) for n,A in [('train',X),('validation',V)]}]
            result['audits']=[m.audit_domain(spec['bounds'])]
        cls=DimenSRRegressor if single else DimenSRMIMORegressor;models=[cls.load(folder/n) for n in result['models']]
        system,constants=delivery.graph(models,single);bundle=delivery.record(system,constants,spec,X,Y,V,Z)
        expected=np.column_stack([m.predict(X) for m in models]) if single else models[0].predict(X)
        np.testing.assert_array_equal(expected,system.evaluate(X,constants))
        bundle.update(formulas=system.formulas(constants,spec['feature_names']),sharing=system.sharing,domain_audits=result['audits'])
        write(folder/'delivery.json',bundle)
        if not single:write(folder/'pool.json',delivery.pool(read(folder/'model.json'),spec))
        result.update(status='complete',data_sha256=arrays_hash(X,Y,V,Z),development_accepted=bundle['development_accepted'],test_scores_computed=False)
    except Exception as e:
        result.update(status='failed',error=repr(e));(folder/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
    result.update(worker_wall_seconds=time.monotonic()-start,worker_cpu_seconds=time.process_time()-cpu)
    write(folder/'result.json',result)


if __name__=='__main__':run(Path(sys.argv[1]),read(Path(sys.argv[2])),read(Path(sys.argv[3])))
