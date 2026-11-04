"""Fit three physical outputs with fixed training coordinates and honest gates."""
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from dimensr_mimo import DimenSRMIMORegressor,SearchConfig,PhysicalModel,AffineCoordinates,audit
from dimensr_mimo.physical import restore_graph
from dimensr_mimo._single import DimenSRRegressor,SearchConfig as SingleConfig
from dimensr_mimo._single.units import unit,sub,scale,ZERO,encode
from experiments.slump_physical_20261026 import data
from experiments.typed_coverage_20261026 import delivery
from experiments.typed_coverage_20261026.worker import base
from experiments.native_api_20261026.run import arrays_hash

SEARCH_LIMITS=dict(nrmse=[.5]*3,shared_graph_size=36,independent_parameters=9,expanded_total_nodes=45,maximum_output_depth=6)
PHYSICAL_LIMITS=dict(nrmse=[.5]*3,shared_graph_size=80,independent_parameters=29,expanded_total_nodes=140,maximum_output_depth=10)
NAMES=['Cement','Slag','Fly_ash','Water','SP','Coarse_aggregate','Fine_aggregate']


def constant_units():
    density=unit(data.DENSITY);catalog=[ZERO,scale(density,-1)]
    for target in (unit(data.L),unit(data.PRESSURE)):
        catalog.extend((target,sub(target,density),sub(target,scale(density,2))))
    return [encode(u) for u in dict.fromkeys(catalog)]


def run(folder,job,settings):
    torch.set_num_threads(1);started=time.monotonic();output=None
    manifest=data.read(HERE/'DATA_MANIFEST.json');coords=AffineCoordinates(**manifest['coordinates'])
    X,Y=data.load('train');V,Z=data.load('val');A,B,C,D=coords.inputs(X),coords.outputs(Y),coords.inputs(V),coords.outputs(Z)
    features=manifest['feature_units'];targets=manifest['target_units'];catalog=constant_units();bounds=coords.input_bounds(data.BOUNDS)
    result=dict(job=job,status='failed',models=[],diagnostics=[],prediction_hashes=[],audits=[],test_scores_computed=False)
    def event(row):
        with (folder/'events.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(dict(row,output=output,monotonic=time.monotonic()),allow_nan=False)+'\n')
    try:
        cfg=SearchConfig(**settings);cfg.random_state=job['seed'];cfg.wall_seconds=52
        single=job['method']=='independent';models=[]
        if single:
            for output in range(3):
                event(dict(phase='output_start',allowance_seconds=20))
                options={k:v for k,v in asdict(cfg).items() if k in SingleConfig.__dataclass_fields__};options['max_evaluations']=400
                stats,restore=base.instrument_single(event)
                try:
                    m=DimenSRRegressor(SingleConfig(**options),features,targets[output],catalog,NAMES)
                    m.fit(A,B[:,output],X_val=C,y_val=D[:,output],input_bounds=bounds,callback=event)
                    filename=f'model-{output}.json';m.save(folder/filename);result['models'].append(filename)
                    result['diagnostics'].append(dict(m.diagnostics_,instrumentation=dict(stats)));models.append(m)
                finally:restore()
                event(dict(phase='output_complete'))
        else:
            cfg.initialization='legacy' if job['method']=='legacy' else 'dimensional';cfg.final_selection='precision';cfg.acceptance=SEARCH_LIMITS
            m=DimenSRMIMORegressor(cfg,features,targets,catalog,NAMES).fit(A,B,X_val=C,Y_val=D,input_bounds=bounds,callback=event)
            m.save(folder/'model.json');models=[m];result.update(models=['model.json'],diagnostics=[m.diagnostics_],
                application_acceptance=m.post_selection_,effective_config=asdict(cfg))
            physical=PhysicalModel.from_estimator(m,coords,data.BOUNDS,PHYSICAL_LIMITS,training_data=(X,Y),validation_data=(V,Z))
            physical.save(folder/'physical.json');result['native_physical_acceptance']=physical.acceptance
        original,constants=delivery.graph(models,single)
        system,values,physical_catalog,roles=restore_graph(original,constants,features,targets,catalog,coords)
        unit_report=audit.units(system,features,targets,physical_catalog);shape=audit.complexity(system);source_shape=audit.complexity(original)
        scales=Y.std(axis=0);metrics={}
        for name,Q,R,S in [('train',X,Y,A),('validation',V,Z,C)]:
            pred=system.evaluate(Q,values);internal=np.column_stack([m.predict(S) for m in models]) if single else models[0].predict(S)
            np.testing.assert_array_equal(pred,coords.restore_outputs(internal))
            metrics[name+'_nrmse']=(np.sqrt(np.mean((pred-R)**2,axis=0))/scales).tolist()
        result['prediction_hashes']=[{n:arrays_hash(m.predict(Q)) for n,Q in [('train',A),('validation',C)]} for m in models]
        result['audits']=[m.audit_domain(input_bounds=bounds) for m in models]
        audits=system.audit(values,data.BOUNDS,max_boxes=128)
        if not all(a['status']=='enclosed' for a in audits):raise ValueError('Physical domain was not enclosed.')
        common=all(delivery.accepted(np.asarray(metrics[k]),shape,PHYSICAL_LIMITS) for k in ('train_nrmse','validation_nrmse'))
        bundle=dict(system=system.to_dict(),constants=values.tolist(),constant_units=physical_catalog,units=unit_report,
            source_system=original.to_dict(),source_constants=constants.tolist(),source_structure=source_shape,physical_structure=shape,
            parameter_roles=roles,parameters=[dict(index=i,value=float(v),units=unit_report['parameter_units'][i],**r) for i,(v,r) in enumerate(zip(values,roles))],
            coordinates=coords.to_dict(),input_bounds=data.BOUNDS,formulas=system.formulas(values,NAMES),source_sharing=original.sharing,
            physical_sharing=system.sharing,physical_limits=PHYSICAL_LIMITS,metrics=metrics,domain_audits=audits,
            common_absolute_accepted=common,joint_relative_constraints=None if single else bool(physical.source['relative_constraints_ok']),
            native_physical_acceptance=None if single else physical.acceptance)
        data.write(folder/'delivery.json',bundle)
        result.update(status='complete',data_sha256=arrays_hash(X,Y,V,Z),common_absolute_accepted=common)
    except Exception as e:
        result.update(status='failed',error=repr(e));(folder/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
    result['worker_wall_seconds']=time.monotonic()-started;data.write(folder/'result.json',result)


if __name__=='__main__':run(Path(sys.argv[1]),data.read(sys.argv[2]),data.read(sys.argv[3]))
