"""Historical holdout scoring and independent original-variable verification."""
from pathlib import Path
import json
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from dimensr_mimo import DimenSRMIMORegressor,PhysicalModel,AffineCoordinates,audit
from dimensr_mimo._single import DimenSRRegressor
from dimensr_mimo.physical import restore_graph
from experiments.slump_physical_20261026 import data,study,worker
from experiments.typed_coverage_20261026 import delivery
from experiments.native_api_20261026.run import arrays_hash
from experiments.native_api_20261026.verify import raw,require
read,write,sha=data.read,data.write,data.sha


def models(folder,result):
    cls=DimenSRRegressor if result['job']['method']=='independent' else DimenSRMIMORegressor
    return [cls.load(folder/n) for n in result['models']]


def reconstruct(ms,single):
    manifest=read(HERE/'DATA_MANIFEST.json');coords=AffineCoordinates(**manifest['coordinates'])
    original,constants=delivery.graph(ms,single)
    system,values,units,roles=restore_graph(original,constants,manifest['feature_units'],manifest['target_units'],worker.constant_units(),coords)
    return system,values,units,roles,original,constants,coords


def predict(ms,X,coords,single):
    A=coords.inputs(X);internal=np.column_stack([m.predict(A) for m in ms]) if single else ms[0].predict(A)
    return coords.restore_outputs(internal)


def gates(errors,shape):return delivery.accepted(np.asarray(errors),shape,worker.PHYSICAL_LIMITS)


def guard():
    f,m=read(HERE/'FREEZE.json'),read(HERE/'ALL_MODELS_FROZEN.json')
    study.check_files(f['files']);study.check_files(m['files'],HERE)
    require(m['freeze_sha256']==sha(HERE/'FREEZE.json'),'Model freeze chain drift.')
    require(not f['test_labels_accessed'] and not m['test_labels_accessed'],'Premature test access.')
    require(f['frozen_unix']<=read(HERE/'TRAINING_STARTED.json')['start_unix']<=m['frozen_unix'],'Freeze chronology drift.')
    return f,m


def evaluate():
    if (HERE/'SCORES.json').exists():raise RuntimeError('Preserve existing scores.')
    f,m=guard();X,Y=data.load('train');V,Z=data.load('val');T,U=data.load('test');scales=Y.std(axis=0);rows=[]
    bounds=np.asarray(data.BOUNDS)
    require(np.all(T>=bounds[:,0]) and np.all(T<=bounds[:,1]),'Historical test inputs exceed frozen physical domain.')
    for job in f['jobs']:
        require(time.time()<study.budget(),'Scoring budget expired.')
        folder=HERE/'runs'/study.name(job);resource=read(folder/'resources.json');row=dict(job=job,status=resource['status'],resources=resource)
        if resource['status']=='complete':
            try:
                r=read(folder/'result.json');ms=models(folder,r);single=job['method']=='independent'
                system,values,units,roles,original,constants,coords=reconstruct(ms,single)
                pred=predict(ms,T,coords,single);rmse=np.sqrt(np.mean((pred-U)**2,axis=0));nrmse=rmse/scales
                shape=audit.complexity(system);b=read(folder/'delivery.json');test_ok=gates(nrmse,shape)
                application=b['common_absolute_accepted'] and (single or b['native_physical_acceptance']['status']=='accepted')
                row.update(status='complete',nrmse=nrmse.tolist(),rmse_si=rmse.tolist(),rmse_cm_cm_mpa=(rmse/np.array([.01,.01,1e6])).tolist(),
                    physical_structure=shape,source_structure=audit.complexity(original),test_accepted=test_ok,
                    development_accepted=application,both_accepted=application and test_ok,common_absolute_development=b['common_absolute_accepted'],
                    mechanism=study.mechanism(folder,r),domain_pass=all(a['status']=='enclosed' for a in b['domain_audits']))
            except Exception as e:row.update(status='scoring_failed',error=repr(e))
        rows.append(row)
    write(HERE/'SCORES.json',dict(rows=rows,scored_unix=time.time(),models_sha256=sha(HERE/'ALL_MODELS_FROZEN.json')))


def verify_cell(job,score,frozen):
    folder=HERE/'runs'/study.name(job);study.check_files(read(folder/'CELL_FROZEN.json')['files'],folder)
    r=read(folder/'result.json');b=read(folder/'delivery.json');require(not r['test_scores_computed'],'Premature scoring.')
    X,Y=data.load('train');V,Z=data.load('val');T,U=data.load('test');scale=Y.std(axis=0)
    require(arrays_hash(X,Y,V,Z)==r['data_sha256']==frozen['data_sha256'],'Physical input drift.')
    ms=models(folder,r);single=job['method']=='independent';system,values,units,roles,original,constants,coords=reconstruct(ms,single)
    require(system.to_dict()==b['system'] and original.to_dict()==b['source_system'],'Raw graph drift.')
    require(np.array_equal(values,b['constants']) and np.array_equal(constants,b['source_constants']),'Parameter drift.')
    require(roles==b['parameter_roles'] and units==b['constant_units'],'Coordinate parameter roles or unit catalog drift.')
    meta=read(HERE/'DATA_MANIFEST.json');unit_report=audit.units(system,meta['feature_units'],meta['target_units'],units)
    require(unit_report==b['units'],'Parameter/output units drift.')
    require(audit.complexity(system)==b['physical_structure']==score['physical_structure'],'Physical complexity drift.')
    require(audit.complexity(original)==b['source_structure']==score['source_structure'],'Source complexity drift.')
    require(system.audit(values,data.BOUNDS,max_boxes=128)==b['domain_audits'],'Physical domain drift.')
    for name,A,B in [('train',X,Y),('validation',V,Z),('test',T,U)]:
        pred=raw(b['system'],b['constants'],A)
        np.testing.assert_array_equal(pred,predict(ms,A,coords,single))
        errors=np.sqrt(np.mean((pred-B)**2,axis=0))/scale
        np.testing.assert_allclose(errors,score['nrmse'] if name=='test' else b['metrics'][name+'_nrmse'],rtol=1e-12,atol=1e-14)
    expected=[{n:arrays_hash(m.predict(coords.inputs(A))) for n,A in [('train',X),('validation',V)]} for m in ms]
    require(expected==r['prediction_hashes'],'Search prediction hash drift.')
    require(study.mechanism(folder,r)==score['mechanism'] and score['mechanism']['passed'],'Runtime mechanism failure.')
    require(score['resources']['wall_seconds']<=60.5,'System deadline exceeded.')
    if single:
        events=[json.loads(line) for line in (folder/'events.jsonl').read_text(encoding='utf-8').splitlines()]
        for j in range(3):
            start=next(e for e in events if e['output']==j and e['phase']=='output_start')
            end=next(e for e in events if e['output']==j and e['phase']=='output_complete')
            require(start['allowance_seconds']==20 and end['monotonic']-start['monotonic']<=20.5,'Per-output deadline exceeded.')
    common=all(gates(b['metrics'][key],b['physical_structure']) for key in ('train_nrmse','validation_nrmse'))
    require(common==b['common_absolute_accepted']==r['common_absolute_accepted'],'Physical development gates drift.')
    require(gates(score['nrmse'],b['physical_structure'])==score['test_accepted'],'Physical test gates drift.')
    if not single:
        p=PhysicalModel.load(folder/'physical.json');require(p.to_dict()==read(folder/'physical.json'),'Native physical reload drift.')
        for A in (X,V,T):np.testing.assert_array_equal(p.predict(A),raw(b['system'],b['constants'],A))
        require(p.acceptance==b['native_physical_acceptance'],'Native physical acceptance drift.')
        source=ms[0].to_dict();internal_scale=ms[0].scales_
        for candidate in source['candidates']:
            for A,B,key in ((X,Y,'errors'),(V,Z,'validation_errors')):
                e=np.sqrt(np.mean((raw(candidate['system'],candidate['constants'],coords.inputs(A))-coords.outputs(B))**2,axis=0))/internal_scale
                np.testing.assert_allclose(e,candidate[key],rtol=1e-9,atol=1e-11)
    return dict(job=job,verified=True,nrmse=score['nrmse'],both_accepted=score['both_accepted'],
        physical_structure=b['physical_structure'],fitted_parameters=sum(x['role']=='fitted' for x in roles),
        fixed_coordinate_parameters=sum(x['role']!='fitted' for x in roles),native_acceptance=None if single else b['native_physical_acceptance']['status'])


def verify():
    if (HERE/'VERIFICATION.json').exists():raise RuntimeError('Preserve existing verification.')
    f,m=guard();s=read(HERE/'SCORES.json');require(s['models_sha256']==sha(HERE/'ALL_MODELS_FROZEN.json'),'Scores freeze chain drift.')
    require(m['frozen_unix']<=s['scored_unix'],'Test scored before global model seal.')
    rows=[]
    for job,score in zip(f['jobs'],s['rows']):
        require(time.time()<study.budget(),'Verification budget expired.')
        try:
            require(score['job']==job and score['status']=='complete','Incomplete cell: '+score['status']);row=verify_cell(job,score,f)
        except Exception as e:row=dict(job=job,verified=False,error=repr(e))
        rows.append(row);print(study.name(job),'verified' if row['verified'] else row['error'],flush=True)
    methods={}
    for method in ('legacy','dimensional','independent'):
        cells=[c for c in s['rows'] if c['job']['method']==method and c['status']=='complete']
        methods[method]=dict(expected=3,complete=len(cells),both_accepted=sum(c['both_accepted'] for c in cells),
            mean_nrmse=np.mean([c['nrmse'] for c in cells],axis=0).tolist() if len(cells)==3 else None)
    pairs=[]
    for seed in (801,802,803):
        a=next(r for r in s['rows'] if r['job']['seed']==seed and r['job']['method']=='dimensional')
        b=next(r for r in s['rows'] if r['job']['seed']==seed and r['job']['method']=='legacy')
        pair=dict(seed=seed,available=a['status']==b['status']=='complete')
        if pair['available']:pair['protected']=(np.asarray(a['nrmse'])<=np.asarray(b['nrmse'])*1.1+1e-8).tolist()
        pairs.append(pair)
    write(HERE/'SUMMARY.json',dict(methods=methods,pairs=pairs,protected_outputs=sum(sum(p.get('protected',[])) for p in pairs),expected_outputs=9,
        production_default_changed=False,scope='Historical public laboratory-data regression and physical-unit delivery; not blind validation or engineering acceptance.'))
    write(HERE/'VERIFICATION.json',dict(status='passed' if all(r['verified'] for r in rows) else 'failed',verified_systems=sum(r['verified'] for r in rows),expected_systems=9,
        rows=rows,elapsed_seconds=time.time()-f['budget']['start_unix'],freeze_sha256=sha(HERE/'FREEZE.json'),models_sha256=sha(HERE/'ALL_MODELS_FROZEN.json'),scores_sha256=sha(HERE/'SCORES.json')))


if __name__=='__main__':evaluate();verify()
