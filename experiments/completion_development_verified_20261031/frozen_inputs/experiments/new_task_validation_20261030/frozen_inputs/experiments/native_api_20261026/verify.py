"""Independent reload, archived-selector comparison and raw JSON interpretation."""
from pathlib import Path
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from dimensr_mimo import DimenSRMIMORegressor,PhysicalModel
from dimensr_mimo import audit
from experiments.native_api_20261026.run import read,write,sha,arrays_hash,cases,example
from experiments.precision_delivery_20261026.selector import select as archived_select


def require(condition,message):
    if not condition:raise ValueError(message)


def raw(graph,constants,X):
    """Evaluate serialized nodes directly, without System.evaluate or SymPy."""
    functions=dict(add=np.add,sub=np.subtract,mul=np.multiply,div=np.divide,neg=np.negative,
        square=np.square,sqrt=np.sqrt,sin=np.sin,cos=np.cos,exp=np.exp,log=np.log)
    values=[]
    with np.errstate(all='ignore'):
        for v in graph['nodes']:
            if v['parameter'] is not None:value=np.full(len(X),constants[v['parameter']])
            elif not v['children']:value=X[:,int(v['token'][1:])]
            else:
                args=[values[c] for c in v['children']];value=functions[v['token']](*args)
                value=np.where(np.isfinite(value)&np.logical_and.reduce([np.isfinite(a) for a in args]),value,np.nan)
            values.append(value)
    return np.column_stack([values[r] for r in graph['roots']])


def untimed(history):return [{k:v for k,v in r.items() if not k.endswith('_seconds')} for r in history]


def mechanism(model):
    history=model['history'];d=model['diagnostics']
    out=dict(neural_valid=sum(r.get('neural_valid',0) for r in history),
        policy_parameter_delta=sum(r.get('parameter_delta',0) for r in history),
        gp_new_evaluations=sum(r['new_evaluations'] for r in history if r['phase']=='gp'),
        nsga2_new_evaluations=sum(r['new_evaluations'] for r in history if r['phase']=='nsga2'),
        nsga2_generations=sum(r['phase']=='nsga2' for r in history),rademacher_systems=d['rademacher_systems'])
    require(all(v>0 for v in out.values()) and out['nsga2_generations']>=2,'Incomplete runtime mechanism evidence.')
    return out


def verify_cell(job):
    case,method=job['case'],job['method'];folder=HERE/'runs'/f'{case}_{method}'
    result=read(folder/'result.json');require(result['status']=='complete','Incomplete cell.')
    saved=read(folder/'model.json');model=DimenSRMIMORegressor.load(folder/'model.json')
    require(model.to_dict()==saved,'Save/reload/resave drift.')
    require(not saved['diagnostics']['deadline_reached'],'Time cutoff changed search budget.')
    require(saved['config']==job['config'],'Configuration drift.')
    evidence=dict(case=case,method=method,mechanism=mechanism(saved),verified=True)
    if method=='physical':
        X,Y,V,Z=example.data();physical=PhysicalModel.load(folder/'physical.json');p=read(folder/'physical.json')
        require(physical.to_dict()==p,'Physical metadata drift.')
        predictions=[]
        for name,A,B in [('train',X,Y),('validation',V,Z)]:
            pred=raw(p['system'],p['constants'],A)
            np.testing.assert_array_equal(pred,physical.predict(A))
            np.testing.assert_array_equal(pred,physical.coordinates.restore_outputs(model.predict(physical.coordinates.inputs(A))))
            np.testing.assert_allclose(np.sqrt(np.mean((pred-B)**2,axis=0))/p['source']['nrmse_scale_physical'],
                p['source']['physical_accuracy'][name+'_nrmse'],rtol=1e-12,atol=1e-14)
            require(arrays_hash(A,B)==p['source']['physical_accuracy'][name+'_sha256'],'Physical input hash drift.')
            require(arrays_hash(pred)==result['prediction_hashes'][name],'Physical prediction hash drift.')
            predictions.append(pred)
        require(p['physical_limits']==example.PHYSICAL_LIMITS,'Physical limits drift.')
        require(all(a['status']=='enclosed' for a in p['domain_audits']),'Physical domain not enclosed.')
        structure=audit.complexity(physical.system);require(structure==p['physical_structure'],'Physical complexity differs.')
        actual=all(a<=b for key in ('train_nrmse','validation_nrmse') for a,b in zip(p['source']['physical_accuracy'][key],example.PHYSICAL_LIMITS['nrmse']))
        actual=actual and all(structure[k]<=example.PHYSICAL_LIMITS[k] for k in example.PHYSICAL_LIMITS if k!='nrmse') and p['source']['relative_constraints_ok']
        require((p['physical_acceptance']['status']=='accepted')==actual,'Physical acceptance differs.')
        evidence.update(search_acceptance=saved['post_selection']['status'],physical_acceptance=p['physical_acceptance'],
            accuracy=p['source']['physical_accuracy'],source_structure=p['source_structure'],physical_structure=structure,
            fitted_parameters=p['fitted_parameters'],fixed_coordinate_parameters=p['fixed_coordinate_parameters'],formulas=p['formulas'])
    else:
        baseline=read(HERE/('baseline' if method=='legacy' else 'baseline_corrected')/f'{case}_{method}/model.json')
        require(saved['search_fingerprint']==baseline['search_fingerprint'],'Search fingerprint changed.')
        require(saved['candidates']==baseline['candidates'],'Scored candidate pool changed.')
        keys=('frozen_candidate_sha256','evaluations','constant_nfev','operations','changed','novel','successful','rademacher_systems')
        require(all(saved['diagnostics'][k]==baseline['diagnostics'][k] for k in keys),'Search counters changed.')
        require(untimed(saved['history'])==untimed(baseline['history']),'Algorithm trace changed.')
        expected=baseline if method=='legacy' else archived_select(baseline,cases.design(case)['acceptance'])
        require(saved['best']==expected['best'],'Final choice differs from archived selector.')
        if method=='normalized':
            for k,v in expected['post_selection'].items():
                if k!='source_model_sha256':require(saved['post_selection'][k]==v,'Selection record drift: '+k)
        X,Y=cases.training_data(case,'train',601);V,Z=cases.training_data(case,'validation',601)
        for name,A,B in [('train',X,Y),('validation',V,Z)]:
            pred=raw(saved['best']['system'],saved['best']['constants'],A)
            np.testing.assert_array_equal(pred,model.predict(A))
            np.testing.assert_array_equal(pred,raw(expected['best']['system'],expected['best']['constants'],A))
            require(arrays_hash(pred)==result['prediction_hashes'][name],'Prediction hash drift.')
        for c in saved['candidates']:
            for A,B,key in [(X,Y,'errors'),(V,Z,'validation_errors')]:
                errors=np.sqrt(np.mean((raw(c['system'],c['constants'],A)-B)**2,axis=0))/saved['scales']
                np.testing.assert_allclose(errors,c[key],rtol=1e-9,atol=1e-11)
        require(model.audit_domain(cases.design(case)['bounds'])['status']=='enclosed','Domain not enclosed.')
        evidence.update(search_and_pool_identical=True,candidates_recomputed=len(saved['candidates']),
            final_choice_identical=True,acceptance=saved.get('post_selection',{}).get('status','legacy_weighted'))
    require(arrays_hash(X,Y,V,Z)==job['data_sha256'],'Input arrays drift.')
    return evidence


def main():
    if (HERE/'VERIFICATION.json').exists():raise RuntimeError('Retain existing verification.')
    frozen=read(HERE/'FREEZE.json');models=read(HERE/'ALL_MODELS_FROZEN.json')
    require(models['freeze_sha256']==sha(HERE/'FREEZE.json'),'Freeze chain drift.')
    for key,value in frozen['source_sha256'].items():
        require(sha(ROOT/key)==value==sha(HERE/'source'/key),'Source drift: '+key)
    for key,value in frozen['files'].items():require(sha(ROOT/key)==value,'Protocol drift: '+key)
    for key,value in models['files'].items():require(sha(HERE/key)==value,'Artifact drift: '+key)
    rows=[]
    for job in frozen['jobs']:
        try:row=verify_cell(job)
        except Exception as e:row=dict(case=job['case'],method=job['method'],verified=False,error=repr(e))
        rows.append(row);print({k:v for k,v in row.items() if k in ('case','method','verified','error')},flush=True)
    write(HERE/'VERIFICATION.json',dict(status='passed' if all(r['verified'] for r in rows) else 'failed',
        expected_systems=len(frozen['jobs']),verified_systems=sum(r['verified'] for r in rows),rows=rows,
        verified_unix=time.time(),freeze_sha256=sha(HERE/'FREEZE.json'),models_sha256=sha(HERE/'ALL_MODELS_FROZEN.json'),
        test_scores_computed=False,claim_scope='API integration on known development tasks; not new benchmark evidence.'))


if __name__=='__main__':main()
