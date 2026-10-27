"""Score only sealed systems, then independently interpret their raw JSON graphs."""
from collections import Counter
from pathlib import Path
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from dimensr_mimo import DimenSRMIMORegressor,PhysicalModel,audit
from dimensr_mimo._single import DimenSRRegressor
from experiments.compound_search_20261027 import cases
from experiments.mutation_units_20261027 import study
from experiments.typed_coverage_20261026 import delivery
from experiments.native_api_20261026.run import read,write,sha,arrays_hash
from experiments.native_api_20261026.verify import raw,require


def models(folder,result,single):
    cls=DimenSRRegressor if single else DimenSRMIMORegressor
    return [cls.load(folder/n) for n in result['models']]


def prediction(ms,X,single):return np.column_stack([m.predict(X) for m in ms]) if single else ms[0].predict(X)


def error(pred,true,scale):return (np.sqrt(np.mean((pred-true)**2,axis=0))/scale).tolist()


def guards():
    f=read(HERE/'FREEZE.json');m=read(HERE/'ALL_MODELS_FROZEN.json')
    study.verify_files(f['files']);study.verify_files(m['files'],HERE)
    require(m['freeze_sha256']==sha(HERE/'FREEZE.json'),'Freeze chain drift.')
    require(not f['test_labels_accessed'] and not m['test_labels_accessed'],'Test labels were not isolated.')
    require(f['frozen_unix']<=read(HERE/'TRAINING_STARTED.json')['start_unix']<=m['frozen_unix'],'Freeze chronology drift.')
    return f,m


def evaluate():
    if (HERE/'SCORES.json').exists():raise RuntimeError('Keep existing scores.')
    f,m=guards();rows=[]
    for job in f['jobs']:
        require(time.time()<study.budget(),'Scoring budget expired.')
        folder=HERE/'runs'/study.name(job);resource=read(folder/'resources.json')
        row=dict(job=job,status=resource['status'],resources=resource)
        if resource['status']=='complete':
            try:
                r=read(folder/'result.json');bundle=read(folder/'delivery.json');single=job['method']=='independent'
                ms=models(folder,r,single);X,Y=cases.generate(job['case'],'test',job['seed']);clean=cases.latent(job['case'],X)
                _,train=cases.training_data(job['case'],'train',job['seed']);scale=train.std(axis=0);pred=prediction(ms,X,single)
                observed,latent=error(pred,Y,scale),error(pred,clean,scale);spec=cases.design(job['case'])
                accepted=delivery.accepted(np.asarray(observed),bundle['structure'],spec['acceptance']) and all(a<=b for a,b in zip(latent,spec['clean_test_nrmse']))
                application=bundle['development_accepted'] and (single or ms[0].post_selection_['status']=='accepted')
                mechanism=study.mechanism(folder)
                rad=[d['instrumentation'].get('local_rademacher_calls',0) if single else d.get('rademacher_systems',0) for d in r['diagnostics']]
                mechanism.update(rademacher_calls=rad,full_mechanism_pass=mechanism['full_mechanism_pass'] and all(n>0 for n in rad))
                row.update(status='complete',observed_nrmse=observed,clean_nrmse=latent,mean_clean_nrmse=float(np.mean(latent)),
                    development_accepted=application,test_accepted=accepted,both_accepted=application and accepted,structure=bundle['structure'],
                    sharing=bundle['sharing'],mechanism=mechanism,domain_pass=all(a['status']=='enclosed' for a in r['audits']))
            except Exception as e:row.update(status='scoring_failed',error=repr(e))
        rows.append(row)
    write(HERE/'SCORES.json',dict(rows=rows,scored_unix=time.time(),all_models_sha256=sha(HERE/'ALL_MODELS_FROZEN.json')))


def verify_cell(job,score,frozen):
    folder=HERE/'runs'/study.name(job);seal=read(folder/'CELL_FROZEN.json');study.verify_files(seal['files'],folder)
    result=read(folder/'result.json');require(result['status']=='complete' and not result['test_scores_computed'],'Incomplete or unisolated cell.')
    spec=cases.design(job['case']);single=job['method']=='independent';ms=models(folder,result,single)
    X,Y=cases.training_data(job['case'],'train',job['seed']);V,Z=cases.training_data(job['case'],'validation',job['seed']);T,U=cases.generate(job['case'],'test',job['seed'])
    require(arrays_hash(X,Y,V,Z)==result['data_sha256']==frozen['data_sha256'][f"{job['case']}_s{job['seed']}"],'Data drift.')
    system,constants=delivery.graph(ms,single);bundle=read(folder/'delivery.json')
    require(system.to_dict()==bundle['system'] and np.array_equal(constants,[p['value'] for p in bundle['parameters']]),'Raw graph or parameters drift.')
    require(audit.units(system,spec['features'],spec['targets'],spec['constants'])==bundle['units'],'Unit audit drift.')
    require(audit.complexity(system)==bundle['structure']==score['structure'],'Structure drift.')
    require(bundle['sharing']==system.sharing==score['sharing'],'Sharing drift.')
    expected=delivery.record(system,constants,spec,X,Y,V,Z)
    require(all(bundle[k]==v for k,v in expected.items()),'Delivery metrics drift.')
    for A in (X,V,T):np.testing.assert_array_equal(prediction(ms,A,single),raw(bundle['system'],constants,A))
    hashes=[{n:arrays_hash(m.predict(A)) for n,A in [('train',X),('validation',V)]} for m in ms]
    require(hashes==result['prediction_hashes'],'Saved prediction hashes drift.')
    scale=Y.std(axis=0);pred=raw(bundle['system'],constants,T)
    np.testing.assert_allclose(error(pred,U,scale),score['observed_nrmse'],rtol=1e-12,atol=1e-14)
    clean=error(pred,cases.latent(job['case'],T),scale)
    np.testing.assert_allclose(clean,score['clean_nrmse'],rtol=1e-12,atol=1e-14)
    audits=[m.audit_domain(input_bounds=spec['bounds']) for m in ms]
    require(audits==result['audits']==bundle['domain_audits'],'Domain audit drift.')
    require(all(a['status']=='enclosed' for a in audits),'Domain failure.')
    base_mechanism=study.mechanism(folder)
    require(base_mechanism['outputs']==score['mechanism']['outputs'] and score['mechanism']['full_mechanism_pass'],'Mechanism failure.')
    accepted=delivery.accepted(np.asarray(score['observed_nrmse']),bundle['structure'],spec['acceptance']) and all(a<=b for a,b in zip(clean,spec['clean_test_nrmse']))
    require(accepted==score['test_accepted'],'Test acceptance drift.')
    application=bundle['development_accepted'] and (single or ms[0].post_selection_['status']=='accepted')
    require(application==score['development_accepted'] and (application and accepted)==score['both_accepted'],'Development acceptance drift.')
    evidence=dict(job=job,verified=True,clean_nrmse=clean,observed_nrmse=score['observed_nrmse'],both_accepted=score['both_accepted'],
        structure=bundle['structure'],mechanism_pass=True,domain_pass=True)
    if not single:
        physical=PhysicalModel.load(folder/'physical.json');require(physical.to_dict()==read(folder/'physical.json'),'Physical metadata drift.')
        for A in (X,V,T):np.testing.assert_array_equal(physical.predict(A),prediction(ms,A,False))
        pool=read(folder/'pool.json');require(pool==delivery.pool(ms[0].to_dict(),spec),'Pool diagnostics drift.')
        for c in ms[0].candidates_:
            for A,B,key in ((X,Y,'errors'),(V,Z,'validation_errors')):
                np.testing.assert_allclose(error(raw(c['system'],c['constants'],A),B,scale),c[key],rtol=1e-9,atol=1e-11)
        evidence.update(candidate_count=pool['candidates'],absolute_structure_feasible=pool['feasible_count'],
            application_acceptance=ms[0].post_selection_['status'],physical_acceptance=physical.acceptance['status'])
    return evidence


def summarize(scores,verified):
    methods={};pairs=[];index={(r['job']['case'],r['job']['seed'],r['job']['method']):r for r in scores}
    for method in study.METHODS:
        rows=[r for r in scores if r['job']['method']==method];valid=[r for r in rows if r['status']=='complete']
        methods[method]=dict(expected=len(rows),complete=len(valid),both_accepted=sum(r['both_accepted'] for r in valid),
            mean_clean_nrmse=float(np.mean([r['mean_clean_nrmse'] for r in valid])) if len(valid)==len(rows) else None,
            mean_shared_graph_size=float(np.mean([r['structure']['shared_graph_size'] for r in valid])) if len(valid)==len(rows) else None,
            statuses=dict(Counter(r['status'] for r in rows)),strata={})
        groups={'original_n0':lambda c:c.endswith('_n0'), 'original_n1':lambda c:c.endswith('_n1'),
            'variants':lambda c:c in cases.VARIANTS, 'regressions':lambda c:c in cases.REGRESSIONS}
        for label,predicate in groups.items():
            expected=[r for r in rows if predicate(r['job']['case'])]
            layer=[r for r in valid if predicate(r['job']['case'])]
            methods[method]['strata'][label]=dict(expected=len(expected),complete=len(layer),both_accepted=sum(r['both_accepted'] for r in layer),
                mean_clean_nrmse=float(np.mean([r['mean_clean_nrmse'] for r in layer])) if layer and len(layer)==len(expected) else None)
    for case in ('polynomial_n0','polynomial_n1'):
        for seed in cases.seeds(case):
            a,b=index[case,seed,'compound'],index[case,seed,'dimensional'];p=dict(case=case,seed=seed,available=a['status']==b['status']=='complete')
            if p['available']:
                bounds=np.asarray(b['clean_nrmse'])*1.1+1e-8
                p.update(difference=(np.asarray(a['clean_nrmse'])-b['clean_nrmse']).tolist(),protected=(np.asarray(a['clean_nrmse'])<=bounds).tolist(),
                    candidate_accepted=a['both_accepted'],control_accepted=b['both_accepted'])
            pairs.append(p)
    protected=sum(sum(p.get('protected',[])) for p in pairs);technical=all(v.get('verified',False) for v in verified)
    candidate,base=methods['compound'],methods['dimensional']
    adoption=technical and protected==8 and candidate['both_accepted']>=base['both_accepted'] and candidate['mean_clean_nrmse']<=base['mean_clean_nrmse']
    return dict(methods=methods,pairs=pairs,protected_outputs=protected,expected_outputs=8,technical_pass=technical,
        adoption_pass=adoption,production_default_changed=False,claim_scope='Declared new-family and noise development regression, not blind or real-data validation.')


def verify():
    if (HERE/'VERIFICATION.json').exists():raise RuntimeError('Retain verification.')
    f,m=guards();s=read(HERE/'SCORES.json');require(s['all_models_sha256']==sha(HERE/'ALL_MODELS_FROZEN.json'),'Scores chain drift.')
    require(m['frozen_unix']<=s['scored_unix'],'Scoring preceded model freeze.')
    rows=[]
    for job,score in zip(f['jobs'],s['rows']):
        require(time.time()<study.budget(),'Verification budget expired.')
        try:
            require(score['job']==job and score['status']=='complete','Incomplete cell: '+score['status'])
            row=verify_cell(job,score,f)
        except Exception as e:row=dict(job=job,verified=False,error=repr(e))
        rows.append(row);print(study.name(job),'verified' if row['verified'] else row['error'],flush=True)
    summary=summarize(s['rows'],rows)
    write(HERE/'VERIFICATION.json',dict(status='passed' if summary['technical_pass'] else 'failed',verified_systems=sum(r['verified'] for r in rows),
        expected_systems=8,rows=rows,verified_unix=time.time(),elapsed_seconds=time.time()-f['budget']['start_unix'],
        freeze_sha256=sha(HERE/'FREEZE.json'),models_sha256=sha(HERE/'ALL_MODELS_FROZEN.json'),scores_sha256=sha(HERE/'SCORES.json')))
    write(HERE/'SUMMARY.json',summary)


if __name__=='__main__':
    evaluate();verify()
