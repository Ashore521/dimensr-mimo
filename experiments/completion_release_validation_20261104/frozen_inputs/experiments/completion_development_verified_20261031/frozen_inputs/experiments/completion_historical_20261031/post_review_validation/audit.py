"""Read-only reconstruction of a failed historical identity batch.

Never changes its original verification, summary, models or acceptance gate.
No initializer, optimization, fitting, official scorer/verifier entry point runs.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent.parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
PRIOR=ROOT/'experiments/precision_repair_20261028'
IDENTITY=ROOT/'experiments/completion_historical_identity_audit_20261031'
SHARED=ROOT/'experiments/compositional_completion_20261031'
OUT=HERE/'post_review_checks.json'
issues=[]
maxdiff=Counter()
TIME={'constant_fit_seconds','evaluation_seconds','elapsed_seconds','cpu_seconds','phase_seconds','worker_wall_seconds','worker_cpu_seconds','monotonic'}
PHASE={'epoch_allowance_seconds','neural_until_seconds','gp_until_seconds'}
FIELDS=('system','constants','errors','complexity','rademacher','rademacher_complexity')
STRUCTURE=('shared_graph_size','independent_parameters','expanded_total_nodes','maximum_output_depth')


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(o):return hashlib.sha256(json.dumps(o,sort_keys=True,allow_nan=False).encode()).hexdigest()
def ahash(*a):return hashlib.sha256(b''.join(np.asarray(x,dtype='<f8').tobytes() for x in a)).hexdigest()
def check(ok,label):
    if not ok:issues.append(label)
def close(a,b,label):
    a,b=np.asarray(a,float),np.asarray(b,float)
    check(a.shape==b.shape,label+':shape')
    if a.shape==b.shape:
        d=float(np.max(np.abs(a-b))) if a.size else 0.
        maxdiff[label.split(':')[0]]=max(maxdiff[label.split(':')[0]],d)
        check(np.isfinite(a).all() and np.isfinite(b).all() and np.allclose(a,b,rtol=1e-9,atol=1e-11),label)
def hashes(files,root):
    bad=[n for n,h in files.items() if not (root/n).is_file() or sha(root/n)!=h]
    check(not bad,'hashes:'+str(root));return bad
def norm(v):
    if isinstance(v,list):return [norm(x) for x in v]
    if isinstance(v,dict):
        d={k:norm(x) for k,x in v.items() if k not in TIME|{'package_version','schema_version','source_model_sha256'}}
        for k in ('config','effective_config'):
            if k in d:d[k].setdefault('compositional_completion',False)
        if 'job' in d:d['job'].pop('method',None)
        return d
    return v
def differences(a,b,path=''):
    if type(a) is not type(b):return [dict(path=path,kind='type',current=a,original=b)]
    if isinstance(a,dict):
        rows=[]
        for k in sorted(set(a)|set(b)):
            if k not in a or k not in b:rows.append(dict(path=path+'/'+k,kind='missing',current=a.get(k),original=b.get(k)))
            else:rows.extend(differences(a[k],b[k],path+'/'+k))
        return rows
    if isinstance(a,list):
        if len(a)!=len(b):return [dict(path=path,kind='length',current=len(a),original=len(b))]
        return [r for i,(x,y) in enumerate(zip(a,b)) for r in differences(x,y,path+'/'+str(i))]
    return [] if a==b else [dict(path=path,kind='value',current=a,original=b)]
def timing_path(path):
    p=path.split('/');return len(p)==4 and p[1]=='history' and p[2].isdigit() and p[3] in PHASE
def key(j):return j['case'],j['seed'],j['method']
def payload(c):return {k:c[k] for k in FIELDS}


def arrays(case,seed):
    """Independent reconstruction of exactly the historical frozen split streams."""
    original=('damped','rational','polynomial')
    variants=('damped_variant','rational_variant','polynomial_variant')
    family=case.split('_')[0]
    def latent(X):
        a,b=X.T
        if case=='products':return np.column_stack((a*b,.5*a*b*b))
        if case=='boundary':return np.column_stack((np.log(a+b),np.sqrt(a+b)))
        if family=='damped':
            decay,frequency=(.65,1.6) if case in variants else (.4,1.1)
            z=b*np.exp(-decay*a);return np.column_stack((z*np.cos(frequency*a),z*np.sin(frequency*a)))
        if family=='rational':
            k=.45 if case in variants else .7;z=1+(k*a)**2;return np.column_stack((b/z,b*b/z))
        k,factor=(.55,.8) if case in variants else (.3,.5)
        return np.column_stack((a*b*(1+k*b*b),factor*a*b*b*(1+k*b*b)))
    def inputs(split):
        code={'train':111,'validation':222,'test':333}[split];n={'train':96,'validation':48,'test':192}[split]
        if case in ('products','boundary'):
            seq=[20261026,seed,code,0 if case=='products' else 5]
            bounds=[[.3,1.4],[.2,1.3]] if case=='products' else [[.2,1.2],[.3,1.3]]
        elif case in variants:seq=[20261027,seed,code,variants.index(case)];bounds=[[.15,1.4],[.3,1.2]]
        else:seq=[20261027,seed,code,original.index(family)];bounds=[[.3,1.4],[.2,1.3]] if family=='polynomial' else [[.1,1.6],[.5,1.5]]
        rng=np.random.default_rng(np.random.SeedSequence(seq));bounds=np.asarray(bounds)
        return rng.uniform(bounds[:,0],bounds[:,1],(n,2))
    clean_sd=latent(inputs('train')).std(axis=0);noisy=case in variants or case.endswith('_n1');result={}
    for split in ('train','validation','test'):
        X=inputs(split);Y=latent(X)
        if case not in ('products','boundary'):
            i=variants.index(case) if case in variants else original.index(family)
            rng=np.random.default_rng(np.random.SeedSequence([20261027,seed,{'train':411,'validation':422,'test':433}[split],i]))
            observed=Y+(.01 if noisy else 0.)*clean_sd*rng.normal(size=Y.shape)
        else:observed=Y.copy()
        result[split]=(X,observed,Y)
    return result


def main():
    if OUT.exists():raise RuntimeError('Keep existing review checkpoint.')
    helper=ROOT/'experiments/new_task_validation_20261030/post_review_audit.py'
    spec=importlib.util.spec_from_file_location('_private_independent_raw_review_helpers',helper)
    helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
    helpers.check=check
    raw,graph_audit=helpers.raw,helpers.graph_audit
    error=lambda p,y,s:np.sqrt(np.mean((p-y)**2,axis=0))/s
    f,m,sc,v,summary,archive,exe=(read(HERE/n) for n in ('FREEZE.json','ALL_MODELS_FROZEN.json','SCORES.json','VERIFICATION.json','SUMMARY.json','FROZEN_INPUT_ARCHIVE.json','EXECUTION.json'))
    diagnosis=read(IDENTITY/'DIAGNOSIS.json');lock=read(SHARED/'CANDIDATE_LOCK.json');jobs=f['jobs']
    check(len(jobs)==34 and len({key(j) for j in jobs})==34,'matrix')
    check([r['job'] for r in sc['rows']]==[r['job'] for r in v['rows']]==jobs,'ordered_denominators')
    check(len(f['files'])==archive['file_count']==966 and f['files']==archive['files'],'input_manifest')
    integrity=dict(live_mismatches=hashes(f['files'],ROOT),archive_mismatches=hashes(f['files'],HERE/'frozen_inputs'),run_mismatches=hashes(m['files'],HERE))
    check(len(m['files'])==archive['sealed_run_file_count']==374 and archive['runs_unchanged'] is True,'run_count')
    check(set(m['files'])=={p.relative_to(HERE).as_posix() for p in (HERE/'runs').rglob('*') if p.is_file()},'all_run_members')
    check(m['freeze_sha256']==archive['freeze_sha256']==v['freeze_sha256']==sha(HERE/'FREEZE.json'),'freeze_chain')
    check(sc['all_models_sha256']==v['models_sha256']==sha(HERE/'ALL_MODELS_FROZEN.json'),'model_chain');check(v['scores_sha256']==sha(HERE/'SCORES.json'),'score_chain')
    check(f['candidate_lock_sha256']==sha(SHARED/'CANDIDATE_LOCK.json') and len(lock['source_files'])==41,'source_lock')
    integrity['source_mismatches']=hashes(lock['source_files'],ROOT)
    check(all(sha(HERE/'source'/p)==h for p,h in lock['source_files'].items()),'source_copies')
    b=read(HERE/'BUDGET.json');check(b==f['budget'],'budget_payload')
    check(all(b[k]==val for k,val in dict(limit_seconds=2880,verification_reserve_seconds=600,system_cap_seconds=60,fit_seconds=52,sequential_workers=1,threads=1).items()),'budget_limits')
    timestamps=dict(freeze=f['frozen_unix'],training=read(HERE/'TRAINING_STARTED.json')['start_unix'],all_models=m['frozen_unix'],scores=sc['scored_unix'],verification=v['verified_unix'])
    check(list(timestamps.values())==sorted(timestamps.values()),'global_chronology')
    check(exe['returncode']==0 and exe['termination_reason'] is None and exe['verification_passed'] is False and exe['wall_seconds']<2880,'execution_failed_terminal')
    check(v['status']=='failed' and v['verified_systems']==17 and v['budget_met'] and v['elapsed_seconds']<2880,'original_verification_failure')
    check(not f['test_labels_accessed'] and not m['test_labels_accessed'],'test_flags')
    check(diagnosis['diagnostic_sha256']==sha(IDENTITY/'diagnose.py'),'diagnosis_script_hash')
    check(all(sha(HERE/n)==h for n,h in diagnosis['source_sha256'].items()),'diagnosis_source_hashes')
    check(diagnosis['pairs_only_history_timing']==17 and diagnosis['expected_pairs']==17 and diagnosis['original_technical_pass'] is False and diagnosis['original_adoption_pass'] is False and diagnosis['permits_development'] is False,'diagnosis_verdict')
    # Native load/domain and the frozen replay are pure postprocessing here.
    from dimensr_mimo import DimenSRMIMORegressor,PhysicalModel
    from experiments.completion_historical_20261031.evaluate import verifier
    old_scores=read(PRIOR/'SCORES.json')['rows'];old_ix={key(r['job']):r for r in old_scores}
    results=[];models={};identity_rows=[];event_count=Counter();data_hashes={};last=timestamps['training'];margins={k:[] for k in ('current_neural','current_gp','prior_neural','prior_gp')}
    for job,score,vr in zip(jobs,sc['rows'],v['rows']):
        tag=f"{job['case']}_s{job['seed']}_{job['method']}";folder=HERE/'runs'/tag;model=read(folder/'model.json');models[key(job)]=model
        cell=read(folder/'CELL_FROZEN.json');check(last<=cell['frozen_unix']<=timestamps['all_models'],tag+':cell_time');last=cell['frozen_unix'];hashes(cell['files'],folder)
        check(set(cell['files'])=={p.name for p in folder.iterdir() if p.is_file() and p.name!='CELL_FROZEN.json'},tag+':cell_manifest')
        check(m['files'][(folder/'CELL_FROZEN.json').relative_to(HERE).as_posix()]==sha(folder/'CELL_FROZEN.json') and all(m['files'][(folder/n).relative_to(HERE).as_posix()]==h for n,h in cell['files'].items()),tag+':cell_membership')
        r=read(folder/'result.json');res=read(folder/'resources.json');bundle=read(folder/'delivery.json');case=f['specs'][job['case']]
        check(r['job']==res['job']==read(folder/'job.json')==job and r['status']==res['status']==score['status']=='complete' and res['returncode']==0,tag+':completion')
        check(not r['test_scores_computed'] and res['wall_seconds']<=60.5 and model['diagnostics']['elapsed_seconds']<=52.5 and model['diagnostics']['evaluations']<=1200,tag+':resources')
        expected=dict(f['config'],random_state=job['seed'],initialization='compound',sharing_strategy='closure',precision_repair=job['method']=='base',compositional_completion=job['method']=='completion',final_selection='precision',acceptance=case['acceptance'])
        check(model['config']==r['effective_config']==expected,tag+':config')
        d=arrays(job['case'],job['seed']);X,Y,_=d['train'];V,Z,_=d['validation'];T,U,clean=d['test'];scale=Y.std(axis=0)
        dh=ahash(X,Y,V,Z);check(dh==r['data_sha256']==f['data_sha256'][f"{job['case']}_s{job['seed']}"],tag+':data_hash');data_hashes[f"{job['case']}_s{job['seed']}"]=dh
        close(scale,model['scales'],'scales:'+tag)
        graph=model['best']['system'];constants=model['best']['constants'];shape,sharing=graph_audit(graph,case,tag)
        check(shape==score['structure']==bundle['structure'] and sharing==score['sharing']==bundle['sharing'],tag+':shape_sharing')
        check(graph==bundle['system'] and constants==[q['value'] for q in bundle['parameters']],tag+':bundle_graph')
        predictions={split:raw(graph,constants,d[split][0]) for split in d};errors={split:error(predictions[split],d[split][1],scale).tolist() for split in d};errors['clean']=error(predictions['test'],clean,scale).tolist()
        close(errors['train'],model['best']['errors'],'selected_train:'+tag);close(errors['train'],bundle['train_nrmse'],'selected_train:'+tag);close(errors['validation'],model['validation_nrmse'],'selected_validation:'+tag);close(errors['validation'],bundle['validation_nrmse'],'selected_validation:'+tag);close(errors['test'],score['observed_nrmse'],'selected_observed:'+tag);close(errors['clean'],score['clean_nrmse'],'selected_clean:'+tag)
        check(r['prediction_hashes']==[dict(train=ahash(predictions['train']),validation=ahash(predictions['validation']))],tag+':prediction_hashes')
        loaded=DimenSRMIMORegressor.load(folder/'model.json');physical=PhysicalModel.load(folder/'physical.json')
        for split in d:
            check(np.array_equal(loaded.predict(d[split][0]),predictions[split]) and np.array_equal(physical.predict(d[split][0]),predictions[split]),tag+':native_physical_reload_'+split)
        domain=loaded.audit_domain(case['bounds']);check([domain]==r['audits']==bundle['domain_audits'] and domain['status']=='enclosed',tag+':domain')
        good_shape=all(shape[k]<=case['acceptance'][k] for k in STRUCTURE)
        dev=bool(good_shape and all(np.all(np.asarray(errors[s])<=case['acceptance']['nrmse']) for s in ('train','validation')))
        test=bool(good_shape and np.all(np.asarray(errors['test'])<=case['acceptance']['nrmse']) and np.all(np.asarray(errors['clean'])<=case['clean_test_nrmse']))
        accepted=dev and test and model['post_selection']['status']=='accepted';check(accepted==score['both_accepted'] and dev==bundle['development_accepted'] and test==score['test_accepted'],tag+':acceptance')
        events=[json.loads(line) for line in (folder/'events.jsonl').read_text(encoding='utf-8').splitlines()]
        neural=[e for e in events if e['phase']=='neural'];gp=[e for e in events if e['phase']=='gp'];nsga=[e for e in events if e['phase']=='nsga2' and e['new_evaluations']>0]
        mechanism=dict(policy_delta=sum(e.get('parameter_delta',0) for e in events),neural_valid=sum(e.get('neural_valid',0) for e in events),gp_new_evaluations=sum(e['new_evaluations'] for e in gp),nsga_steps_with_new=len({e['step'] for e in nsga}),rademacher_systems=model['diagnostics']['rademacher_systems'])
        check(all(x>0 for x in mechanism.values()) and mechanism['nsga_steps_with_new']>=2 and score['mechanism']['full_mechanism_pass'],tag+':mechanism')
        rec=model['diagnostics']['sharing_closure'];n=rec['base_candidate_count'];base=[payload(c) for c in model['candidates'][:n]];check(digest(base)==rec['base_candidate_sha256'],tag+':foundation_hash')
        known={digest(c) for c in base};kind='precision_repair' if job['method']=='base' else 'compositional_completion';supplement=rec[kind];retained={};stats={};seen=set()
        parent=min(base,key=lambda c:(float(np.mean(c['errors'])),c['complexity'],digest(c['system'])))
        if supplement['triggered']:check(supplement['parent']==parent,tag+':fixed_parent')
        for label,record in [('supplement',supplement),('closure',rec)]:
            kept=[];valid=0
            for event in record['events']:
                event_count[label]+=1;check(digest(event['parent']) in known,tag+':parent_source')
                gh=digest(event['graph']);check(gh not in seen,tag+':unique_event_graph');seen.add(gh)
                graph_audit(event['graph'],case,tag+':event')
                parent_error=error(raw(event['parent']['system'],event['parent']['constants'],X),Y,model['scales']);close(parent_error,event['parent']['errors'],'event_parent:'+tag)
                child=event['child'];protected=False
                if child is not None:
                    valid+=1;check(event['new_evaluation'] and child['system']==event['graph'],tag+':child_source');ce=error(raw(child['system'],child['constants'],X),Y,model['scales']);close(ce,child['errors'],'event_child:'+tag);protected=bool(np.all(ce<=parent_error*1.1+1e-8))
                    if protected:kept.append(child);known.add(digest(child))
                check(protected==event['protected'],tag+':training_guard')
            new=sum(e['new_evaluation'] for e in record['events']);check(new==record['new_evaluations'] and valid==record['valid'] and len(kept)==record['protected'] and valid-len(kept)==record['protection_rejections'],tag+':'+label+'_counts')
            check(new<=(12 if label=='supplement' else 48),tag+':'+label+'_cap')
            valid_children=[c for c in kept if np.isfinite(raw(c['system'],c['constants'],V)).all() and np.isfinite(error(raw(c['system'],c['constants'],V),Z,model['scales'])).all()]
            retained[label]=kept;retained[label+'_validation']=valid_children;stats[label]=dict(new_evaluations=new,valid=valid,protected=len(kept),validation_retained=len(valid_children))
        check(rec['base_evaluations']+stats['supplement']['new_evaluations']+stats['closure']['new_evaluations']==model['diagnostics']['evaluations'],tag+':total_evaluations')
        check(all(sum(e['new_evaluation'] for e in rec['events'] if e['round']==q)<=24 for q in (0,1)),tag+':closure_round_caps')
        check(digest(base+retained['supplement']+retained['closure'])==rec['full_training_pool_sha256'],tag+':training_pool')
        check([payload(c) for c in model['candidates'][n:]]==retained['supplement_validation']+retained['closure_validation'],tag+':validation_subset')
        check(rec['validation_retained_extensions']==len(retained['supplement_validation'])+len(retained['closure_validation']),tag+':validation_count')
        if job['method']=='completion':
            check(vr['verified'] and stats['closure']==vr['closure'],tag+':candidate_original_verification')
            for k,value in stats['supplement'].items():check(value==vr['completion'][k],tag+':completion_counter_'+k)
        else:
            check(vr['verified'] is False and vr['error']=="ValueError('Archived control identity drift: model.json')",tag+':original_failure_cause')
            # Supplemental frozen replay does not call control_identity or alter its failure.
            replay_closure,replay_repair=verifier.closure_evidence(job,f)
            check(replay_closure==stats['closure'] and all(replay_repair[k]==value for k,value in stats['supplement'].items()),tag+':supplemental_control_replay')
            old_folder=PRIOR/'runs'/f"{job['case']}_s{job['seed']}_repair"
            changes={name:differences(norm(read(folder/name)),norm(read(old_folder/name))) for name in ('model.json','result.json','delivery.json','pool.json')}
            old_score=old_ix[job['case'],job['seed'],'repair'];changes['score']=differences(norm({k:x for k,x in score.items() if k not in {'job','resources'}}),norm({k:x for k,x in old_score.items() if k not in {'job','resources'}}))
            remaining={name:[z for z in ds if name!='model.json' or not timing_path(z['path'])] for name,ds in changes.items()}
            check(len(changes['model.json'])==12 and not any(remaining.values()),tag+':only_phase_times')
            identity_rows.append(dict(job=job,differences=changes,differences_outside_three_history_timing_fields=remaining,only_declared_history_timing_differs=not any(remaining.values())))
            for prefix,mm in [('current',model),('prior',read(old_folder/'model.json'))]:
                hh=mm['history'];budgets={q['step']:q for q in hh if q['phase']=='phase_budget'}
                for q in hh:
                    if q['phase']=='neural':
                        check(q['proposals']==24,tag+':'+prefix+'_neural_full24');margins[prefix+'_neural'].append(budgets[q['step']]['neural_until_seconds']-q['elapsed_seconds'])
                    if q['phase']=='gp':margins[prefix+'_gp'].append(budgets[q['step']//2]['gp_until_seconds']-q['elapsed_seconds'])
                check(mm['diagnostics']['deadline_reached'] is False,tag+':'+prefix+'_global_deadline')
        results.append(dict(job=job,status='complete',both_accepted=accepted,train_nrmse=errors['train'],validation_nrmse=errors['validation'],observed_nrmse=errors['test'],clean_nrmse=errors['clean'],mean_clean_nrmse=float(np.mean(errors['clean'])),structure=shape,mechanism=mechanism,actual_closure=stats['closure'],actual_supplement=stats['supplement'],supplement_record=supplement,evaluations=model['diagnostics']['evaluations'],fit_elapsed=model['diagnostics']['elapsed_seconds'],cell_wall=res['wall_seconds'],saved_candidates=len(model['candidates'])))
        print('reviewed',tag,flush=True)
    check(identity_rows==diagnosis['rows'],'identity_diagnosis_full_fields')
    check(all(min(vv)>0 for vv in margins.values()),'phase_deadlines_not_binding')
    index={key(r['job']):r for r in results};pairs=[]
    for row in summary['pairs']:
        case,seed=row['case'],row['seed'];a,b=index[case,seed,'completion'],index[case,seed,'base'];ma,mb=models[case,seed,'completion'],models[case,seed,'base'];ra,rb=(x['diagnostics']['sharing_closure'] for x in (ma,mb));pa,pb=([payload(c) for c in x['candidates'][:r['base_candidate_count']]] for x,r in ((ma,ra),(mb,rb)))
        identity=pa==pb and all(ra[k]==rb[k] for k in ('base_candidate_count','base_candidate_sha256','base_evaluations','base_search_fingerprint'))
        pair=dict(case=case,seed=seed,available=True,protected=(np.asarray(a['clean_nrmse'])<=np.asarray(b['clean_nrmse'])*1.1+1e-8).tolist(),candidate_accepted=a['both_accepted'],control_accepted=b['both_accepted'],base_identity=identity)
        check(pair==row,'summary_pair:'+case+str(seed));pairs.append(pair)
    means={}
    for method in ('base','completion'):
        rs=[r for r in results if r['job']['method']==method];means[method]=dict(expected=17,complete=17,failed_or_missing=0,both_accepted=sum(r['both_accepted'] for r in rs),mean_clean_nrmse=float(np.mean([r['clean_nrmse'] for r in rs])),mean_shared_graph_size=float(np.mean([r['structure']['shared_graph_size'] for r in rs])))
        check(all(means[method][k]==summary['methods'][method][k] if not isinstance(value,float) else np.isclose(value,summary['methods'][method][k],rtol=1e-12,atol=1e-14) for k,value in means[method].items()),'summary_methods:'+method)
    actual_matching={method:{k:sum(r['actual_closure'][k] for r in results if r['job']['method']==method) for k in ('new_evaluations','valid','protected','validation_retained')} for method in ('base','completion')}
    original_matching={method:{k:sum(row.get('closure',{}).get(k,0) for row in v['rows'] if row['job']['method']==method) for k in ('new_evaluations','valid','protected','validation_retained')} for method in ('base','completion')}
    check(original_matching==summary['closure_by_method'],'original_summary_lost_control_counts')
    supplemental=dict(protected_outputs=sum(sum(p['protected']) for p in pairs),foundation_pairs=sum(p['base_identity'] for p in pairs),control_acceptance_preserved=all(not p['control_accepted'] or p['candidate_accepted'] for p in pairs),macro_nonincrease=means['completion']['mean_clean_nrmse']<=means['base']['mean_clean_nrmse'],graph_nonincrease=means['completion']['mean_shared_graph_size']<=means['base']['mean_shared_graph_size'],actual_matching=actual_matching)
    check(supplemental['protected_outputs']==summary['protected_outputs']==34 and supplemental['foundation_pairs']==17 and supplemental['control_acceptance_preserved']==summary['control_acceptance_preserved'],'summary_protection')
    check(summary['technical_pass'] is False and summary['adoption_pass'] is False and summary['archived_control_identity_pass'] is False and summary['closure_mechanism_pass'] is False,'preserve_original_failed_gates')
    targets={}
    for case,seed,threshold,limit in [('damped_n0',701,1e-12,20),('rational_n0',702,1e-4,15)]:
        r=index[case,seed,'completion'];targets[f'{case}_s{seed}']=r['both_accepted'] and max(r['train_nrmse']+r['validation_nrmse']+r['clean_nrmse'])<=threshold and r['structure']['shared_graph_size']<=limit and r['structure']['independent_parameters']<=4
    check(targets==summary['historical_targets'],'historical_targets')
    failed=[r for r in results if r['job']['method']=='completion' and not r['both_accepted']]
    check([r['job'] for r in failed]==[r['job'] for r in summary['completion_failures']] and len(failed)==4,'four_candidate_failures')
    family={}
    for name,expected in summary['completion_discovery_by_family'].items():
        rs=[r for r in results if r['job']['method']=='completion' and f['specs'][r['job']['case']]['family']==name]
        tally={}
        for field in expected:
            if field in ('expected_systems','acceptance_gains'):continue
            if field=='validation_retained':tally[field]=sum(r['actual_supplement']['validation_retained'] for r in rs)
            elif field=='triggered':tally[field]=sum(int(r['supplement_record']['triggered']) for r in rs)
            elif field in ('new_evaluations','valid','protected'):tally[field]=sum(r['actual_supplement'][field] for r in rs)
            elif field=='compositional_validation_retained':
                tally[field]=sum(sum(bool(e['compositional_outputs']) and e['protected'] and np.isfinite(helpers.raw(e['child']['system'],e['child']['constants'],arrays(r['job']['case'],r['job']['seed'])['validation'][0])).all() for e in r['supplement_record']['events']) for r in rs)
            elif field.endswith('_event_roots'):
                field_kind=field.split('_')[0];tally[field]=sum(sum(len(e[field_kind+'_outputs']) for e in r['supplement_record']['events']) for r in rs)
            elif field in ('compositional_new_systems','compositional_valid_systems','compositional_protected_systems'):
                tally[field]=sum(sum((e['new_evaluation'] if field.endswith('_new_systems') else e['child'] is not None if field.endswith('_valid_systems') else e['protected']) for e in r['supplement_record']['events'] if e['compositional_outputs']) for r in rs)
            else:tally[field]=sum(r['supplement_record'][field] for r in rs)
        ps=[p for p in pairs if f['specs'][p['case']]['family']==name];tally['expected_systems']=len(ps);tally['acceptance_gains']=sum(p['candidate_accepted'] and not p['control_accepted'] for p in ps)
        check(tally==expected,'family_discovery:'+name);family[name]=tally
    integrity['final_live_mismatches']=hashes(f['files'],ROOT);integrity['final_archive_mismatches']=hashes(f['files'],HERE/'frozen_inputs');integrity['final_run_mismatches']=hashes(m['files'],HERE)
    development=ROOT/'experiments/completion_development_20261031';check(not any((development/n).exists() for n in ('PREFLIGHT.json','BUDGET.json','FREEZE.json','TRAINING_STARTED.json','ALL_MODELS_FROZEN.json')),'development_not_started');check(not (HERE/'POST_REVIEW_ACCEPTANCE.json').exists(),'no_false_acceptance_receipt')
    result=dict(status='review_consistent_with_original_failure' if not issues else 'review_findings',issues=issues,scope='Independent data/selected graph/identity-difference/event-protection/statistical reconstruction. Native load/domain reused; control closure replay reused frozen verifier only as supplemental evidence. No official verification overwrite, fit, search, initializer or full-pool error recomputation.',key_sha256={n:sha(HERE/n) for n in ('FREEZE.json','ALL_MODELS_FROZEN.json','SCORES.json','VERIFICATION.json','SUMMARY.json','EXECUTION.json','FROZEN_INPUT_ARCHIVE.json','PROTOCOL.md','HARNESS_REVIEW.md')},identity_diagnosis_sha256=sha(IDENTITY/'DIAGNOSIS.json'),identity_diagnosis_script_sha256=sha(IDENTITY/'diagnose.py'),helper_sha256=sha(helper),script_sha256=sha(Path(__file__)),integrity=integrity,timestamps=timestamps,training_validation_hashes=data_hashes,selected_systems_recomputed=34,native_estimator_reloads=34,native_physical_reloads=34,native_domain_checks=34,maximum_absolute_differences=dict(maxdiff),identity_rows=identity_rows,phase_deadline_margins={k:dict(count=len(vv),minimum=min(vv),maximum=max(vv)) for k,vv in margins.items()},systems=results,event_counts=dict(event_count),methods_recomputed=means,pairs_recomputed=pairs,original_summary_matching=original_matching,supplemental_observations_not_official_reclassification=supplemental,historical_targets_recomputed=targets,family_discovery_recomputed=family,saved_candidate_count=sum(r['saved_candidates'] for r in results),full_candidate_metrics_independently_recomputed=False,maximum_cell_wall=max(r['cell_wall'] for r in results),maximum_fit_elapsed=max(r['fit_elapsed'] for r in results),maximum_evaluations=max(r['evaluations'] for r in results),original_technical_pass=False,original_adoption_pass=False,permits_development=False)
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else (_ for _ in ()).throw(TypeError(type(x).__name__)))+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],issues=issues,methods=means,supplemental=supplemental,event_counts=dict(event_count),maxdiff=dict(maxdiff),receipt_sha256=sha(OUT))))


if __name__=='__main__':main()
