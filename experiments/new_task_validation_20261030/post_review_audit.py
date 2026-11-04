"""Read-only post-run reconstruction; writes only post_review_checks.json.

No experiment entry points, initializers, fitting, search or scoring modules are
called. Data, raw graph interpretation, units, shapes and summary are rebuilt
here. Native load/domain methods are used only as additional compatibility checks.
"""
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'post_review_checks.json'
FAMILIES = ('logarithmic_gain', 'logistic_gain', 'algebraic_norm', 'bilinear_saturation')
CASES = tuple(f'{f}_n{n}' for f in FAMILIES for n in (0, 1))
SEEDS = (9131, 9132)
METHODS = ('base', 'repair')
FIELDS = ('system', 'constants', 'errors', 'complexity', 'rademacher', 'rademacher_complexity')
STRUCTURE = ('shared_graph_size', 'independent_parameters', 'expanded_total_nodes', 'maximum_output_depth')
ZERO = (Fraction(0),) * 7
findings = []
maxdiff = Counter()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=False).encode()).hexdigest()


def ahash(*arrays):
    return hashlib.sha256(b''.join(np.asarray(a, dtype='<f8').tobytes() for a in arrays)).hexdigest()


def check(condition, label):
    if not condition:
        findings.append(label)


def close(actual, expected, label):
    a, b = np.asarray(actual, float), np.asarray(expected, float)
    check(a.shape == b.shape, label + ':shape')
    if a.shape == b.shape:
        delta = float(np.max(np.abs(a-b))) if a.size else 0.
        maxdiff[label.split(':')[0]] = max(maxdiff[label.split(':')[0]], delta)
        check(np.isfinite(a).all() and np.isfinite(b).all() and np.allclose(a,b,rtol=1e-9,atol=1e-11), label)


def verify_files(files, root, label):
    bad = []
    for n, h in files.items():
        path = (root/n).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file() or sha(path) != h:
            bad.append(n)
    check(not bad, label)
    return bad


def data(name, seed):
    """Rebuild the frozen RNG streams without importing the task generator."""
    i = CASES.index(name)
    def inputs(split):
        code, n = {'train':(111,96),'validation':(222,48),'test':(333,192)}[split]
        rng = np.random.default_rng(np.random.SeedSequence([20261030,seed,code,i]))
        return rng.uniform(np.array([.2,.2]),np.array([1.4,1.4]),(n,2))
    def latent(X):
        a,b = X.T
        f = name.rsplit('_n',1)[0]
        if f == 'logarithmic_gain':
            z=np.log(1+(.9*a)**2);return np.column_stack((z,b*z))
        if f == 'logistic_gain':
            z=1+np.exp(-.8*a);return np.column_stack((b/z,b*b/z))
        if f == 'algebraic_norm':
            z=np.sqrt(a*a+b*b);return np.column_stack((z,a/z))
        z=1+.6*a*b;return np.column_stack((a/z,b/z))
    clean_train=latent(inputs('train'));noise_sd=clean_train.std(axis=0)*(.01 if name.endswith('_n1') else 0.)
    result={}
    for split in ('train','validation','test'):
        X=inputs(split);clean=latent(X)
        rng=np.random.default_rng(np.random.SeedSequence([20261030,seed,{'train':411,'validation':422,'test':433}[split],i]))
        result[split]=(X,clean+noise_sd*rng.normal(size=clean.shape),clean)
    return result


def raw(graph, constants, X):
    values=[]
    binary={'add':np.add,'sub':np.subtract,'mul':np.multiply,'div':np.divide}
    unary={'neg':np.negative,'square':np.square,'sqrt':np.sqrt,'sin':np.sin,'cos':np.cos,'exp':np.exp,'log':np.log}
    with np.errstate(all='ignore'):
        for n in graph['nodes']:
            if n['parameter'] is not None:
                z=np.full(len(X),constants[n['parameter']],dtype=float)
            elif not n['children']:
                z=X[:,int(n['token'][1:])]
            else:
                args=[values[c] for c in n['children']]
                z=binary[n['token']](*args) if len(args)==2 else unary[n['token']](args[0])
                z=np.where(np.isfinite(z)&np.logical_and.reduce([np.isfinite(a) for a in args]),z,np.nan)
            values.append(z)
    return np.column_stack([values[r] for r in graph['roots']])


def error(pred, truth, scales):
    return np.sqrt(np.mean((pred-truth)**2,axis=0))/scales


def graph_audit(graph, spec, label):
    nodes=graph['nodes'];units=[];sizes=[];operations=[];depths=[];reach=[];params={}
    decode=lambda u:tuple(Fraction(v) for v in u)
    for i,n in enumerate(nodes):
        children=n['children'];token=n['token'];u=decode(n['units'])
        check(all(type(c) is int and 0<=c<i for c in children),label+':topology')
        ds=[units[c] for c in children]
        if not children:
            table=spec['features'] if token.startswith('x') else spec['constants']
            check(u==decode(table[int(token[1:])]),label+':terminal_units')
            expected=u
            if token.startswith('c'):
                p=n['parameter'];check(type(p) is int and p>=0,label+':parameter')
                check(p not in params or params[p]==u,label+':parameter_units');params[p]=u
            else:check(n['parameter'] is None,label+':input_parameter')
        elif token in ('add','sub'):
            check(len(ds)==2 and ds[0]==ds[1],label+':add_units');expected=ds[0]
        elif token in ('mul','div'):
            expected=tuple(a+b if token=='mul' else a-b for a,b in zip(*ds))
        elif token in ('square','sqrt','neg'):
            expected=tuple(v*{'square':Fraction(2),'sqrt':Fraction(1,2),'neg':Fraction(1)}[token] for v in ds[0])
        else:
            check(len(ds)==1 and ds[0]==ZERO,label+':transcendental_units');expected=ZERO
        check(u==expected,label+':operator_units');units.append(u)
        sizes.append(1+sum(sizes[c] for c in children));operations.append(bool(children)+sum(operations[c] for c in children));depths.append(1+max([depths[c] for c in children],default=0))
        reach.append({i}.union(*(reach[c] for c in children)))
    roots=graph['roots'];check(len(roots)==2,label+':root_count')
    check(set(params)==set(range(len(params))),label+':parameter_ids')
    check(set.union(*(reach[r] for r in roots))==set(range(len(nodes))),label+':unused_nodes')
    check([units[r] for r in roots]==[decode(u) for u in spec['targets']],label+':target_units')
    outputs=[]
    for r in roots:
        outputs.append(dict(expanded_nodes=sizes[r],expanded_operations=operations[r],depth=depths[r],unique_dag_nodes=len(reach[r]),independent_parameters=len({nodes[i]['parameter'] for i in reach[r] if nodes[i]['parameter'] is not None})))
    shape=dict(shared_dag_vertices=len(nodes),shared_graph_size=sum(n['parameter'] is None for n in nodes)+len(params),shared_dag_operations=sum(bool(n['children']) for n in nodes),maximum_output_depth=max(depths[r] for r in roots),independent_parameters=len(params),expanded_total_nodes=sum(sizes[r] for r in roots),expanded_total_operations=sum(operations[r] for r in roots),outputs=outputs)
    shared=dict(shared_nodes=sum(sum(i in reach[r] for r in roots)>1 for i in range(len(nodes))),shared_parameters=sum(sum(any(nodes[i]['parameter']==p for i in reach[r]) for r in roots)>1 for p in params),unique_parameters=len(params),dag_size=shape['shared_graph_size'])
    check(all(row['expanded_nodes']<=15 and row['depth']<=6 for row in outputs),label+':search_bounds')
    return shape,shared


def payload(c):
    return {k:c[k] for k in FIELDS}


def main():
    if OUT.exists():
        raise RuntimeError('Do not overwrite an existing review checkpoint.')
    frozen=read(HERE/'FREEZE.json');sealed=read(HERE/'ALL_MODELS_FROZEN.json');scores=read(HERE/'SCORES.json');verification=read(HERE/'VERIFICATION.json');summary=read(HERE/'SUMMARY.json');archive=read(HERE/'FROZEN_INPUT_ARCHIVE.json');lock=read(HERE/'CANDIDATE_LOCK.json');execution=read(HERE/'EXECUTION.json')
    expected_jobs=[dict(case=c,seed=s,method=m,seconds=60) for c in CASES for s in SEEDS for m in (METHODS if s%2 else METHODS[::-1])]
    check(frozen['jobs']==expected_jobs,'fixed_matrix')
    check(len(frozen['files'])==archive['file_count']==715 and frozen['files']==archive['files'],'input_manifest')
    integrity=dict(input_count=len(frozen['files']),live_input_mismatches=verify_files(frozen['files'],ROOT,'live_inputs'),archive_mismatches=verify_files(frozen['files'],HERE/'frozen_inputs','archived_inputs'),run_count=len(sealed['files']),run_mismatches=verify_files(sealed['files'],HERE,'run_artifacts'))
    check(len(sealed['files'])==archive['runs_unchanged']==352,'run_count')
    actual_runs={p.relative_to(HERE).as_posix() for p in (HERE/'runs').rglob('*') if p.is_file()};check(actual_runs==set(sealed['files']),'run_manifest_completeness')
    check(len(lock['source_files'])==40 and set(lock['source_files'])=={p.relative_to(ROOT).as_posix() for p in (ROOT/'src').rglob('*.py')},'source_manifest')
    integrity['source_mismatches']=verify_files(lock['source_files'],ROOT,'source_lock')
    check(all(frozen['files'][n]==h==sha(HERE/'source'/n) for n,h in lock['source_files'].items()),'source_copies')
    check(archive['freeze_sha256']==sealed['freeze_sha256']==verification['freeze_sha256']==sha(HERE/'FREEZE.json'),'freeze_chain')
    check(scores['all_models_sha256']==verification['models_sha256']==sha(HERE/'ALL_MODELS_FROZEN.json'),'models_chain')
    check(verification['scores_sha256']==sha(HERE/'SCORES.json'),'score_chain')
    times=dict(freeze=frozen['frozen_unix'],training=read(HERE/'TRAINING_STARTED.json')['start_unix'],all_models=sealed['frozen_unix'],scores=scores['scored_unix'],verification=verification['verified_unix'])
    check(list(times.values())==sorted(times.values()),'global_chronology');check(not frozen['test_labels_accessed'] and not sealed['test_labels_accessed'],'test_access_flags')
    check([r['job'] for r in scores['rows']]==[r['job'] for r in verification['rows']]==expected_jobs,'result_denominators')
    check(len({tuple(j.values()) for j in expected_jobs})==32,'unique_jobs')
    budget=read(HERE/'BUDGET.json');check(budget==frozen['budget'],'budget_freeze')
    check(all(budget[k]==v for k,v in dict(limit_seconds=2880,sequential_workers=1,threads=1,system_cap_seconds=60,fit_seconds=52,verification_reserve_seconds=600).items()),'budget_limits')
    check(execution['returncode']==0 and execution['termination_reason'] is None and execution['verification_passed'],'execution_terminal')
    check(verification['elapsed_seconds']<2880 and execution['wall_seconds']<2880 and verification['budget_met'],'elapsed_budget')
    check(verification['status']=='passed' and verification['verified_systems']==32 and verification['preparation_error'] is None,'verification_terminal')
    check(not (HERE/'SCORING_FAILURE.json').exists(),'no_scoring_failure')
    tv=read(HERE/'TEST_VALIDATION.json');integrity['test_log_mismatches']=verify_files(tv['logs'],HERE/'validation','test_logs');check(tv['source_sha256']==lock['source_files'],'test_source_lock')
    novelty=read(HERE/'NOVELTY_AUDIT.json');integrity['novelty_archive_mismatches']=verify_files({r['path']:r['sha256'] for r in novelty['files']},HERE/'history_sources','history_sources')
    old=ROOT/'experiments/precision_repair_20261028';check(sha(old/'SUMMARY.json')==lock['source_summary_sha256'],'prior_summary_lock')
    results=[];models={};event_rows=[];paired=[];previous_time=times['training'];data_checks=[]
    # Native loading is permitted here; no fit/search or official verifier imports.
    from dimensr_mimo import DimenSRMIMORegressor,PhysicalModel
    for job,score,vr in zip(expected_jobs,scores['rows'],verification['rows']):
        tag=f"{job['case']}_s{job['seed']}_{job['method']}";folder=HERE/'runs'/tag
        cell=read(folder/'CELL_FROZEN.json');check(previous_time<=cell['frozen_unix']<=times['all_models'],tag+':cell_time');previous_time=cell['frozen_unix']
        verify_files(cell['files'],folder,tag+':cell_hashes')
        check(sealed['files'][(folder/'CELL_FROZEN.json').relative_to(HERE).as_posix()]==sha(folder/'CELL_FROZEN.json'),tag+':cell_membership')
        check(all(sealed['files'][(folder/n).relative_to(HERE).as_posix()]==h for n,h in cell['files'].items()),tag+':artifact_membership')
        check(set(cell['files'])=={p.name for p in folder.iterdir() if p.is_file() and p.name!='CELL_FROZEN.json'},tag+':cell_manifest_complete')
        model=read(folder/'model.json');models[job['case'],job['seed'],job['method']]=model
        result=read(folder/'result.json');resource=read(folder/'resources.json');bundle=read(folder/'delivery.json');spec=frozen['specs'][job['case']]
        check(read(folder/'job.json')==result['job']==resource['job']==job,tag+':job_identity')
        check(result['status']==resource['status']==score['status']=='complete' and vr['verified'] and resource['returncode']==0,tag+':completion')
        check(resource['wall_seconds']<=60.5 and model['diagnostics']['elapsed_seconds']<=52.5 and model['diagnostics']['evaluations']<=1200,tag+':cell_budget')
        expected_cfg=dict(frozen['config'],random_state=job['seed'],initialization='compound',sharing_strategy='closure',precision_repair=job['method']=='repair',final_selection='precision',acceptance=spec['acceptance'])
        check(model['config']==result['effective_config']==expected_cfg,tag+':effective_config')
        check(not result['test_scores_computed'],tag+':test_isolation_flag')
        d=data(job['case'],job['seed']);X,Y,_=d['train'];V,Z,_=d['validation'];T,U,clean=d['test'];scales=Y.std(axis=0)
        dh=ahash(X,Y,V,Z);check(dh==frozen['data_sha256'][f"{job['case']}_s{job['seed']}"]==result['data_sha256'],tag+':data_hash')
        close(model['scales'],scales,'scales:'+tag);check(X.shape==Y.shape==(96,2) and V.shape==Z.shape==(48,2) and T.shape==U.shape==(192,2),tag+':data_shapes')
        if job['method']=='base':data_checks.append(dict(case=job['case'],seed=job['seed'],sha256=dh))
        graph=model['best']['system'];constants=model['best']['constants'];shape,sharing=graph_audit(graph,spec,tag)
        check(shape==bundle['structure']==score['structure']==vr['structure'],tag+':structure');check(sharing==bundle['sharing']==score['sharing'],tag+':sharing')
        check(graph==bundle['system'] and constants==[p['value'] for p in bundle['parameters']],tag+':delivery_graph')
        pred={s:raw(graph,constants,d[s][0]) for s in d};errs={s:error(pred[s],d[s][1],scales).tolist() for s in d};errs['clean']=error(pred['test'],clean,scales).tolist()
        close(errs['train'],model['best']['errors'],'selected_train:'+tag);close(errs['train'],bundle['train_nrmse'],'selected_train:'+tag);close(errs['validation'],model['validation_nrmse'],'selected_validation:'+tag);close(errs['validation'],bundle['validation_nrmse'],'selected_validation:'+tag);close(errs['test'],score['observed_nrmse'],'selected_observed:'+tag);close(errs['clean'],score['clean_nrmse'],'selected_clean:'+tag)
        check(result['prediction_hashes']==[dict(train=ahash(pred['train']),validation=ahash(pred['validation']))],tag+':prediction_hashes')
        loaded=DimenSRMIMORegressor.load(folder/'model.json');physical=PhysicalModel.load(folder/'physical.json')
        for split in d:
            check(np.array_equal(loaded.predict(d[split][0]),pred[split]),tag+':native_reload_'+split)
            check(np.array_equal(physical.predict(d[split][0]),pred[split]),tag+':physical_reload_'+split)
        domain=loaded.audit_domain(spec['bounds']);check([domain]==result['audits']==bundle['domain_audits'] and domain['status']=='enclosed',tag+':domain_reload')
        events=[json.loads(line) for line in (folder/'events.jsonl').read_text(encoding='utf-8').splitlines()]
        relevant=[r for r in events if r['output'] is None]
        mech=dict(policy_delta=sum(r.get('parameter_delta',0) for r in relevant),neural_valid=sum(r.get('neural_valid',0) for r in relevant),gp_new_evaluations=sum(r.get('new_evaluations',0) for r in relevant if r['phase']=='gp'),nsga_generations=sum(r['phase']=='nsga2' for r in relevant),nsga_new_evaluations=sum(r.get('new_evaluations',0) for r in relevant if r['phase']=='nsga2'))
        scmech=score['mechanism'];check(scmech['outputs']==[dict(output=None,**mech)],tag+':mechanism_events')
        nsga_steps={r['step'] for r in relevant if r['phase']=='nsga2' and r['new_evaluations']>0}
        check(all(v>0 for v in mech.values()) and len(nsga_steps)>=2 and model['diagnostics']['rademacher_systems']>0 and scmech['full_mechanism_pass'],tag+':mechanism_gate')
        check(all(all(e.get(k)==v for k,v in h.items()) for h in model['history'] for e in events if e['phase']==h['phase'] and e['step']==h['step']),tag+':history_events')
        record=model['diagnostics']['sharing_closure'];base_payload=[payload(c) for c in model['candidates'][:record['base_candidate_count']]]
        check(digest(base_payload)==record['base_candidate_sha256'],tag+':base_payload_hash')
        known={digest(c) for c in base_payload};all_new_graphs=set();retained_by_kind={};stats_by_kind={}
        repair=record.get('precision_repair');eligible=sum(all(a<=b for a,b in zip(c['errors'],spec['acceptance']['nrmse'])) for c in base_payload)
        check(job['method']=='repair' or repair is None,tag+':base_no_repair')
        if repair:
            check(repair['training_eligible']==eligible and repair['triggered']==(eligible==0) and repair['seed_limit']==12,tag+':trigger')
            fixed_parent=min(base_payload,key=lambda c:(float(np.mean(c['errors'])),c['complexity'],digest(c['system'])))
            if repair['triggered']:check(repair['parent']==fixed_parent,tag+':repair_parent')
            else:check(not repair['events'] and not repair['initialization_diagnostics'] and repair['stop_reason']=='training_precision_met',tag+':repair_skip')
        for kind,rec in [('repair',repair),('closure',record)]:
            kept=[];valid=0;new=0;validation_kept=0
            if rec:
                for event in rec['events']:
                    check(digest(event['parent']) in known,tag+':'+kind+'_parent_provenance')
                    if kind=='repair':check(event['parent']==fixed_parent,tag+':fixed_parent')
                    gh=digest(event['graph']);check(gh not in all_new_graphs,tag+':unique_extension_graph');all_new_graphs.add(gh)
                    shape_new,_=graph_audit(event['graph'],spec,tag+':'+kind+'_graph')
                    check(event['new_evaluation'],tag+':'+kind+'_new_evaluation');new+=int(event['new_evaluation'])
                    if kind=='closure':
                        ps,_=graph_audit(event['parent']['system'],spec,tag+':closure_parent')
                        check(shape_new['shared_graph_size']<ps['shared_graph_size'],tag+':closure_reduces_size')
                    else:
                        ids=event['implicit_outputs'];k=event['seed_index'];init=rec['initialization_diagnostics'];expected_implicit=[j for j,row in enumerate(init) if k<min(12,row['proposed_structures']) and k in row['implicit_retained_indices']]
                        check(ids==expected_implicit and bool(ids),tag+':implicit_root_provenance')
                    parent_errors=error(raw(event['parent']['system'],event['parent']['constants'],X),Y,model['scales'])
                    close(parent_errors,event['parent']['errors'],'event_parent:'+tag)
                    child=event['child'];protected=False
                    if child is not None:
                        valid+=1;check(child['system']==event['graph'],tag+':child_graph')
                        child_errors=error(raw(child['system'],child['constants'],X),Y,model['scales']);close(child_errors,child['errors'],'event_child:'+tag)
                        protected=bool(np.all(child_errors<=parent_errors*1.1+1e-8))
                        if protected:kept.append(child);known.add(digest(child))
                    check(protected==event['protected'],tag+':'+kind+'_protection')
                    event_rows.append(dict(job=job,kind=kind,new_evaluation=event['new_evaluation'],valid=child is not None,protected=protected))
                check(new==rec['new_evaluations'] and valid==rec['valid'] and len(kept)==rec['protected'] and valid-len(kept)==rec['protection_rejections'],tag+':'+kind+'_counts')
                check(new<=(12 if kind=='repair' else 48),tag+':'+kind+'_budget')
            retained_by_kind[kind]=kept
            accepted=[];rejected=[]
            for child in kept:
                predv=raw(child['system'],child['constants'],V);ev=error(predv,Z,model['scales']);ok=bool(np.isfinite(predv).all() and np.isfinite(ev).all())
                (accepted if ok else rejected).append(child)
            stats_by_kind[kind]=dict(new_evaluations=new,valid=valid,protected=len(kept),validation_retained=len(accepted))
            retained_by_kind[kind+'_validated']=accepted;retained_by_kind[kind+'_rejected']=rejected
        check(0<=record['rounds']<=2 and all(0<=e['round']<record['rounds'] for e in record['events']),tag+':closure_rounds')
        check(all(sum(e['new_evaluation'] for e in record['events'] if e['round']==r)<=24 for r in range(2)),tag+':round_caps')
        check(record['base_evaluations']+stats_by_kind['repair']['new_evaluations']+stats_by_kind['closure']['new_evaluations']==model['diagnostics']['evaluations'],tag+':total_accounting')
        check(digest(base_payload+retained_by_kind['repair']+retained_by_kind['closure'])==record['full_training_pool_sha256'],tag+':full_training_pool')
        check([payload(c) for c in model['candidates'][len(base_payload):]]==retained_by_kind['repair_validated']+retained_by_kind['closure_validated'],tag+':extension_validation_subset')
        check(record['validation_retained_extensions']==sum(stats_by_kind[k]['validation_retained'] for k in ('repair','closure')),tag+':validation_count')
        rejected=retained_by_kind['repair_rejected']+retained_by_kind['closure_rejected'];check([r['graph_sha256'] for r in record.get('validation_rejections',[])]==[digest(c['system']) for c in rejected],tag+':validation_rejections')
        hist=next(h for h in model['history'] if h['phase']=='sharing_closure');check(hist['new_evaluations']==stats_by_kind['repair']['new_evaluations']+stats_by_kind['closure']['new_evaluations'] and hist['new_systems']==stats_by_kind['closure']['new_evaluations'] and hist['protected_systems']==stats_by_kind['closure']['protected'],tag+':closure_history')
        closure=dict(stats_by_kind['closure'],base_candidate_count=record['base_candidate_count'],base_evaluations=record['base_evaluations'],base_search_fingerprint=record['base_search_fingerprint'],base_candidate_sha256=record['base_candidate_sha256'])
        rep={}
        if repair:
            init=repair['initialization_diagnostics'];rep=dict(triggered=int(repair['triggered']),training_eligible=eligible,**stats_by_kind['repair'],linear_subproblems=sum(r.get('implicit_linear_subproblems',0) for r in init),retained_roots=sum(len(r.get('implicit_retained_indices',[])) for r in init),implicit_new_systems=sum(bool(e['implicit_outputs']) and e['new_evaluation'] for e in repair['events']))
        check(closure==vr['closure'] and rep==vr['repair'],tag+':verified_extension_counters')
        good_shape=all(shape[k]<=spec['acceptance'][k] for k in STRUCTURE)
        dev=good_shape and all(np.all(np.asarray(errs[s])<=spec['acceptance']['nrmse']) for s in ('train','validation'))
        test=good_shape and np.all(np.asarray(errs['test'])<=spec['acceptance']['nrmse']) and np.all(np.asarray(errs['clean'])<=spec['clean_test_nrmse'])
        accepted=bool(dev and test and model['post_selection']['status']=='accepted')
        check(accepted==score['both_accepted'] and bool(dev)==bundle['development_accepted'] and bool(test)==score['test_accepted'],tag+':absolute_acceptance')
        failures=[]
        for split in ('train','validation','test','clean'):
            limits=spec['clean_test_nrmse'] if split=='clean' else spec['acceptance']['nrmse']
            for j,(v,limit) in enumerate(zip(errs[split],limits)):
                if v>limit:failures.append(dict(gate=split,output=j,value=v,limit=limit))
        for k in STRUCTURE:
            if shape[k]>spec['acceptance'][k]:failures.append(dict(gate=k,value=shape[k],limit=spec['acceptance'][k]))
        results.append(dict(job=job,verified=True,status='complete',both_accepted=accepted,mean_clean_nrmse=float(np.mean(errs['clean'])),clean_nrmse=errs['clean'],observed_nrmse=errs['test'],train_nrmse=errs['train'],validation_nrmse=errs['validation'],structure=shape,closure=closure,repair=rep,failures=failures,mechanism=mech,nsga_steps_with_new=len(nsga_steps),wall_seconds=resource['wall_seconds'],fit_seconds=model['diagnostics']['elapsed_seconds'],evaluations=model['diagnostics']['evaluations'],saved_candidate_count=len(model['candidates'])))
        print('reviewed',tag,flush=True)
    index={(r['job']['case'],r['job']['seed'],r['job']['method']):r for r in results}
    for case in CASES:
        for seed in SEEDS:
            a,b=(index[case,seed,m] for m in ('repair','base'));ma,mb=(models[case,seed,m] for m in ('repair','base'));ra,rb=(m['diagnostics']['sharing_closure'] for m in (ma,mb))
            base_match=all(ra[k]==rb[k] for k in ('base_candidate_count','base_candidate_sha256','base_evaluations','base_search_fingerprint'))
            skipped=not ra['precision_repair']['triggered'];pool=ma['candidates']==mb['candidates'] and ma['search_fingerprint']==mb['search_fingerprint'];choice=ma['best']==mb['best'] and ma['validation_nrmse']==mb['validation_nrmse']
            paired.append(dict(case=case,seed=seed,available=True,protected=(np.asarray(a['clean_nrmse'])<=np.asarray(b['clean_nrmse'])*1.1+1e-8).tolist(),candidate_accepted=a['both_accepted'],control_accepted=b['both_accepted'],base_prefix_matches=base_match,base_cache_matches=ra['base_search_fingerprint']==rb['base_search_fingerprint'],repair_skipped=skipped,skipped_pool_matches=pool if skipped else None,skipped_choice_matches=choice if skipped else None))
    def aggregate(rows):
        return dict(expected=len(rows),complete=len(rows),failed_or_missing=0,both_accepted=sum(r['both_accepted'] for r in rows),mean_clean_nrmse=float(np.mean([np.mean(r['clean_nrmse']) for r in rows])),mean_shared_graph_size=float(np.mean([r['structure']['shared_graph_size'] for r in rows])))
    method_summary={}
    for m in METHODS:
        rs=[r for r in results if r['job']['method']==m];method_summary[m]=dict(aggregate(rs),statuses={'complete':16},strata={})
        for label in FAMILIES+('noise0','noise1'):
            subset=[r for r in rs if (r['job']['case'].endswith('_n'+label[-1]) if label.startswith('noise') else r['job']['case'].rsplit('_n',1)[0]==label)]
            method_summary[m]['strata'][label]=aggregate(subset)
    fields=('new_evaluations','valid','protected','validation_retained')
    closure={k:sum(r['closure'][k] for r in results) for k in fields};by_method={m:{k:sum(r['closure'][k] for r in results if r['job']['method']==m) for k in fields} for m in METHODS}
    repair={k:sum(r['repair'].get(k,0) for r in results) for k in ('triggered','new_evaluations','valid','protected','validation_retained','linear_subproblems','retained_roots','implicit_new_systems')}
    gains=sum(p['candidate_accepted'] and not p['control_accepted'] for p in paired);losses=sum(p['control_accepted'] and not p['candidate_accepted'] for p in paired)
    by_family={}
    for f in FAMILIES:
        rs=[r for r in results if r['job']['method']=='repair' and r['job']['case'].rsplit('_n',1)[0]==f];ps=[p for p in paired if p['case'].rsplit('_n',1)[0]==f]
        by_family[f]=dict({k:sum(r['repair'].get(k,0) for r in rs) for k in repair},expected_systems=4,verified_systems=4,acceptance_gains=sum(p['candidate_accepted'] and not p['control_accepted'] for p in ps))
    discovery=dict(repair,acceptance_gains=gains,acceptance_losses=losses,net_acceptance_gain=gains-losses,actual_implicit_evaluation=all(repair[k]>0 for k in ('triggered','linear_subproblems','retained_roots','implicit_new_systems','valid')),validation_retained_discovery=repair['validation_retained']>0,acceptance_gain_observed=gains>0,by_family=by_family,families_with_validation_retained=sum(r['validation_retained']>0 for r in by_family.values()),families_with_acceptance_gain=sum(r['acceptance_gains']>0 for r in by_family.values()),claim_scope='Actual events on this fixed matrix; zero triggers or retained candidates do not establish repair discovery benefit across families.')
    base_identity=all(p['base_prefix_matches'] and p['base_cache_matches'] for p in paired);skip_identity=all(not p['repair_skipped'] or p['skipped_pool_matches'] and p['skipped_choice_matches'] for p in paired);closure_gate=all(v>0 for r in by_method.values() for v in r.values());protected=sum(sum(p['protected']) for p in paired);preserved=all(not p['control_accepted'] or p['candidate_accepted'] for p in paired)
    a,b=method_summary['repair'],method_summary['base'];candidate_all=a['both_accepted']==16;macro=a['mean_clean_nrmse']<=b['mean_clean_nrmse'];graph=a['mean_shared_graph_size']<=b['mean_shared_graph_size'];technical=not findings and base_identity and skip_identity and closure_gate
    reconstructed=dict(methods=method_summary,pairs=paired,expected_systems=32,expected_pairs=16,expected_outputs=32,full_denominator=True,protected_outputs=protected,technical_pass=technical,closure=closure,closure_by_method=by_method,closure_mechanism_pass=closure_gate,repair=repair,repair_discovery_evidence=discovery,paired_base_identity_pass=base_identity,untriggered_identity_pass=skip_identity,candidate_all_accepted=candidate_all,control_acceptance_preserved=preserved,macro_clean_error_nonincrease=macro,mean_graph_nonincrease=graph,adoption_pass=bool(technical and candidate_all and protected==32 and preserved and macro and graph),production_default_changed=False,claim_scope='Descriptive validation of four fixed synthetic output-pair families; not statistical significance, public benchmark or real engineering acceptance.')
    differences=[]
    def compare(a,b,path='SUMMARY'):
        if isinstance(a,dict) and isinstance(b,dict):
            if set(a)!=set(b):differences.append(path+':keys')
            for k in set(a)&set(b):compare(a[k],b[k],path+'.'+k)
        elif isinstance(a,list) and isinstance(b,list):
            if len(a)!=len(b):differences.append(path+':length')
            for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+f'[{i}]')
        elif type(a) is float and type(b) in (float,int):
            if not np.isclose(a,b,rtol=1e-12,atol=1e-14):differences.append(path)
        elif a!=b:differences.append(path)
    compare(reconstructed,summary);check(not differences,'summary_reconstruction')
    # Recheck all protected bytes after numerical interpretation and native loads.
    integrity['final_live_mismatches']=verify_files(frozen['files'],ROOT,'final_live_inputs');integrity['final_archive_mismatches']=verify_files(frozen['files'],HERE/'frozen_inputs','final_archive');integrity['final_run_mismatches']=verify_files(sealed['files'],HERE,'final_runs')
    evidence=dict(status='passed' if not findings else 'findings',findings=findings,scope='Independent data RNG/formulas, raw JSON interpreter, selected units/shape, all extension parent-child errors/protection, full summary; native load/domain calls reuse locked production implementation. No fit/search/refit or official scorer/verifier run.',versions=dict(python=sys.version,numpy=np.__version__),key_sha256={n:sha(HERE/n) for n in ('FREEZE.json','ALL_MODELS_FROZEN.json','SCORES.json','VERIFICATION.json','SUMMARY.json','FROZEN_INPUT_ARCHIVE.json','EXECUTION.json','PLAN.md','PROTOCOL.md','PRE_RUN_REVIEW.md','HARNESS_REVIEW.md','TEST_VALIDATION.json')},integrity=integrity,timestamps=times,data_hashes=data_checks,maximum_absolute_differences=dict(maxdiff),summary_differences=differences,summary_reconstructed=reconstructed,systems=results,event_count=len(event_rows),event_counts_by_kind=dict(Counter(r['kind'] for r in event_rows)),selected_systems_recomputed=32,native_estimator_reloads=32,native_physical_reloads=32,native_domain_rechecks=32,saved_candidate_count=sum(r['saved_candidate_count'] for r in results),full_candidate_metrics_independently_recomputed=False,skip_pair_count=sum(p['repair_skipped'] for p in paired),maximum_cell_wall=max(r['wall_seconds'] for r in results),maximum_saved_fit_elapsed=max(r['fit_seconds'] for r in results),maximum_evaluations=max(r['evaluations'] for r in results),script_sha256=sha(Path(__file__)))
    OUT.write_text(json.dumps(evidence,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=evidence['status'],findings=findings,summary_differences=differences,event_count=len(event_rows),maximum_absolute_differences=dict(maxdiff),output_sha256=sha(OUT))))


if __name__ == '__main__':
    main()
