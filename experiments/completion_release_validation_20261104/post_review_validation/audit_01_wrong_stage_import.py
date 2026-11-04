"""Independent sealed new-output reconstruction. Never fits or rewrites scores."""
from collections import Counter
import hashlib,importlib.util,json,math,sys,time
from pathlib import Path
import numpy as np

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent.parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from dimensr_mimo import DimenSRMIMORegressor,PhysicalModel
from dimensr_mimo._single.expression import Node
from dimensr_mimo._single.units import unit
from dimensr_mimo.generation import SystemGrammar
from dimensr_mimo.graph import Builder,System
from dimensr_mimo.genetics import clone_subgraph
from experiments.completion_development_verified_20261031 import archive_io
from experiments.compositional_completion_20261031.verification import Verifier

OUT=HERE/'post_review_checks.json'
PRIOR=ROOT/'experiments/completion_development_verified_20261031'
SHARED=ROOT/'experiments/compositional_completion_20261031'
FIELDS=('system','constants','errors','complexity','rademacher','rademacher_complexity')
STRUCTURE=('shared_graph_size','independent_parameters','expanded_total_nodes','maximum_output_depth')
PHASE={'epoch_allowance_seconds','neural_until_seconds','gp_until_seconds'}
issues=[];maxdiff=Counter();source_count=Counter();event_count=Counter();source_rows=[]
read=lambda p:json.loads(archive_io.disk_path(p).read_text(encoding='utf-8'))
sha=archive_io.sha
digest=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True,allow_nan=False).encode()).hexdigest()
ahash=lambda *a:hashlib.sha256(b''.join(np.asarray(x,dtype='<f8').tobytes() for x in a)).hexdigest()
key=lambda j:(j['case'],j['seed'],j['method'])
payload=lambda c:{k:c[k] for k in FIELDS}
def check(ok,label):
    if not ok:issues.append(label)
def close(a,b,label):
    a,b=np.asarray(a,float),np.asarray(b,float)
    check(a.shape==b.shape,label+':shape')
    if a.shape==b.shape:
        delta=float(np.max(np.abs(a-b))) if a.size else 0.
        maxdiff[label.split(':')[0]]=max(maxdiff[label.split(':')[0]],delta)
        check(np.isfinite(a).all() and np.isfinite(b).all() and np.allclose(a,b,rtol=1e-9,atol=1e-11),label)
def hashes(files,root):
    bad=[n for n,h in files.items() if not archive_io.disk_path(root/n).is_file() or sha(root/n)!=h]
    check(not bad,'hashes:'+str(root));return bad
def load_helper(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def raw_tree(tree,constants,X):
    values=iter(constants)
    binary={'add':np.add,'sub':np.subtract,'mul':np.multiply,'div':np.divide}
    unary={'neg':np.negative,'square':np.square,'sqrt':np.sqrt,'log':np.log,'exp':np.exp,'sin':np.sin,'cos':np.cos}
    def visit(n):
        if not n['children']:return X[:,int(n['token'][1:])] if n['token'].startswith('x') else np.full(len(X),next(values))
        args=[visit(c) for c in n['children']]
        return binary[n['token']](*args) if len(args)==2 else unary[n['token']](args[0])
    with np.errstate(all='ignore'):return np.asarray(visit(tree),float)

def source_audit(job,model,base,supplement,X,Y,helpers):
    tag='_'.join(map(str,key(job)));stats=supplement['initialization_diagnostics'];parent=supplement['parent']
    cfg=__import__('dimensr_mimo').SearchConfig(**model['config'])
    grammar=SystemGrammar(tuple(map(unit,model['feature_units'])),tuple(map(unit,model['target_units'])),tuple(map(unit,model['constant_units'])),cfg)
    check(supplement['mode']=='compositional_completion_v1' and supplement['triggered'] is True and len(stats)==2,tag+':completion_mode')
    for j,row in enumerate(stats):
        sources=row.get('returned_sources',[]);check([s['returned_index'] for s in sources]==list(range(len(sources))),tag+':returned_positions')
        ranks=[]
        for s in sources:
            tree=Node.from_dict(s['tree']);constants=np.asarray(s['constants'],float)
            check(len(constants)==tree.n_constants and np.isfinite(constants).all(),tag+':source_constants')
            canonical,warm=tree.canonical(constants);check(tree==canonical and np.array_equal(warm,constants),tag+':canonical_source')
            grammar.plain[j].trace(tree)
            pred=raw_tree(s['tree'],constants,X)
            score=float(np.mean((pred-Y[:,j])**2))/max(float(np.var(Y[:,j])),1e-12)+1e-8*tree.size
            close(score,s['score'],'raw_Y_source_score:'+tag)
            ranks.append((s['score'],tree.size,repr(tree)));source_count[s['kind']]+=1
            source_rows.append(dict(job=job,output=j,returned_index=s['returned_index'],kind=s['kind'],tree_sha256=digest(s['tree']),recomputed_score=score,saved_score=s['score']))
            if s['kind']=='compositional':
                check(s['transform'] in ('log','sqrt') and s['form'] in ('direct','multiply','divide'),tag+':transform_kind')
                check(type(s['intercept']) is bool and type(s['outer_amplitude']) is bool and all(type(i) is int and 0<=i<24 for i in s['basis_indices']),tag+':transform_metadata')
                Verifier.outer_source(tree,s)
            else:
                check(s['original_index']==s['proposal_index'] and (s['kind']=='implicit')==(s['original_index'] in row.get('structured_implicit_retained_indices',[])),tag+':old_source_indices')
        check(ranks==sorted(ranks) and len({digest(s['tree']) for s in sources})==len(sources),tag+':source_ranking_uniqueness')
        for kind in ('implicit','compositional'):check(row.get(kind+'_retained_indices',[])==[i for i,s in enumerate(sources) if s['kind']==kind],tag+':source_retained_slots')
    for field in ('compositional_linear_subproblems','implicit_linear_subproblems'):check(supplement[field]==sum(r.get(field,0) for r in stats),tag+':'+field)
    check(supplement['linear_subproblems']==sum(r.get('linear_subproblems',0)+r.get('compositional_linear_subproblems',0) for r in stats),tag+':linear_total')
    for kind in ('implicit','compositional'):check(supplement[kind+'_returned_roots']==sum(s['kind']==kind for r in stats for s in r.get('returned_sources',[])),tag+':'+kind+'_root_total')
    indices=[e['seed_index'] for e in supplement['events']];check(indices==sorted(set(indices)),tag+':event_seed_positions')
    for e in supplement['events']:
        k=e['seed_index'];sources=[r['returned_sources'][k] if k<len(r['returned_sources']) else dict(kind='parent') for r in stats]
        check(sources==e['root_sources'],tag+':actual_root_slots')
        check(e['implicit_outputs']==[j for j,s in enumerate(sources) if s['kind']=='implicit'] and e['compositional_outputs']==[j for j,s in enumerate(sources) if s['kind']=='compositional'],tag+':event_root_kinds')
        b=Builder();columns=[]
        for j,s in enumerate(sources):
            if s['kind']=='parent':
                pg=System.from_dict(parent['system']);b.roots.append(clone_subgraph(b,pg,parent['constants'],pg.roots[j]));columns.append(helpers.raw(parent['system'],parent['constants'],X)[:,j])
            else:b.roots.append(b.tree(Node.from_dict(s['tree']),s['constants']));columns.append(raw_tree(s['tree'],s['constants'],X))
        graph,warm=b.finish();check(graph.to_dict()==e['graph'] and np.array_equal(warm,e['warm']),tag+':joint_warm_source_graph')
        close(helpers.raw(e['graph'],e['warm'],X),np.column_stack(columns),'joint_warm_route:'+tag)


def data(name,seed):
    families=('log_affine_ratio','reciprocal_quadratic_norm','oscillatory_affine_saturation','exponential_affine_saturation')
    cases=tuple(f'{f}_n{n}' for f in families for n in [0,1]);family=name.rsplit('_n',1)[0]
    def inputs(split):
        code,count={'train':(111,96),'validation':(222,48),'test':(333,192)}[split]
        rng=np.random.default_rng(np.random.SeedSequence([20261104,seed,code,cases.index(name)]))
        return rng.uniform(np.array([.2,.2]),np.array([1.4,1.4]),(count,2))
    def latent(X):
        a,b=X.T
        if family=='log_affine_ratio':
            z=np.log((1+.4*a)/(1+.7*b));return np.column_stack((z,a*z))
        if family=='reciprocal_quadratic_norm':
            z=np.sqrt(1+a*a);return np.column_stack((1/z,b/z))
        if family=='oscillatory_affine_saturation':return np.column_stack((np.sin(.6*a)/(1+.5*b),np.cos(.6*a)/(1+.5*b)))
        z=np.exp(-.4*a)/(1+.6*a);return np.column_stack((z,b*z))
    scale=latent(inputs('train')).std(axis=0);out={}
    for split in ['train','validation','test']:
        X=inputs(split);clean=latent(X)
        rng=np.random.default_rng(np.random.SeedSequence([20261104,seed,{'train':411,'validation':422,'test':433}[split],cases.index(name)]))
        observed=clean+(.01 if name.endswith('_n1') else 0.)*scale*rng.normal(size=clean.shape)
        out[split]=(X,observed,clean)
    return out

def main():
    if OUT.exists():raise RuntimeError('Keep existing review checkpoint.')
    helper_path=ROOT/'experiments/new_task_validation_20261030/post_review_audit.py'
    helpers=load_helper(helper_path,'_independent_development_raw_data_helpers');helpers.check=check;helpers.data=data
    raw,graph_audit=helpers.raw,helpers.graph_audit
    error=lambda p,y,s:np.sqrt(np.mean((p-y)**2,axis=0))/s
    f,m,scores,verified,summary,archive,execution=(read(HERE/n) for n in ['FREEZE.json','ALL_MODELS_FROZEN.json','SCORES.json','VERIFICATION.json','SUMMARY.json','FROZEN_INPUT_ARCHIVE.json','EXECUTION.json'])
    sealed_hashes={n:sha(HERE/n) for n in ['FREEZE.json','ALL_MODELS_FROZEN.json','SCORES.json','VERIFICATION.json','SUMMARY.json','FROZEN_INPUT_ARCHIVE.json','EXECUTION.json','PRE_RUN_ACCEPTANCE.json','REPORT.md']}
    lock=read(SHARED/'CANDIDATE_LOCK.json');jobs=f['jobs'];acceptance=read(HERE/'PRE_RUN_ACCEPTANCE.json')
    check(len(jobs)==len({key(j) for j in jobs})==32 and [r['job'] for r in scores['rows']]==[r['job'] for r in verified['rows']]==jobs,'complete_ordered_matrix')
    check(len(f['files'])==archive['file_count']==8691 and archive['files']==f['files'],'8691_input_manifest')
    integrity=dict(live=hashes(f['files'],ROOT),archive=hashes(f['files'],HERE/'frozen_inputs'),runs=hashes(m['files'],HERE),source=hashes(lock['source_files'],ROOT))
    physical_archive=archive_io.disk_path(HERE/'frozen_inputs')
    check({p.relative_to(physical_archive).as_posix() for p in physical_archive.rglob('*') if p.is_file()}==set(f['files']),'exact_archive_members')
    check(len(m['files'])==352 and set(m['files'])=={p.relative_to(HERE).as_posix() for p in (HERE/'runs').rglob('*') if p.is_file()},'352_run_members')
    check({p.relative_to(ROOT).as_posix() for p in (ROOT/'src').rglob('*.py')}==set(lock['source_files']) and len(lock['source_files'])==41,'complete_source_manifest')
    check(all(sha(HERE/'source'/n)==h for n,h in lock['source_files'].items()),'source_copies')
    check(m['freeze_sha256']==archive['freeze_sha256']==verified['freeze_sha256']==sha(HERE/'FREEZE.json') and scores['all_models_sha256']==verified['models_sha256']==sha(HERE/'ALL_MODELS_FROZEN.json') and verified['scores_sha256']==sha(HERE/'SCORES.json'),'receipt_cross_hashes')
    check(f['candidate_lock_sha256']==acceptance['source_lock_sha256']==sha(SHARED/'CANDIDATE_LOCK.json'),'candidate_lock')
    check(acceptance['status']=='accepted_for_single_locked_candidate_validation' and acceptance['accepted_unix']<f['budget']['start_unix'] and acceptance['harness_review_sha256']==sha(HERE/'HARNESS_REVIEW.md') and acceptance['harness_review_checks_sha256']==sha(HERE/'HARNESS_REVIEW_CHECKS.json'),'independent_pre_run_acceptance')
    check(acceptance['test_validation_sha256']==sha(HERE/'TEST_VALIDATION.json'),'pre_run_tests_binding')
    check(acceptance['prior_acceptance_sha256']==sha(PRIOR/'POST_REVIEW_ACCEPTANCE.json') and acceptance['local_candidate_lock_sha256']==sha(HERE/'CANDIDATE_LOCK.json'),'eight_key_prerequisite_and_local_lock')
    budget=read(HERE/'BUDGET.json');check(budget==f['budget'],'budget_snapshot')
    for k,v in dict(limit_seconds=2880,verification_reserve_seconds=600,system_cap_seconds=60,fit_seconds=52,sequential_workers=1,threads=1).items():check(budget[k]==v,'budget_'+k)
    check(execution['returncode']==0 and execution['termination_reason'] is None and execution['verification_passed'] is True and execution['wall_seconds']<2940,'normal_bounded_exit')
    check(verified['status']=='passed' and verified['verified_systems']==32 and verified['budget_met'] is True and verified['preparation_error'] is None and verified['elapsed_seconds']<2880,'official_verification')
    stamps=dict(freeze=f['frozen_unix'],training=read(HERE/'TRAINING_STARTED.json')['start_unix'],all_models=m['frozen_unix'],scores=scores['scored_unix'],verification=verified['verified_unix'])
    check(list(stamps.values())==sorted(stamps.values()) and stamps['verification']<budget['start_unix']+2880,'chronology_budget')
    check(f['test_labels_accessed'] is False and m['test_labels_accessed'] is False,'test_isolation_flags')
    from experiments.completion_development_verified_20261031 import study
    check(study.matrix()==jobs and study.configuration()==f['config'],'actual_fixed_configuration_matrix')
    upstream=study.check_stage_gate()
    previous=stamps['training'];models={};results=[];data_hashes={}
    for job,score,vr in zip(jobs,scores['rows'],verified['rows']):
        tag=study.name(job);folder=HERE/'runs'/tag;model=read(folder/'model.json');models[key(job)]=model
        result=read(folder/'result.json');resource=read(folder/'resources.json');bundle=read(folder/'delivery.json');cell=read(folder/'CELL_FROZEN.json');case=f['specs'][job['case']]
        check(previous<=cell['frozen_unix']<=stamps['all_models'],tag+':cell_time');previous=cell['frozen_unix'];hashes(cell['files'],folder)
        check(set(cell['files'])=={p.name for p in folder.iterdir() if p.is_file() and p.name!='CELL_FROZEN.json'},tag+':cell_members')
        check(all(m['files'][(folder/n).relative_to(HERE).as_posix()]==h for n,h in cell['files'].items()) and m['files'][(folder/'CELL_FROZEN.json').relative_to(HERE).as_posix()]==sha(folder/'CELL_FROZEN.json'),tag+':all_membership')
        check(result['job']==resource['job']==read(folder/'job.json')==score['job']==job and result['status']==resource['status']==score['status']=='complete' and resource['returncode']==0 and vr['verified'] is True,tag+':completion')
        check(not result['test_scores_computed'] and resource['wall_seconds']<=60.5 and model['diagnostics']['elapsed_seconds']<=52.5 and model['diagnostics']['evaluations']<=1200,tag+':resources')
        expected=dict(f['config'],random_state=job['seed'],initialization='compound',sharing_strategy='closure',precision_repair=job['method']=='base',compositional_completion=job['method']=='completion',final_selection='precision',acceptance=case['acceptance'])
        check(model['config']==result['effective_config']==expected,tag+':effective_config')
        arrays=helpers.data(job['case'],job['seed']);X,Y,_=arrays['train'];V,Z,_=arrays['validation'];T,U,clean=arrays['test'];scale=Y.std(axis=0)
        dh=ahash(X,Y,V,Z);check(dh==result['data_sha256']==f['data_sha256'][f"{job['case']}_s{job['seed']}"],tag+':independent_data_hash');data_hashes[tag]=dh
        close(scale,model['scales'],'training_scale:'+tag)
        graph,constants=model['best']['system'],model['best']['constants'];shape,sharing=graph_audit(graph,case,tag)
        check(shape==score['structure']==bundle['structure'] and sharing==score['sharing']==bundle['sharing'] and graph==bundle['system'] and constants==[p['value'] for p in bundle['parameters']],tag+':selected_graph_shape')
        pred={s:raw(graph,constants,a[0]) for s,a in arrays.items()};err={s:error(pred[s],arrays[s][1],scale).tolist() for s in arrays};err['clean']=error(pred['test'],clean,scale).tolist()
        for a,b,label in [(err['train'],model['best']['errors'],'selected_train'),(err['validation'],model['validation_nrmse'],'selected_validation'),(err['test'],score['observed_nrmse'],'selected_test'),(err['clean'],score['clean_nrmse'],'selected_clean')]:close(a,b,label+':'+tag)
        check(result['prediction_hashes']==[dict(train=ahash(pred['train']),validation=ahash(pred['validation']))],tag+':prediction_hashes')
        native=DimenSRMIMORegressor.load(folder/'model.json');physical=PhysicalModel.load(folder/'physical.json')
        for s,a in arrays.items():check(np.array_equal(native.predict(a[0]),pred[s]) and np.array_equal(physical.predict(a[0]),pred[s]),tag+':native_physical_'+s)
        domain=native.audit_domain(case['bounds']);check([domain]==result['audits']==bundle['domain_audits'] and domain['status']=='enclosed',tag+':domain')
        good_shape=all(shape[k]<=case['acceptance'][k] for k in STRUCTURE)
        dev=bool(good_shape and all(np.all(np.asarray(err[s])<=case['acceptance']['nrmse']) for s in ['train','validation']))
        test=bool(good_shape and np.all(np.asarray(err['test'])<=case['acceptance']['nrmse']) and np.all(np.asarray(err['clean'])<=case['clean_test_nrmse']))
        accepted=dev and test and model['post_selection']['status']=='accepted'
        check(accepted==score['both_accepted'] and dev==bundle['development_accepted'] and test==score['test_accepted'],tag+':acceptance')
        history=[json.loads(line) for line in (folder/'events.jsonl').read_text(encoding='utf-8').splitlines()]
        mechanism=dict(policy_delta=sum(e.get('parameter_delta',0) for e in history),neural_valid=sum(e.get('neural_valid',0) for e in history),gp_evaluations=sum(e.get('new_evaluations',0) for e in history if e['phase']=='gp'),nsga_steps=len({e['step'] for e in history if e['phase']=='nsga2' and e['new_evaluations']>0}),rademacher_systems=model['diagnostics']['rademacher_systems'])
        check(all(x>0 for x in mechanism.values()) and mechanism['nsga_steps']>=2 and score['mechanism']['full_mechanism_pass'] is True,tag+':mechanisms')
        rec=model['diagnostics']['sharing_closure'];n=rec['base_candidate_count'];base=[payload(c) for c in model['candidates'][:n]];check(digest(base)==rec['base_candidate_sha256'],tag+':foundation_hash')
        supplement=rec['precision_repair' if job['method']=='base' else 'compositional_completion'];parent=min(base,key=lambda c:(float(np.mean(c['errors'])),c['complexity'],digest(c['system'])))
        if supplement['triggered']:check(supplement['parent']==parent,tag+':fixed_parent')
        if job['method']=='completion':source_audit(job,model,base,supplement,X,Y,helpers)
        known={digest(c) for c in base};retained={};stats={};seen=set()
        for label,record in [('supplement',supplement),('closure',rec)]:
            kept=[];valid=0
            for e in record['events']:
                event_count[label]+=1;check(digest(e['parent']) in known,tag+':parent_pool_source');gh=digest(e['graph']);check(gh not in seen,tag+':unique_event');seen.add(gh);graph_audit(e['graph'],case,tag+':event')
                pe=error(raw(e['parent']['system'],e['parent']['constants'],X),Y,scale);close(pe,e['parent']['errors'],'event_parent:'+tag)
                child=e['child'];protected=False
                if child is not None:
                    valid+=1;check(e['new_evaluation'] and child['system']==e['graph'],tag+':child_source');ce=error(raw(child['system'],child['constants'],X),Y,scale);close(ce,child['errors'],'event_child:'+tag);protected=bool(np.all(ce<=pe*1.1+1e-8))
                    if protected:kept.append(child);known.add(digest(child))
                check(protected==e['protected'],tag+':training_protection')
            new=sum(e['new_evaluation'] for e in record['events']);check(new==record['new_evaluations'] and valid==record['valid'] and len(kept)==record['protected'] and valid-len(kept)==record['protection_rejections'],tag+':'+label+'_counts')
            check(new<=(12 if label=='supplement' else 48),tag+':'+label+'_cap')
            vk=[c for c in kept if np.isfinite(raw(c['system'],c['constants'],V)).all() and np.isfinite(error(raw(c['system'],c['constants'],V),Z,scale)).all()]
            retained[label]=kept;retained[label+'_validation']=vk;stats[label]=dict(new_evaluations=new,valid=valid,protected=len(kept),validation_retained=len(vk))
        check(stats['closure']==vr['closure'],tag+':formal_matching_counts')
        for k,value in stats['supplement'].items():check(vr['repair' if job['method']=='base' else 'completion'][k]==value,tag+':formal_supplement_'+k)
        check(rec['base_evaluations']+stats['supplement']['new_evaluations']+stats['closure']['new_evaluations']==model['diagnostics']['evaluations'],tag+':evaluation_total')
        check(all(sum(e['new_evaluation'] for e in rec['events'] if e['round']==q)<=24 for q in [0,1]),tag+':round_caps')
        check(digest(base+retained['supplement']+retained['closure'])==rec['full_training_pool_sha256'],tag+':training_pool')
        check([payload(c) for c in model['candidates'][n:]]==retained['supplement_validation']+retained['closure_validation'] and rec['validation_retained_extensions']==len(retained['supplement_validation'])+len(retained['closure_validation']),tag+':validation_retention')
        info=dict(triggered=int(supplement['triggered']),**stats['supplement'])
        if job['method']=='completion':
            for k in ['compositional_linear_subproblems','implicit_linear_subproblems','linear_subproblems','compositional_returned_roots','implicit_returned_roots']:info[k]=supplement[k]
            for kind in ['compositional','implicit']:info[kind+'_event_roots']=sum(len(e[kind+'_outputs']) for e in supplement['events'])
            comp=[e for e in supplement['events'] if e['compositional_outputs']]
            info.update(compositional_new_systems=sum(e['new_evaluation'] for e in comp),compositional_valid_systems=sum(e['child'] is not None for e in comp),compositional_protected_systems=sum(e['protected'] for e in comp),compositional_validation_retained=sum(e['protected'] and np.isfinite(raw(e['child']['system'],e['child']['constants'],V)).all() for e in comp))
            check(all(vr['completion'][k]==value for k,value in info.items()),tag+':all_discovery_counts')
        results.append(dict(job=job,accepted=accepted,train_nrmse=err['train'],validation_nrmse=err['validation'],observed_nrmse=err['test'],clean_nrmse=err['clean'],shape=shape,mechanism=mechanism,closure=stats['closure'],supplement=info,candidates=len(model['candidates']),fit_seconds=model['diagnostics']['elapsed_seconds'],wall_seconds=resource['wall_seconds'],evaluations=model['diagnostics']['evaluations']))
        print('reviewed',tag,flush=True)
    index={key(r['job']):r for r in results};pairs=[];selected_pairs=[]
    for row in summary['pairs']:
        case,seed=row['case'],row['seed'];a,b=index[case,seed,'completion'],index[case,seed,'base'];ma,mb=models[case,seed,'completion'],models[case,seed,'base'];ra,rb=ma['diagnostics']['sharing_closure'],mb['diagnostics']['sharing_closure']
        pa,pb=[payload(c) for c in ma['candidates'][:ra['base_candidate_count']]],[payload(c) for c in mb['candidates'][:rb['base_candidate_count']]]
        p=dict(case=case,seed=seed,available=True,protected=(np.asarray(a['clean_nrmse'])<=np.asarray(b['clean_nrmse'])*1.1+1e-8).tolist(),candidate_accepted=a['accepted'],control_accepted=b['accepted'],base_identity=pa==pb and all(ra[k]==rb[k] for k in ['base_candidate_count','base_candidate_sha256','base_evaluations','base_search_fingerprint']))
        check(p==row,'paired_summary:'+case+str(seed));pairs.append(p)
        same=dict(case=case,seed=seed,graph_equal=ma['best']['system']==mb['best']['system'],constants_equal=ma['best']['constants']==mb['best']['constants'],best_payload_equal=ma['best']==mb['best'],base_selected_is_foundation=any(c['system']==mb['best']['system'] and c['constants']==mb['best']['constants'] for c in pb),completion_selected_is_foundation=any(c['system']==ma['best']['system'] and c['constants']==ma['best']['constants'] for c in pa))
        selected_pairs.append(same)
    methods={};matching={}
    for method in ['base','completion']:
        rows=[r for r in results if r['job']['method']==method]
        methods[method]=dict(expected=16,complete=16,failed_or_missing=0,both_accepted=sum(r['accepted'] for r in rows),mean_clean_nrmse=float(np.mean([r['clean_nrmse'] for r in rows])),mean_shared_graph_size=float(np.mean([r['shape']['shared_graph_size'] for r in rows])))
        for k,value in methods[method].items():
            if isinstance(value,float):close(value,summary['methods'][method][k],'method_'+k)
            else:check(value==summary['methods'][method][k],'method_'+k)
        matching[method]={k:sum(r['closure'][k] for r in rows) for k in ['new_evaluations','valid','protected','validation_retained']}
    check(matching==summary['closure_by_method'],'matching_totals')
    families={}
    for family,expected in summary['completion_discovery_by_family'].items():
        rows=[r for r in results if r['job']['method']=='completion' and f['specs'][r['job']['case']]['family']==family];ps=[p for p in pairs if f['specs'][p['case']]['family']==family]
        tally={k:sum(r['supplement'][k] for r in rows) for k in expected if k not in ['expected_systems','acceptance_gains']}
        tally.update(expected_systems=len(rows),acceptance_gains=sum(p['candidate_accepted'] and not p['control_accepted'] for p in ps))
        check(tally==expected,'family_discovery:'+family);families[family]=tally
    protected=sum(sum(p['protected']) for p in pairs);preserved=all(not p['control_accepted'] or p['candidate_accepted'] for p in pairs)
    technical=all(r['mechanism']['nsga_steps']>=2 for r in results) and all(p['base_identity'] for p in pairs) and all(all(v>0 for v in c.values()) for c in matching.values())
    adoption=technical and methods['completion']['both_accepted']==16 and protected==32 and preserved and methods['completion']['mean_clean_nrmse']<=methods['base']['mean_clean_nrmse'] and methods['completion']['mean_shared_graph_size']<=methods['base']['mean_shared_graph_size']
    check(technical==summary['technical_pass'] is True and adoption==summary['adoption_pass'] is False and summary['protected_outputs']==protected==32,'scientific_gates')
    for name in ['full_denominator','paired_base_identity_pass','closure_mechanism_pass','control_acceptance_preserved','macro_clean_error_nonincrease','mean_graph_nonincrease']:check(summary[name] is True,'summary_'+name)
    check(summary['candidate_all_accepted'] is False,'candidate_all_accepted_false')
    check('archived_control_identity' not in json.dumps(summary)+json.dumps(verified) and summary['historical_control_comparison']=='not_applicable_no_same_task_archive','no_fake_archived_control')
    failures=[]
    for r in results:
        if r['job']['method']!='completion' or r['accepted']:continue
        spec=f['specs'][r['job']['case']]
        reasons=[k+' exceeds '+str(limit) for k,limit in spec['acceptance'].items() if k!='nrmse' and r['shape'][k]>limit]
        for label,errors,limits in [('train',r['train_nrmse'],spec['acceptance']['nrmse']),('validation',r['validation_nrmse'],spec['acceptance']['nrmse']),('observed_test',r['observed_nrmse'],spec['acceptance']['nrmse']),('clean_test',r['clean_nrmse'],spec['clean_test_nrmse'])]:
            reasons.extend(label+' output'+str(i)+' exceeds '+str(limit) for i,(e,limit) in enumerate(zip(errors,limits)) if not math.isfinite(e) or e<0 or e>limit)
        saved=next(x for x in summary['completion_failures'] if x['job']==r['job'])
        check(reasons==saved['failure_reasons'],str(r['job'])+':failure_reasons')
        failures.append(dict(job=r['job'],failure_reasons=reasons,shape=r['shape'],train_nrmse=r['train_nrmse'],validation_nrmse=r['validation_nrmse'],observed_nrmse=r['observed_nrmse'],clean_nrmse=r['clean_nrmse']))
    check(len(failures)==len(summary['completion_failures'])==11,'all_11_failures')
    family_acceptance={family:{method:sum(r['accepted'] for r in results if r['job']['method']==method and f['specs'][r['job']['case']]['family']==family) for method in ['base','completion']} for family in families}
    integrity['final_live']=hashes(f['files'],ROOT);integrity['final_archive']=hashes(f['files'],HERE/'frozen_inputs');integrity['final_runs']=hashes(m['files'],HERE)
    check(all(sha(HERE/n)==h for n,h in sealed_hashes.items()),'terminal_receipts_unchanged')
    check(not (ROOT/'experiments/completion_historical_20261031/POST_REVIEW_ACCEPTANCE.json').exists(),'original_failed_no_acceptance')
    check(not (HERE/'POST_REVIEW_ACCEPTANCE.json').exists(),'review_did_not_write_acceptance')
    output=dict(status='independent_reconstruction_consistent' if not issues else 'review_findings',issues=issues,checked_unix=time.time(),receipt_sha256=sealed_hashes,script_sha256=sha(Path(__file__)),helper_sha256=sha(helper_path),integrity=integrity,upstream_gate=upstream,
        source_lock_sha256=sha(SHARED/'CANDIDATE_LOCK.json'),source_count=41,pre_run_acceptance=acceptance,timestamps=stamps,budget=budget,execution=execution,
        reconstructed_selected_systems=32,native_reloads=32,physical_reloads=32,domain_rechecks=32,maximum_absolute_differences=dict(maxdiff),training_validation_hashes=data_hashes,
        systems=results,pairs=pairs,methods=methods,matching=matching,family_discovery=families,source_kinds_checked=dict(source_count),returned_sources=source_rows,event_counts=dict(event_count),
        historical_control_comparison='not_applicable_no_same_task_archive',selected_pair_identity=selected_pairs,completion_failures=failures,family_acceptance=family_acceptance,all_output_protection=protected,old_acceptances_preserved=preserved,recomputed_technical_pass=technical,recomputed_adoption_pass=adoption,
        saved_candidate_count=sum(r['candidates'] for r in results),maximum_fit_seconds=max(r['fit_seconds'] for r in results),maximum_cell_wall=max(r['wall_seconds'] for r in results),maximum_evaluations=max(r['evaluations'] for r in results),
        all_candidate_metrics_independently_recomputed=False,rademacher_and_initializer_enumeration_recomputed=False,source_canonical_grammar_and_joint_builder_utilities_reused=True,no_fit_search_or_official_scoring=True,no_acceptance_written=True,adapted_from_independent_audit_sha256=sha(ROOT/'experiments/completion_development_verified_20261031/post_review_validation/audit.py'))
    OUT.write_text(json.dumps(output,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else str(v))+'\n',encoding='utf-8')
    print(json.dumps(dict(status=output['status'],issues=issues,methods=methods,matching=matching,source_counts=dict(source_count),event_counts=dict(event_count),maxdiff=dict(maxdiff),receipt_sha256=sha(OUT))))

if __name__=='__main__':main()
