from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import pytest

from dimensr_mimo import SearchConfig
from dimensr_mimo._single.expression import Node
from dimensr_mimo._single.units import ZERO,encode
from dimensr_mimo.graph import Builder,System,Vertex
from dimensr_mimo.selection import digest
from experiments.compositional_completion_20261031 import harness,verification
from experiments.completion_historical_20261031 import study as historical,worker as hw,evaluate as he
from experiments.completion_development_20261031 import study as development,evaluate as de


def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj),encoding='utf-8')


@pytest.fixture(params=[historical,development],ids=['historical','development'])
def stage(request):return request.param


@pytest.fixture
def summary_fixture(stage,monkeypatch):
    s=stage;v=he.verifier if s is historical else de.verifier;jobs=s.matrix()
    scores=[dict(job=j,status='complete',both_accepted=True,clean_nrmse=[1e-14]*2,observed_nrmse=[1e-14]*2,
        mean_clean_nrmse=1e-14,structure=dict(shared_graph_size=12,independent_parameters=4,expanded_total_nodes=20,maximum_output_depth=5)) for j in jobs]
    if s is historical:
        for j in {tuple((r['job']['case'],r['job']['seed'])) for r in scores[-12:]}:
            for r in scores:
                if (r['job']['case'],r['job']['seed'])==j:r['both_accepted']=False
    rows=[dict(job=j,verified=True,archived_control_identity=True,closure=dict(new_evaluations=1,valid=1,protected=1,validation_retained=1),
        completion=dict(triggered=1,new_evaluations=0,valid=0,protected=0,validation_retained=0)) for j in jobs]
    fixture_candidate=dict(system={},constants=[],errors=[0.,0.],complexity=1.,rademacher=0.,rademacher_complexity=0.)
    models={s.name(j):dict(best=dict(errors=[1e-14]*2),validation_nrmse=[1e-14]*2,candidates=[deepcopy(fixture_candidate)],
        diagnostics=dict(sharing_closure=dict(base_candidate_count=1,base_candidate_sha256='pool',base_evaluations=3,base_search_fingerprint='cache'))) for j in jobs}
    monkeypatch.setattr(verification,'read',lambda p:models[p.parent.name])
    return s,v,scores,rows,models


def test_matrix_and_isolated_verifiers(stage):
    s=stage;v=he.verifier if s is historical else de.verifier
    assert len(s.matrix())==len({s.name(j) for j in s.matrix()})==s.EXPECTED
    assert s.LIMIT==2880 and all(j['seconds']==60 for j in s.matrix())
    assert v.legacy.HERE==v.base.HERE==s.HERE
    assert v.legacy.cases is v.base.cases is s.cases
    assert v.legacy.study is v.base.study is s
    assert v.base.evaluate.__globals__['guards'].__self__ is v
    assert v.legacy is not (de.verifier.legacy if s is historical else he.verifier.legacy)
    assert historical.cases is not development.cases
    with pytest.raises(RuntimeError,match='seal'):
        s.cases.generate(s.cases.CASES[0],'test',s.cases.seeds(s.cases.CASES[0])[0])
    with pytest.raises(ValueError):s.cases.training_data('unused','test',0)


def test_two_flags_only_change_worker_configuration(stage,monkeypatch,tmp_path):
    captured=[];splits=[]
    class Stop:
        def __init__(self,cfg,*args):captured.append(asdict(cfg))
        def fit(self,*args,**kw):raise RuntimeError('Fixture stops before any fit.')
    monkeypatch.setattr(harness,'DimenSRMIMORegressor',Stop)
    monkeypatch.setattr(stage,'check_candidate_lock',lambda:None);monkeypatch.setattr(stage,'check_stage_gate',lambda:None)
    def data(case,split,seed):
        splits.append(split);assert split in ('train','validation')
        return np.ones((4,2)),np.ones((4,2))
    monkeypatch.setattr(stage.cases,'training_data',data)
    case=stage.cases.CASES[0]
    for method in stage.METHODS:
        folder=tmp_path/method;folder.mkdir()
        harness.worker_run(stage,folder,dict(case=case,seed=stage.cases.seeds(case)[0],method=method),stage.configuration())
    assert [(c.pop('precision_repair'),c.pop('compositional_completion')) for c in captured]==[(True,False),(False,True)]
    assert captured[0]==captured[1] and captured[0]['max_evaluations']==1200 and captured[0]['wall_seconds']==52
    assert splits==['train','validation']*2


def test_historical_does_not_claim_all_17_solved(summary_fixture):
    s,v,scores,rows,_=summary_fixture;result=v.summarize(scores,rows)
    assert result['technical_pass'] and result['adoption_pass']
    if s is historical:
        assert result['methods']['base']['both_accepted']==result['methods']['completion']['both_accepted']==11
        assert not result['candidate_all_accepted'] and len(result['completion_failures'])==6
    else:assert result['candidate_all_accepted']
    assert result['protected_outputs']==s.EXPECTED


@pytest.mark.parametrize('fault',['missing','duplicate','wrong_case','wrong_verification_case','missing_verification','malformed_job','nan','inf','negative','negative_mean','missing_metric','negative_structure','nan_counter'])
def test_complete_denominators_and_invalid_metrics_fail_closed(summary_fixture,fault):
    s,v,scores,rows,_=summary_fixture
    if fault=='missing':scores.pop()
    elif fault=='duplicate':scores[-1]=scores[0]
    elif fault=='wrong_case':scores[-1]['job']=dict(scores[-1]['job'],case='wrong_case')
    elif fault=='wrong_verification_case':rows[-1]['job']=dict(rows[-1]['job'],case='wrong_case')
    elif fault=='missing_verification':rows.pop()
    elif fault=='malformed_job':scores[-1]['job']={}
    elif fault in ('nan','inf','negative'):scores[-1]['clean_nrmse'][0]={'nan':float('nan'),'inf':float('inf'),'negative':-1.}[fault]
    elif fault=='negative_mean':scores[-1]['mean_clean_nrmse']=-1.
    elif fault=='missing_metric':scores[-1].pop('observed_nrmse')
    elif fault=='nan_counter':rows[-1]['closure']['valid']=float('nan')
    else:scores[-1]['structure']['shared_graph_size']=-1
    result=v.summarize(scores,rows)
    assert not result['technical_pass'] and not result['adoption_pass']
    assert result['expected_systems']==s.EXPECTED and result['expected_outputs']==s.EXPECTED and len(result['pairs'])==s.EXPECTED//2
    if fault not in ('missing_verification','wrong_verification_case','nan_counter'):assert any(m['mean_clean_nrmse'] is None for m in result['methods'].values())
    json.dumps(result,allow_nan=False)


@pytest.mark.parametrize('gate',['protection','macro','graph','control','foundation','matching','target_graph','target_parameters','target_precision'])
def test_no_scientific_gate_can_be_averaged_away(summary_fixture,gate):
    s,v,scores,rows,models=summary_fixture
    target=next(r for r in scores if r['job']['method']=='completion')
    if gate.startswith('target') and s is development:return
    if gate=='protection':
        for r in scores:
            if r['job']['method']=='base':r['clean_nrmse']=[1e-5]*2;r['mean_clean_nrmse']=1e-5
        target['clean_nrmse'][0]=3e-5
        for r in scores:
            if r is not target and r['job']['method']=='completion':r['clean_nrmse']=[0.,0.]
    elif gate=='macro':target['clean_nrmse']=[2e-14]*2
    elif gate=='graph':target['structure']['shared_graph_size']=13
    elif gate=='control':next(r for r in rows if r['job']['method']=='base')['archived_control_identity']=False
    elif gate=='foundation':models[s.name(target['job'])]['diagnostics']['sharing_closure']['base_evaluations']=4
    elif gate=='matching':
        for r in rows:
            if r['job']['method']=='base':r['closure']['validation_retained']=0
    elif gate=='target_graph':target['structure']['shared_graph_size']=21
    elif gate=='target_parameters':target['structure']['independent_parameters']=5
    else:models[s.name(target['job'])]['best']['errors'][0]=1e-8
    result=v.summarize(scores,rows);assert not result['adoption_pass']
    if gate=='protection':assert result['macro_clean_error_nonincrease'] and result['protected_outputs']==s.EXPECTED-1


@pytest.fixture
def completion_fixture():
    tree=Node('sqrt',(Node('square',(Node('x0'),)),))
    X=np.array([[.5,.9],[.7,1.1],[.9,1.4],[1.1,1.7]]);Y=np.zeros((4,2))
    parentgraph=System((Vertex('x1'),),(0,0))
    parent=dict(system=parentgraph.to_dict(),constants=[],errors=np.sqrt(np.mean(X[:,1]**2))*np.ones(2),complexity=1.,rademacher=0.,rademacher_complexity=0.)
    parent['errors']=parent['errors'].tolist()
    b=Builder()
    for _ in range(2):b.roots.append(b.tree(tree,[]))
    graph,warm=b.finish()
    child=dict(system=graph.to_dict(),constants=[],errors=(np.sqrt(np.mean(X[:,0]**2))*np.ones(2)).tolist(),complexity=float(graph.size),rademacher=0.,rademacher_complexity=0.)
    source=dict(kind='compositional',returned_index=0,proposal_index=0,tree=tree.to_dict(),constants=[],
        score=float(np.mean(X[:,0]**2))/1e-12+1e-8*tree.size,transform='sqrt',form='direct',variable=None,basis_indices=[0],intercept=False,outer_amplitude=False)
    stats=dict(returned_sources=[source],proposed_structures=0,implicit_retained_indices=[],compositional_retained_indices=[0],
        compositional_basis_count=1,compositional_linear_subproblems=1,implicit_linear_subproblems=0,linear_subproblems=0)
    event=dict(seed_index=0,parent=parent,graph=graph.to_dict(),warm=warm.tolist(),child=child,protected=True,new_evaluation=True,
        implicit_outputs=[],compositional_outputs=[0,1],root_sources=[deepcopy(source),deepcopy(source)])
    report=dict(mode='compositional_completion_v1',triggered=True,training_eligible=0,seed_limit=12,new_evaluations=1,valid=1,protected=1,
        protection_rejections=0,cache_skips=0,structure_rejections=0,nonimplicit_skips=0,missing_roots=0,
        initialization_diagnostics=[deepcopy(stats),deepcopy(stats)],events=[event],parent=parent,stop_reason='completed',
        compositional_linear_subproblems=2,implicit_linear_subproblems=0,linear_subproblems=2,implicit_returned_roots=0,compositional_returned_roots=2)
    cfg=SearchConfig(**dict(historical.configuration(),compositional_completion=True,final_selection='precision',acceptance=historical.cases.design('rational_n0')['acceptance']))
    model=dict(feature_units=[encode(ZERO)]*2,target_units=[encode(ZERO)]*2,constant_units=[encode(ZERO)],scales=[1.,1.],diagnostics=dict(sharing_closure=dict(compositional_completion=report)))
    return he.verifier,model,[parent],X,Y,cfg,{digest(parent)},report


def test_completion_replay_uses_actual_sources_and_joint_child(completion_fixture):
    v,model,payload,X,Y,cfg,known,report=completion_fixture
    retained,info=v.completion_evidence(model,payload,X,Y,cfg,known)
    assert len(retained)==1 and info['new_evaluations']==info['valid']==info['protected']==1
    assert info['compositional_returned_roots']==info['compositional_event_roots']==2
    assert info['compositional_linear_subproblems']==info['linear_subproblems']==2


@pytest.mark.parametrize('fault',['mode','trigger','parent','root_source','warm','tree','score','ranking_index','linear','new_count','protection','negative_count','fake_transform','missing_roots'])
def test_completion_rejects_corrupt_provenance(completion_fixture,fault):
    v,model,payload,X,Y,cfg,known,r=completion_fixture
    if fault=='mode':r['mode']='training_trigger'
    elif fault=='trigger':r['triggered']=False
    elif fault=='parent':r['parent']=dict(payload[0],complexity=3.)
    elif fault=='root_source':r['events'][0]['root_sources'][0]['kind']='implicit'
    elif fault=='warm':r['events'][0]['warm']=[2.]
    elif fault=='tree':r['events'][0]['graph']['nodes'][0]['token']='x1'
    elif fault=='score':r['initialization_diagnostics'][0]['returned_sources'][0]['score']=0.
    elif fault=='ranking_index':r['initialization_diagnostics'][0]['returned_sources'][0]['returned_index']=1
    elif fault=='linear':r['linear_subproblems']=3
    elif fault=='new_count':r['new_evaluations']=13
    elif fault=='protection':r['events'][0]['protected']=False
    elif fault=='negative_count':r['cache_skips']=-1
    elif fault=='fake_transform':r['initialization_diagnostics'][0]['returned_sources'][0]['transform']='log'
    else:r['missing_roots']=1
    with pytest.raises((ValueError,AssertionError)):
        v.completion_evidence(model,payload,X,Y,cfg,known)


def test_completion_deadline_retains_partial_receipt_without_events(completion_fixture):
    v,model,payload,X,Y,cfg,known,r=completion_fixture
    r.update(events=[],initialization_diagnostics=[],new_evaluations=0,valid=0,protected=0,
        compositional_linear_subproblems=0,implicit_linear_subproblems=0,linear_subproblems=0,stop_reason='deadline',implicit_returned_roots=0,compositional_returned_roots=0)
    retained,info=v.completion_evidence(model,payload,X,Y,cfg,known)
    assert not retained and info['triggered']==1 and info['new_evaluations']==0 and info['stop_reason']=='deadline'


def test_control_keeps_original_training_trigger_skip(completion_fixture):
    v,model,payload,X,Y,cfg,known,r=completion_fixture
    cfg.precision_repair=True;cfg.compositional_completion=False
    payload[0]['errors']=[1e-5]*2
    r.update(triggered=False,training_eligible=1,events=[],initialization_diagnostics=[],new_evaluations=0,valid=0,protected=0,stop_reason='training_precision_met')
    model['diagnostics']['sharing_closure']={'precision_repair':r}
    retained,info=v.legacy.repair_evidence(model,payload,X,Y,cfg,known)
    assert not retained and info['triggered']==0 and info['new_evaluations']==0
    r['triggered']=True
    with pytest.raises(ValueError,match='trigger'):v.legacy.repair_evidence(model,payload,X,Y,cfg,known)


def test_partial_deadline_stats_do_not_claim_old_positions_as_new_roots(completion_fixture):
    v,model,payload,X,Y,cfg,known,r=completion_fixture
    r.update(events=[],initialization_diagnostics=[dict(returned_sources=[],implicit_retained_indices=[0],
        structured_implicit_retained_indices=[0],implicit_linear_subproblems=4,linear_subproblems=9,
        compositional_linear_subproblems=3)],new_evaluations=0,valid=0,protected=0,compositional_linear_subproblems=3,
        implicit_linear_subproblems=4,linear_subproblems=12,stop_reason='deadline',implicit_returned_roots=0,compositional_returned_roots=0)
    retained,info=v.completion_evidence(model,payload,X,Y,cfg,known)
    assert not retained and info['linear_subproblems']==12 and info['implicit_returned_roots']==info['compositional_returned_roots']==0


@pytest.mark.parametrize('method',['base','completion'])
def test_full_supplement_closure_configuration_and_accounting(completion_fixture,monkeypatch,tmp_path,method):
    v,model,payload,X,Y,cfg,known,report=completion_fixture
    s=historical;monkeypatch.setattr(s,'HERE',tmp_path)
    monkeypatch.setattr(s.cases,'training_data',lambda *a:(X,Y))
    job=dict(case='rational_n0',seed=701,method=method,seconds=60)
    config=s.configuration();cfg=harness.effective_config(s,job,config)
    supplement=report['events'][0]['child']
    smaller=deepcopy(supplement);smaller['system']=System((Vertex('x0'),),(0,0)).to_dict();smaller['complexity']=1.
    closure_event=dict(round=0,parent=supplement,graph=smaller['system'],warm=[],child=smaller,protected=True,new_evaluation=True)
    label='compositional_completion'
    if method=='base':
        label='precision_repair'
        report=deepcopy(report)
        report['initialization_diagnostics']=[dict(proposed_structures=1,implicit_retained_indices=[0],implicit_denominators=1,implicit_linear_subproblems=1) for _ in range(2)]
        report['events'][0]['implicit_outputs']=[0,1]
    record=dict(base_candidate_count=1,base_candidate_sha256=digest(payload),base_evaluations=3,base_search_fingerprint='base-cache',
        rounds=1,new_evaluations=1,valid=1,protected=1,protection_rejections=0,events=[closure_event],
        full_training_pool_sha256=digest(payload+[supplement,smaller]),validation_retained_extensions=2,**{label:report})
    prefix='repair' if method=='base' else 'completion'
    history=dict(phase='sharing_closure',new_evaluations=2,new_systems=1,protected_systems=1,
        **{prefix+'_new_systems':1,prefix+'_protected_systems':1})
    model.update(config=asdict(cfg),candidates=payload+[supplement,smaller],best=smaller,
        validation_nrmse=smaller['errors'],history=[history],diagnostics=dict(evaluations=5,sharing_closure=record),
        post_selection=dict(base_candidate_index=0,base_choice_sha256=digest(payload[0]),
            base_training_bounds=(np.asarray(payload[0]['errors'])*1.1+1e-8).tolist(),base_validation_bounds=(np.asarray(payload[0]['errors'])*1.1+1e-8).tolist()))
    folder=tmp_path/'runs'/s.name(job)
    save(folder/'model.json',model);save(folder/'result.json',dict(effective_config=asdict(cfg)))
    save(folder/'resources.json',dict(wall_seconds=1.,returncode=0))
    (folder/'events.jsonl').write_text(json.dumps(history)+'\n')
    closure,info=v.closure_evidence(job,dict(config=config))
    assert closure['new_evaluations']==info['new_evaluations']==1
    assert closure['validation_retained']==info['validation_retained']==1
    if method=='completion':assert info['compositional_validation_retained']==1
    model['diagnostics']['evaluations']=4;save(folder/'model.json',model)
    with pytest.raises(ValueError,match='Evaluation total'):v.closure_evidence(job,dict(config=config))


@pytest.mark.parametrize('bad',['missing_scores','malformed_scores','broken_guard','scoring_failure'])
def test_verification_failures_always_write_full_receipts(stage,monkeypatch,tmp_path,bad):
    s=stage;v=verification.Verifier(s)
    monkeypatch.setattr(s,'HERE',tmp_path);monkeypatch.setattr(s,'budget',lambda:time_limit())
    save(tmp_path/'FREEZE.json',{});save(tmp_path/'ALL_MODELS_FROZEN.json',{})
    frozen=dict(jobs=s.matrix(),budget=dict(start_unix=verification.time.time()))
    def guard():
        if bad=='broken_guard':raise ValueError('broken cell hash')
        return frozen,dict(frozen_unix=1)
    monkeypatch.setattr(v,'guards',guard)
    if bad=='malformed_scores':(tmp_path/'SCORES.json').write_text('{bad')
    if bad=='scoring_failure':save(tmp_path/'SCORING_FAILURE.json',dict(status='failed'))
    v.verify()
    result=harness.read(tmp_path/'VERIFICATION.json');summary=harness.read(tmp_path/'SUMMARY.json')
    assert result['status']=='failed' and len(result['rows'])==s.EXPECTED and result['verified_systems']==0
    assert not summary['technical_pass'] and not summary['adoption_pass'] and summary['methods']['completion']['mean_clean_nrmse'] is None


def time_limit():return verification.time.time()+2880


def test_global_scoring_failure_is_preserved_with_full_verification(stage,monkeypatch,tmp_path):
    s=stage;v=verification.Verifier(s)
    monkeypatch.setattr(s,'HERE',tmp_path);monkeypatch.setattr(s,'budget',time_limit)
    monkeypatch.setattr(v,'guards',lambda:(dict(jobs=s.matrix(),budget={'start_unix':verification.time.time()}),{}))
    def fail():raise ValueError('fixture scoring phase failure')
    monkeypatch.setattr(v,'evaluate',fail)
    v.run()
    assert harness.read(tmp_path/'SCORING_FAILURE.json')['status']=='failed'
    result=harness.read(tmp_path/'VERIFICATION.json')
    assert result['status']=='failed' and len(result['rows'])==s.EXPECTED
    with pytest.raises(ValueError,match='evidence'):v.run()


def test_control_identity_only_ignores_declared_metadata(monkeypatch,tmp_path):
    s=historical;v=verification.Verifier(s)
    monkeypatch.setattr(s,'EXPECTED',1)
    monkeypatch.setattr(s,'HERE',tmp_path/'new');monkeypatch.setattr(s,'PRIOR',tmp_path/'old')
    job=dict(case='fixture',seed=1,method='base',seconds=60);oldjob=dict(job,method='repair')
    now=s.HERE/'runs'/s.name(job);old=s.PRIOR/'runs'/s.name(oldjob)
    model=dict(config=dict(precision_repair=True),best={'errors':[0.,0.]},candidates=[1],search_fingerprint='cache',
        post_selection={'source_model_sha256':'old'},history=[dict(evaluations=4,elapsed_seconds=2.)],diagnostics=dict(sharing_closure=dict(precision_repair=dict(triggered=True))))
    current=deepcopy(model);current['config']['compositional_completion']=False;current['history'][0]['elapsed_seconds']=5.;current['post_selection']['source_model_sha256']='new'
    score=dict(job=job,status='complete',clean_nrmse=[0.,0.],observed_nrmse=[0.,0.],mean_clean_nrmse=0.,both_accepted=True,
        structure=dict(shared_graph_size=1,independent_parameters=0,expanded_total_nodes=2,maximum_output_depth=1))
    for n in ('result.json','delivery.json','pool.json'):save(now/n,{});save(old/n,{})
    save(now/'model.json',current);save(old/'model.json',model);save(s.PRIOR/'SCORES.json',dict(rows=[dict(score,job=oldjob)]))
    assert v.control_identity(job,score)
    current['history'][0]['evaluations']=5;save(now/'model.json',current)
    with pytest.raises(ValueError,match='identity'):v.control_identity(job,score)


def test_source_lock_is_dynamic_and_blocks_before_worker_data(stage,monkeypatch,tmp_path):
    s=stage;monkeypatch.setattr(s,'ROOT',tmp_path);monkeypatch.setattr(s,'SHARED',tmp_path/'shared')
    paths=[]
    for i in range(3):
        path=tmp_path/f'src/m{i}.py';path.parent.mkdir(exist_ok=True);path.write_text('# locked');paths.append(path)
    save(s.SHARED/'CANDIDATE_LOCK.json',dict(expected_source_count=3,source_files={p.relative_to(tmp_path).as_posix():harness.sha(p) for p in paths}))
    s.check_candidate_lock()
    paths[0].write_text('changed')
    monkeypatch.setattr(s.cases,'training_data',lambda *a:pytest.fail('No arrays when source drifted.'))
    with pytest.raises(ValueError,match='drift'):harness.worker_run(s,tmp_path,dict(method='base'),{})
    assert not (tmp_path/'result.json').exists()


def test_existing_execution_is_not_restarted(stage,monkeypatch,tmp_path):
    monkeypatch.setattr(stage,'HERE',tmp_path)
    save(tmp_path/'PREFLIGHT.json',{})
    monkeypatch.setattr(harness.subprocess,'Popen',lambda *a,**k:pytest.fail('No rerun process.'))
    with pytest.raises(ValueError,match='no restart'):harness.bounded(stage)


@pytest.mark.parametrize('fault',['early','late','missing_cell','missing_artifact','changed_cell'])
def test_reused_cell_chronology_and_membership_guards(monkeypatch,tmp_path,fault):
    s=historical;v=verification.Verifier(s);legacy=v.legacy
    monkeypatch.setattr(legacy,'HERE',tmp_path)
    jobs=[dict(case='fixture',seed=1,method=m) for m in s.METHODS];frozen=dict(jobs=jobs);sealed=dict(frozen_unix=20,files={})
    save(tmp_path/'TRAINING_STARTED.json',dict(start_unix=10))
    for i,j in enumerate(jobs):
        folder=tmp_path/'runs'/s.name(j);save(folder/'state.json',{})
        stamp=12+i
        if i:stamp={'early':9,'late':21}.get(fault,stamp)
        save(folder/'CELL_FROZEN.json',dict(frozen_unix=stamp,files={'state.json':harness.sha(folder/'state.json')}))
        for n in ('state.json','CELL_FROZEN.json'):sealed['files'][(folder/n).relative_to(tmp_path).as_posix()]=harness.sha(folder/n)
    if fault=='missing_cell':sealed['files'].pop((folder/'CELL_FROZEN.json').relative_to(tmp_path).as_posix())
    elif fault=='missing_artifact':sealed['files'].pop((folder/'state.json').relative_to(tmp_path).as_posix())
    elif fault=='changed_cell':(folder/'CELL_FROZEN.json').write_text('changed')
    monkeypatch.setattr(legacy,'_base_guards',lambda:(frozen,sealed))
    with pytest.raises((ValueError,json.JSONDecodeError)):legacy.guards()
