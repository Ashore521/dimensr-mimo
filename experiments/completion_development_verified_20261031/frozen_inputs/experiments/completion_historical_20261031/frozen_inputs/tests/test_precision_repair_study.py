from copy import deepcopy
from dataclasses import asdict
import json

import numpy as np
import pytest

from dimensr_mimo import SearchConfig
from dimensr_mimo._single.units import ZERO, encode
from dimensr_mimo.graph import System, Vertex
from dimensr_mimo.selection import digest
from experiments.precision_repair_20261028 import study, worker, evaluate


def test_worker_only_changes_repair_flag_without_training(monkeypatch,tmp_path):
    captured=[]; splits=[]
    class StopAfterConfiguration:
        def __init__(self,cfg,*args): captured.append(asdict(cfg))
        def fit(self,*args,**kwargs): raise RuntimeError('Construction test stops before fitting.')
    monkeypatch.setattr(worker,'DimenSRMIMORegressor',StopAfterConfiguration)
    def data(case,split,seed):
        splits.append(split)
        return np.ones((4,2)),np.ones((4,2))
    monkeypatch.setattr(worker.cases,'training_data',data)
    for method in study.METHODS:
        path=tmp_path/method; path.mkdir()
        worker.run(path,dict(method=method,case='rational_n0',seed=702),study.configuration())
    assert [c.pop('precision_repair') for c in captured]==[False,True]
    assert captured[0]==captured[1]
    assert captured[0]['initialization']=='compound' and captured[0]['sharing_strategy']=='closure'
    assert captured[0]['wall_seconds']==52 and captured[0]['max_evaluations']==1200
    assert splits==['train','validation']*2
    with pytest.raises(ValueError,match='frozen'):
        worker.run(tmp_path,dict(method='implicit'),{})
    jobs=study.matrix()
    assert len(jobs)==len({study.name(j) for j in jobs})==34 and study.LIMIT==2880
    assert all(j['seconds']==60 for j in jobs)
    for case in study.cases.CASES:
        for method in study.METHODS:
            SearchConfig(**dict(study.configuration(),precision_repair=method=='repair',final_selection='precision',
                acceptance=study.cases.design(case)['acceptance'])).validate()


def candidate(index=0,wrapped=False):
    nodes=(Vertex('x'+str(index)),)
    if wrapped: nodes+=(Vertex('square',(0,)),Vertex('sqrt',(1,)))
    graph=System(nodes,(len(nodes)-1,)*2)
    return dict(system=graph.to_dict(),constants=[],errors=[1. if index==0 else .5]*2,
        complexity=float(graph.size),rademacher=0.,rademacher_complexity=0.)


def repair_report(parent,child):
    event=dict(parent=parent,graph=child['system'],warm=[],child=child,protected=True,new_evaluation=True,
        implicit_outputs=[0,1],seed_index=0)
    return dict(triggered=True,training_eligible=0,seed_limit=12,new_evaluations=1,valid=1,protected=1,
        protection_rejections=0,cache_skips=0,nonimplicit_skips=0,structure_rejections=0,
        initialization_diagnostics=[dict(implicit_denominators=1,implicit_linear_subproblems=3,
            proposed_structures=1,implicit_retained_indices=[0]) for _ in range(2)],events=[event],
        stop_reason='completed',parent=parent)


@pytest.fixture
def evidence(monkeypatch,tmp_path):
    monkeypatch.setattr(evaluate,'HERE',tmp_path)
    job=dict(case='rational_n0',seed=702,method='repair',seconds=60)
    cfg=SearchConfig(**dict(study.configuration(),precision_repair=True,random_state=702,final_selection='precision',
        acceptance=study.cases.design('rational_n0')['acceptance']))
    X=np.tile([1.,.5],(4,1)); Y=np.zeros((4,2))
    monkeypatch.setattr(evaluate.cases,'training_data',lambda *args:(X,Y))
    parent,repair,closure=candidate(wrapped=True),candidate(1,True),candidate(1)
    report=repair_report(parent,repair)
    event=dict(round=0,parent=repair,graph=closure['system'],warm=[],child=closure,protected=True,new_evaluation=True)
    record=dict(base_candidate_count=1,base_candidate_sha256=digest([parent]),base_evaluations=3,
        base_search_fingerprint='base-cache',rounds=1,new_evaluations=1,valid=1,protected=1,protection_rejections=0,
        events=[event],precision_repair=report,full_training_pool_sha256=digest([parent,repair,closure]),
        validation_retained_extensions=2)
    history=dict(phase='sharing_closure',new_evaluations=2,new_systems=1,protected_systems=1,
        repair_new_systems=1,repair_protected_systems=1)
    model=dict(config=asdict(cfg),feature_units=[encode(ZERO)]*2,target_units=[encode(ZERO)]*2,constant_units=[encode(ZERO)],
        candidates=[parent,repair,closure],best=closure,validation_nrmse=[.5]*2,scales=[1.,1.],history=[history],
        diagnostics=dict(evaluations=5,sharing_closure=record),post_selection=dict(base_candidate_index=0,
            base_choice_sha256=digest(parent),base_training_bounds=[1.1]*2,base_validation_bounds=[1.1]*2))
    folder=tmp_path/'runs'/study.name(job);folder.mkdir(parents=True)
    def save():
        for name,obj in (('model.json',model),('result.json',dict(effective_config=asdict(cfg))),
                         ('resources.json',dict(wall_seconds=1.,returncode=0))):
            (folder/name).write_text(json.dumps(obj),encoding='utf-8')
        (folder/'events.jsonl').write_text(json.dumps(history)+'\n',encoding='utf-8')
    return model,job,dict(config=study.configuration()),save


def test_replay_combines_repair_and_closure_with_separate_counters(evidence):
    model,job,frozen,save=evidence;save()
    closure,repair=evaluate.closure_evidence(job,frozen)
    assert closure['new_evaluations']==repair['new_evaluations']==1
    assert closure['validation_retained']==repair['validation_retained']==1
    assert repair['implicit_new_systems']==1 and repair['linear_subproblems']==6
    assert model['diagnostics']['evaluations']==3+repair['new_evaluations']+closure['new_evaluations']


@pytest.mark.parametrize('mutation,match',[
    ('trigger','trigger'),('parent','parent'),('implicit','implicit root'),('count','accounting'),
    ('protection','guard'),('pool','training pool'),('total','Evaluation total'),('history','history accounting'),
    ('base_choice','base choice'),('cap','accounting'),('round','round cap'),('prefix','Base pool')])
def test_replay_rejects_corrupted_provenance(evidence,mutation,match):
    model,job,frozen,save=evidence
    record=model['diagnostics']['sharing_closure'];repair=record['precision_repair']
    if mutation=='trigger': repair['triggered']=False
    elif mutation=='parent': repair['parent']=candidate(1)
    elif mutation=='implicit': repair['events'][0]['implicit_outputs']=[]
    elif mutation=='count': repair['valid']=0
    elif mutation=='protection': repair['events'][0]['protected']=False
    elif mutation=='pool': record['full_training_pool_sha256']='wrong'
    elif mutation=='total': model['diagnostics']['evaluations']=4
    elif mutation=='history': model['history'][0]['new_evaluations']=1
    elif mutation=='base_choice': model['post_selection']['base_candidate_index']=1
    elif mutation=='cap': repair['new_evaluations']=13
    elif mutation=='round': record['events']*=25
    elif mutation=='prefix': record['base_candidate_sha256']='wrong'
    save()
    with pytest.raises(ValueError,match=match): evaluate.closure_evidence(job,frozen)


def test_training_skip_and_partial_deadline_reports(evidence):
    model,job,frozen,save=evidence
    cfg=SearchConfig(**model['config']);payload=evaluate.training_payload(model)[:1]
    report=model['diagnostics']['sharing_closure']['precision_repair']
    report.update(events=[],new_evaluations=0,valid=0,protected=0,initialization_diagnostics=[dict(linear_subproblems=3)],stop_reason='deadline')
    X=np.tile([1.,.5],(4,1));Y=np.zeros((4,2))
    retained,info=evaluate.repair_evidence(model,payload,X,Y,cfg,{digest(c) for c in payload})
    assert not retained and info['triggered']==1 and info['linear_subproblems']==0
    payload[0]['errors']=[1e-4]*2
    report.update(triggered=False,training_eligible=1,initialization_diagnostics=[],stop_reason='training_precision_met')
    retained,info=evaluate.repair_evidence(model,payload,X,Y,cfg,set())
    assert not retained and info['triggered']==0 and info['training_eligible']==1
    report['cache_skips']=1
    with pytest.raises(ValueError,match='counters'): evaluate.repair_evidence(model,payload,X,Y,cfg,set())


def test_implicit_provenance_does_not_cycle_short_output_lists(evidence):
    model,job,frozen,save=evidence
    cfg=SearchConfig(**model['config']);payload=evaluate.training_payload(model)[:1]
    report=model['diagnostics']['sharing_closure']['precision_repair']
    report['initialization_diagnostics'][1].update(proposed_structures=2,implicit_retained_indices=[1])
    report['events'][0].update(seed_index=1,implicit_outputs=[1])
    graph=System((Vertex('x0'),Vertex('square',(0,)),Vertex('sqrt',(1,)),
        Vertex('x1'),Vertex('square',(3,)),Vertex('sqrt',(4,))),(2,5)).to_dict()
    report['events'][0]['graph']=graph
    report['events'][0]['child'].update(system=graph,errors=[1.,.5],complexity=6.)
    X=np.tile([1.,.5],(4,1));Y=np.zeros((4,2))
    assert evaluate.repair_evidence(model,payload,X,Y,cfg,{digest(c) for c in payload})[1]['implicit_new_systems']==1
    report['events'][0]['implicit_outputs']=[0,1]
    with pytest.raises(ValueError,match='implicit root'):
        evaluate.repair_evidence(model,payload,X,Y,cfg,{digest(c) for c in payload})
    report['events'][0]['implicit_outputs']=[1]
    graph['nodes'][0]['token']='x1'
    with pytest.raises(ValueError,match='copied'):
        evaluate.repair_evidence(model,payload,X,Y,cfg,{digest(c) for c in payload})


def test_validation_rejections_replay_both_extension_classes(evidence,monkeypatch):
    model,job,frozen,save=evidence
    # Finite validation inputs make sqrt(x1) invalid while its smaller child x1 remains valid.
    graph=System((Vertex('x1'),Vertex('sqrt',(0,))),(1,1)).to_dict()
    repair=model['candidates'][1];repair.update(system=graph,errors=[np.sqrt(.5)]*2,complexity=2.)
    record=model['diagnostics']['sharing_closure'];report=record['precision_repair']
    report['events'][0]['graph']=graph
    record['full_training_pool_sha256']=digest(evaluate.training_payload(model))
    record['validation_retained_extensions']=1
    record['validation_rejections']=[dict(graph_sha256=digest(graph))]
    model['candidates'].pop(1)
    def data(case,split,seed):
        return np.tile([1.,.5 if split=='train' else -.5],(4,1)),np.zeros((4,2))
    monkeypatch.setattr(evaluate.cases,'training_data',data)
    save()
    with np.errstate(all='ignore'):
        closure,repair=evaluate.closure_evidence(job,frozen)
    assert closure['validation_retained']==1 and repair['validation_retained']==0 and repair['protected']==1
    record['validation_rejections']=[];save()
    with np.errstate(all='ignore'),pytest.raises(ValueError,match='Validation rejection'):
        evaluate.closure_evidence(job,frozen)


@pytest.fixture
def summary_inputs(monkeypatch):
    jobs=study.matrix()
    scores=[dict(job=j,status='complete',both_accepted=True,mean_clean_nrmse=1e-14,clean_nrmse=[1e-14]*2,
        structure=dict(shared_graph_size=20)) for j in jobs]
    verified=[dict(job=j,verified=True,closure=dict(new_evaluations=1,valid=1,protected=1,validation_retained=1),
        repair=dict(triggered=1,new_evaluations=1,valid=1,protected=1,validation_retained=1,linear_subproblems=1,
            retained_roots=1,implicit_new_systems=1) if j['method']=='repair' else {}) for j in jobs]
    pool=[candidate()]
    models={study.name(j):dict(best=dict(errors=[1e-14]*2),validation_nrmse=[1e-14]*2,candidates=deepcopy(pool),
        search_fingerprint='full-cache',diagnostics=dict(sharing_closure=dict(base_candidate_count=1,
            base_candidate_sha256='base-pool',base_evaluations=3,base_search_fingerprint='base-cache',
            **(dict(precision_repair=dict(triggered=False)) if j['method']=='repair' else {})))) for j in jobs}
    monkeypatch.setattr(evaluate,'read',lambda path:models[path.parent.name])
    return scores,verified,models


def test_summary_retains_all_old_and_new_gates(summary_inputs):
    scores,verified,models=summary_inputs
    assert evaluate.summarize(scores,verified)['adoption_pass']
    for row in verified:
        if row['job']['method']=='repair':row['repair']['valid']=0
    assert not evaluate.summarize(scores,verified)['technical_pass']
    for row in verified:
        if row['job']['method']=='repair':row['repair']['valid']=1
    for row in verified:
        if row['job']['method']=='base':row['closure']['validation_retained']=0
    assert not evaluate.summarize(scores,verified)['technical_pass']
    for row in verified:row['closure']['validation_retained']=1
    next(r for r in scores if r['job']['method']=='base' and r['job']['case']=='products')['both_accepted']=False
    next(r for r in scores if r['job']['method']=='repair' and r['job']['case']=='boundary')['both_accepted']=False
    result=evaluate.summarize(scores,verified)
    assert not result['control_acceptance_preserved'] and not result['adoption_pass']
    for row in scores:row['both_accepted']=True
    target=next(r for r in scores if r['job']['method']=='repair' and r['job']['case']=='rational_n0' and r['job']['seed']==702)
    target['clean_nrmse']=[2e-4,1e-14]
    assert not evaluate.summarize(scores,verified)['rational_training_precision_gained']
    target['clean_nrmse']=[1e-14]*2
    scores[0]=dict(job=scores[0]['job'],status='system_timeout');verified[0]['verified']=False
    result=evaluate.summarize(scores,verified)
    assert not result['technical_pass'] and result['expected_outputs']==34 and result['protected_outputs']==32


@pytest.mark.parametrize('gate',['per_output','compact_size','compact_training','rational_validation','macro_error'])
def test_no_scientific_acceptance_gate_is_relaxed(summary_inputs,gate):
    scores,verified,models=summary_inputs
    target=next(r for r in scores if r['job']['method']=='repair' and r['job']['case']=='damped_n0' and r['job']['seed']==701)
    model=models[study.name(target['job'])]
    if gate=='per_output':target['clean_nrmse'][0]=2e-8
    elif gate=='compact_size':target['structure']['shared_graph_size']=21
    elif gate=='compact_training':model['best']['errors'][0]=2e-12
    elif gate=='rational_validation':models['rational_n0_s702_repair']['validation_nrmse'][0]=2e-4
    else:target['mean_clean_nrmse']=2e-14
    result=evaluate.summarize(scores,verified)
    assert not result['adoption_pass']
    if gate=='per_output':assert result['protected_outputs']==33
    if gate.startswith('compact'):assert not result['compact_damped_gain_preserved']
    if gate=='rational_validation':assert not result['rational_training_precision_gained']


@pytest.mark.parametrize('field',['base_candidate_count','base_candidate_sha256','base_search_fingerprint','base_evaluations'])
def test_paired_base_identity_is_required(summary_inputs,field):
    scores,verified,models=summary_inputs
    candidate=next(m for name,m in models.items() if name.endswith('_repair'))
    candidate['diagnostics']['sharing_closure'][field]='drift'
    result=evaluate.summarize(scores,verified)
    assert not result['paired_base_identity_pass'] and not result['technical_pass'] and not result['adoption_pass']


@pytest.mark.parametrize('field',['candidates','search_fingerprint','best','validation_nrmse'])
def test_untriggered_entire_pool_and_selected_model_must_match(summary_inputs,field):
    scores,verified,models=summary_inputs
    candidate=next(m for name,m in models.items() if name.endswith('_repair'))
    candidate[field] = (dict(errors=[2e-14]*2) if field=='best' else [2e-14]*2 if field=='validation_nrmse'
        else [] if field=='candidates' else 'drift')
    result=evaluate.summarize(scores,verified)
    assert not result['untriggered_identity_pass'] and not result['technical_pass']


@pytest.mark.parametrize('kind',['missing_score','missing_verification','duplicate_score','duplicate_verification'])
def test_summary_preserves_full_denominators(summary_inputs,kind):
    scores,verified,_=summary_inputs
    if kind=='missing_score': scores.pop()
    elif kind=='missing_verification': verified.pop()
    elif kind=='duplicate_score':scores[-1]=scores[0]
    else:verified[-1]=verified[0]
    result=evaluate.summarize(scores,verified)
    assert not result['full_denominator'] and not result['technical_pass'] and not result['adoption_pass']
    assert result['expected_systems']==34 and result['expected_outputs']==34
    assert result['methods']['base']['expected']==result['methods']['repair']['expected']==17


def test_two_nsga_generations_must_each_have_new_evaluations(monkeypatch,tmp_path):
    monkeypatch.setattr(study.prior,'mechanism',lambda folder:dict(full_mechanism_pass=True,outputs=[]))
    events=[dict(phase='nsga2',step=i,new_evaluations=int(i==0)) for i in range(4)]
    path=tmp_path/'events.jsonl';path.write_text('\n'.join(map(json.dumps,events)))
    assert not study.mechanism(tmp_path)['full_mechanism_pass']
    events[1]['new_evaluations']=1;path.write_text('\n'.join(map(json.dumps,events)))
    result=study.mechanism(tmp_path)
    assert result['full_mechanism_pass'] and result['nsga_generations_with_new_evaluations']==2


def test_freeze_includes_transitive_selectors_and_implicit_test_helpers(monkeypatch,tmp_path):
    here=tmp_path/'experiments'/'repair';here.mkdir(parents=True)
    dependencies=['precision_delivery_20261026/selector.py','guarded_selection_20261026/selector.py',
                  'implicit_rational_20261028/study.py','implicit_rational_20261028/worker.py','implicit_rational_20261028/evaluate.py',
                  'shared_discovery_20261026/worker.py','unit_composition_20261026/worker.py',
                  'physical_pool_20261027/worker.py','public_transfer_20261026/worker.py']
    for name in dependencies:
        path=here.parent/name;path.parent.mkdir(exist_ok=True);path.write_text('# fixture')
    source=tmp_path/'src'/'stub.py';source.parent.mkdir();source.write_text('# frozen source fixture')
    monkeypatch.setattr(study,'HERE',here);monkeypatch.setattr(study,'ROOT',tmp_path)
    monkeypatch.setattr(study,'read',lambda path:dict(status='passed'))
    monkeypatch.setattr(study,'sha',lambda path:'fixture-hash')
    monkeypatch.setattr(study.subprocess,'check_output',lambda *a,**k:b'fixture-commit')
    monkeypatch.setattr(study.cases,'CASES',['fixture'])
    monkeypatch.setattr(study.cases,'seeds',lambda case:[1])
    monkeypatch.setattr(study.cases,'design',lambda case:{})
    monkeypatch.setattr(study.cases,'training_data',lambda *args:(np.ones((2,2)),np.ones((2,2))))
    study.freeze()
    frozen=json.loads((here/'FREEZE.json').read_text())
    assert all('experiments/'+name in frozen['files'] for name in dependencies)
    assert 'experiments/phase_budget_20261027/source/src/dimensr_mimo/grammar.py' in frozen['files']
    assert 'experiments/native_api_20261026/runs/dimensional_frequency_normalized/model.json' in frozen['files']
    assert 'experiments/repair/source/src/stub.py' in frozen['files']
    assert (here/'source/src/stub.py').read_bytes()==source.read_bytes()


@pytest.mark.parametrize('fault',[None,'early','late','reordered','missing_cell','missing_artifact','changed_cell'])
def test_cell_seals_are_verified_before_scoring(monkeypatch,tmp_path,fault):
    monkeypatch.setattr(evaluate,'HERE',tmp_path)
    frozen=dict(jobs=[dict(case='fixture',seed=1,method=m) for m in ('base','repair')])
    sealed=dict(frozen_unix=20,files={})
    (tmp_path/'TRAINING_STARTED.json').write_text(json.dumps(dict(start_unix=10)))
    for i,job in enumerate(frozen['jobs']):
        folder=tmp_path/'runs'/study.name(job);folder.mkdir(parents=True)
        artifact=folder/'state.txt';artifact.write_text('state')
        stamp=12+2*i
        if i==1:stamp={'early':9,'late':21,'reordered':11}.get(fault,stamp)
        cell=folder/'CELL_FROZEN.json'
        cell.write_text(json.dumps(dict(frozen_unix=stamp,files={'state.txt':evaluate.sha(artifact)})))
        sealed['files'][cell.relative_to(tmp_path).as_posix()]=evaluate.sha(cell)
        sealed['files'][artifact.relative_to(tmp_path).as_posix()]=evaluate.sha(artifact)
    if fault=='missing_cell':sealed['files'].pop(cell.relative_to(tmp_path).as_posix())
    if fault=='missing_artifact':sealed['files'].pop(artifact.relative_to(tmp_path).as_posix())
    if fault=='changed_cell':cell.write_text(cell.read_text()+' ')
    monkeypatch.setattr(evaluate,'_base_guards',lambda:(frozen,sealed))
    assert evaluate.base.evaluate.__globals__['guards'] is evaluate.guards
    if fault:
        with pytest.raises(ValueError,match='Cell'):
            evaluate.guards()
    else:
        assert evaluate.guards()==(frozen,sealed)
