import json
import numpy as np
import pytest

from experiments.implicit_rational_20261028 import study,worker,evaluate


def test_implicit_worker_changes_only_initializer(monkeypatch,tmp_path):
    captured=[];splits=[]
    class StopAfterConfiguration:
        def __init__(self,cfg,*args):
            captured.append(cfg)
        def fit(self,*args,**kwargs):
            raise RuntimeError('Test stops before fitting.')
    monkeypatch.setattr(worker,'DimenSRMIMORegressor',StopAfterConfiguration)
    def data(case,split,seed):
        splits.append(split)
        return np.ones((4,2)),np.ones((4,2))
    monkeypatch.setattr(worker.cases,'training_data',data)
    for method in study.METHODS:
        path=tmp_path/method;path.mkdir()
        worker.run(path,dict(method=method,case='rational_n0',seed=702),study.configuration())
    assert [c.initialization for c in captured]==['compound','implicit']
    assert all(c.sharing_strategy=='closure' and c.wall_seconds==52 and c.max_evaluations==1200 for c in captured)
    assert splits==['train','validation']*2
    with pytest.raises(ValueError,match='frozen'):
        worker.run(tmp_path,dict(method='random'),{})
    jobs=study.matrix()
    assert len(jobs)==34 and len({study.name(j) for j in jobs})==34 and study.LIMIT==2880


def test_implicit_recommendation_cannot_trade_accepted_cases_or_skip_mechanism(monkeypatch):
    jobs=study.matrix()
    monkeypatch.setattr(evaluate,'read',lambda path:dict(best=dict(errors=[1e-14]*2),
        validation_nrmse=[1e-14]*2,search_fingerprint='base'))
    scores=[dict(job=j,status='complete',both_accepted=True,mean_clean_nrmse=1e-14,clean_nrmse=[1e-14]*2,
        structure=dict(shared_graph_size=20)) for j in jobs]
    verified=[dict(job=j,verified=True,closure=dict(new_evaluations=1,valid=1,protected=1,validation_retained=1),
        initializer=dict(jobs=1,new_evaluations=1,valid=1,linear_subproblems=1,retained_roots=1)
        if j['method']=='implicit' else {}) for j in jobs]
    assert evaluate.summarize(scores,verified)['adoption_pass']
    for row in verified:
        if row['job']['method']=='implicit':row['initializer']['valid']=0
    assert not evaluate.summarize(scores,verified)['technical_pass']
    for row in verified:
        if row['job']['method']=='implicit':row['initializer']['valid']=1
    # Equal total acceptance must not compensate for losing another accepted case.
    next(r for r in scores if r['job']['method']=='compound' and r['job']['case']=='products')['both_accepted']=False
    next(r for r in scores if r['job']['method']=='implicit' and r['job']['case']=='boundary')['both_accepted']=False
    result=evaluate.summarize(scores,verified)
    assert not result['control_acceptance_preserved'] and not result['adoption_pass']
    for row in scores:row['both_accepted']=True
    target=next(r for r in scores if r['job']['method']=='implicit' and r['job']['case']=='rational_n0' and r['job']['seed']==702)
    target['clean_nrmse']=[2e-4,1e-14]
    assert not evaluate.summarize(scores,verified)['rational_training_precision_gained']
    target['clean_nrmse']=[1e-14]*2
    scores[0]=dict(job=jobs[0],status='system_timeout');verified[0]['verified']=False
    result=evaluate.summarize(scores,verified)
    assert not result['technical_pass'] and result['expected_outputs']==34 and result['protected_outputs']==32


def test_initializer_evidence_crosschecks_runtime_event_and_counts(monkeypatch,tmp_path):
    monkeypatch.setattr(evaluate,'HERE',tmp_path)
    job=dict(case='rational_n0',seed=702,method='implicit')
    folder=tmp_path/'runs'/study.name(job);folder.mkdir(parents=True)
    row=dict(phase='initialization',new_evaluations=4,implicit_seed_jobs=3,
        implicit_seed_new_evaluations=2,implicit_seed_valid=1)
    model=dict(history=[row],diagnostics=dict(initialization_diagnostics=[dict(implicit_denominators=2,
        implicit_linear_subproblems=4,implicit_retained_indices=[0])]*2))
    (folder/'model.json').write_text(json.dumps(model))
    (folder/'events.jsonl').write_text(json.dumps(row)+'\n')
    assert evaluate.initializer_evidence(job)['valid']==1
    row['implicit_seed_valid']=3
    (folder/'events.jsonl').write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='event drift'):
        evaluate.initializer_evidence(job)


def test_two_nsga_generations_must_each_do_new_evaluations(monkeypatch,tmp_path):
    monkeypatch.setattr(study.prior,'mechanism',lambda folder:dict(full_mechanism_pass=True,outputs=[]))
    events=[dict(phase='nsga2',step=i,new_evaluations=int(i==0)) for i in range(4)]
    path=tmp_path/'events.jsonl'
    path.write_text('\n'.join(map(json.dumps,events)))
    assert not study.mechanism(tmp_path)['full_mechanism_pass']
    events[1]['new_evaluations']=1
    path.write_text('\n'.join(map(json.dumps,events)))
    report=study.mechanism(tmp_path)
    assert report['full_mechanism_pass'] and report['nsga_generations_with_new_evaluations']==2
