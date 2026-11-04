import pytest
from dimensr_mimo import SearchConfig
from experiments.protected_closure_20261028 import study, worker, evaluate


def test_closure_worker_scope_and_complete_denominators(monkeypatch, tmp_path):
    calls=[]
    monkeypatch.setattr(worker.base,'run',lambda folder,job,config:calls.append(config))
    for method in study.METHODS:
        worker.run(tmp_path,dict(method=method),dict(max_evaluations=1200))
    assert calls==[dict(max_evaluations=1200,sharing_strategy=m) for m in ('random','closure')]
    with pytest.raises(ValueError,match='frozen'):
        worker.run(tmp_path,dict(method='matched'),{})
    jobs=study.matrix()
    assert len(jobs)==34 and len({study.name(j) for j in jobs})==34 and study.LIMIT==2880
    for case in study.cases.CASES:
        SearchConfig(**dict(study.configuration(),sharing_strategy='closure',final_selection='precision',
            acceptance=study.cases.design(case)['acceptance'])).validate()


def test_recommendation_requires_all_outputs_and_compact_gain(monkeypatch):
    jobs=study.matrix()
    monkeypatch.setattr(evaluate,'read',lambda path:dict(best=dict(errors=[1e-14]*2),validation_nrmse=[1e-14]*2,
        search_fingerprint='base',diagnostics=dict(frozen_candidate_sha256='pool',
        sharing_closure=dict(base_search_fingerprint='base',base_candidate_sha256='pool'))))
    scores=[dict(job=j,status='complete',both_accepted=True,mean_clean_nrmse=1e-14,clean_nrmse=[1e-14]*2,
        structure=dict(shared_graph_size=20)) for j in jobs]
    verified=[dict(job=j,verified=True,closure=dict(new_evaluations=1,valid=1,protected=1,validation_retained=1)) for j in jobs]
    assert evaluate.summarize(scores,verified)['adoption_pass']
    for r in verified:r['closure']['validation_retained']=0
    assert not evaluate.summarize(scores,verified)['technical_pass']
    for r in verified:r['closure']['validation_retained']=1
    target=next(r for r in scores if r['job']['case']=='damped_n0' and r['job']['seed']==701 and r['job']['method']=='closure')
    target['structure']['shared_graph_size']=21
    assert not evaluate.summarize(scores,verified)['adoption_pass']
    target['structure']['shared_graph_size']=20
    scores[0]=dict(job=jobs[0],status='system_timeout');verified[0]['verified']=False
    result=evaluate.summarize(scores,verified)
    assert not result['technical_pass'] and not result['adoption_pass']
    assert result['expected_outputs']==34 and result['protected_outputs']==32
