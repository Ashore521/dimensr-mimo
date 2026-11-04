from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
import pytest
from dimensr_mimo import SearchConfig,DimenSRMIMORegressor,regressor,evaluation,fitting


def test_phased_budget_keeps_gp_and_nsga_active_when_neural_proposals_are_slow(monkeypatch):
    clock=[0.]
    fake=SimpleNamespace(monotonic=lambda:clock[0],process_time=time.process_time)
    for module in (regressor,evaluation,fitting):monkeypatch.setattr(module,'time',fake)
    generate=regressor.SystemGrammar.generate
    def slow_generate(self,rng,policy=None):
        answer=generate(self,rng,policy)
        if policy is not None:clock[0]+=2.
        return answer
    monkeypatch.setattr(regressor.SystemGrammar,'generate',slow_generate)
    X=np.random.default_rng(27).uniform(-.8,.8,(32,2));Y=np.column_stack((X[:,0]+X[:,1],X[:,0]-X[:,1]))
    cfg=SearchConfig(random_state=27,population_size=6,epochs=1,gp_generations=2,selection_generations=3,
        max_evaluations=160,min_candidates=3,polish_candidates=0,constant_restarts=1,constant_max_nfev=5,
        structured_seeds=False,wall_seconds=8.)
    def fit(mode):
        clock[0]=0.
        return DimenSRMIMORegressor(replace(cfg,time_allocation=mode)).fit(X[:24],Y[:24],X_val=X[24:],Y_val=Y[24:],input_bounds=[[-.8,.8]]*2)
    original=fit('global');phased=fit('phased')
    assert sum(r['new_evaluations'] for r in original.history_ if r['phase']=='nsga2')==0
    for phase in ('gp','nsga2'):
        assert sum(r['new_evaluations'] for r in phased.history_ if r['phase']==phase)>0
    assert sum(r.get('parameter_delta',0) for r in phased.history_)>0
    assert sum(r['phase']=='nsga2' and r['new_evaluations']>0 for r in phased.history_)>=2
    assert any(r['phase']=='phase_budget' for r in phased.history_)
    assert phased.diagnostics_['elapsed_seconds']<8.


def test_old_precision_model_resaves_with_new_optional_configuration(tmp_path):
    source=Path(__file__).resolve().parents[1]/'experiments/native_api_20261026/runs/dimensional_frequency_normalized/model.json'
    old=DimenSRMIMORegressor.load(source);old.save(tmp_path/'new.json')
    new=DimenSRMIMORegressor.load(tmp_path/'new.json')
    X=np.random.default_rng(24).uniform(-.8,.8,(12,2))
    np.testing.assert_array_equal(old.predict(X),new.predict(X))
    assert new.post_selection_['status']==old.post_selection_['status']=='accepted'
    assert new.to_dict()==old.to_dict()


def test_phase_budget_configuration_rejected_before_search(monkeypatch):
    def forbidden(*a,**k):raise AssertionError('Search should not start.')
    monkeypatch.setattr(evaluation.Evaluator,'__init__',forbidden)
    for cfg in (SearchConfig(time_allocation='unknown'),SearchConfig(time_allocation='phased')):
        with pytest.raises(ValueError):DimenSRMIMORegressor(cfg).fit(np.ones((6,1)),np.ones((6,1)))
