import importlib.util
from pathlib import Path
import sys
from fractions import Fraction
import numpy as np
import pytest
from dimensr_mimo import DimenSRMIMORegressor,SearchConfig
from dimensr_mimo import grammar,generation
from dimensr_mimo._single.units import ZERO,unit


@pytest.fixture
def reference_grammar():
    path=Path(__file__).resolve().parents[1]/'experiments/phase_budget_20261027/source/src/dimensr_mimo/grammar.py'
    name='dimensr_mimo._uncached_grammar_fixture'
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module)
    yield module.Grammar
    sys.modules.pop(name,None)


def same_trace(a,b):
    for field in ('tokens','parents','contexts','production_indices'):assert getattr(a,field)==getattr(b,field)
    for field in ('masks','production_features'):
        assert len(getattr(a,field))==len(getattr(b,field))
        for x,y in zip(getattr(a,field),getattr(b,field)):np.testing.assert_array_equal(x,y)


def test_cached_grammar_preserves_fraction_units_choices_and_traces(reference_grammar):
    L=unit([1,0,0,0,0,0,0]);H=tuple(v*Fraction(1,2) for v in L)
    args=([L,H],L,[ZERO,L,H],('add','sub','mul','div','square','sqrt','sin','cos','exp','log'))
    options=dict(max_nodes=13,max_depth=5,reference_units=(L,H),max_references=3,output_context=[1.,0.])
    old=reference_grammar(*args,**options);new=grammar.Grammar(*args,**options)
    assert old.units==new.units and old.cost==new.cost and old.tokens==new.tokens
    for u in old.units:
        assert [(p.token,p.children) for p in old.productions[u]]==[(p.token,p.children) for p in new.productions[u]]
    a=np.random.default_rng(83);b=np.random.default_rng(83)
    for _ in range(25):
        x,tx=old.generate(a);y,ty=new.generate(b);assert x==y;same_trace(tx,ty)
        same_trace(old.trace(x),new.trace(y))
    choices=next(v for v in new.options(L,13,5).values() if v)
    before=new.production_features(choices,5);changed=new.production_features(choices,5);changed[:]=999
    np.testing.assert_array_equal(new.production_features(choices,5),before)
    for cached in (grammar.add,grammar.sub,grammar.scale):
        assert cached.cache_info().currsize<=cached.cache_info().maxsize


def test_cached_grammar_preserves_complete_untimed_search(reference_grammar,monkeypatch):
    X=np.random.default_rng(52).uniform(.2,1.2,(40,2));Y=np.column_stack((X[:,0]**2+X[:,1],X[:,0]*X[:,1]))
    cfg=SearchConfig(random_state=52,population_size=6,epochs=2,gp_generations=2,selection_generations=3,
        max_evaluations=160,min_candidates=4,polish_candidates=1,polish_max_nfev=10,
        constant_restarts=1,constant_max_nfev=8,structured_seeds=False,wall_seconds=None)
    def fit():return DimenSRMIMORegressor(cfg).fit(X[:28],Y[:28],X_val=X[28:],Y_val=Y[28:],input_bounds=[[.2,1.2]]*2)
    monkeypatch.setattr(generation,'Grammar',reference_grammar);old=fit()
    monkeypatch.setattr(generation,'Grammar',grammar.Grammar);new=fit()
    assert old.search_fingerprint_==new.search_fingerprint_
    assert old.candidates_==new.candidates_ and old.best_.to_dict()==new.best_.to_dict()
    np.testing.assert_array_equal(old.predict(X),new.predict(X))
    for key in ('operations','changed','novel','successful'):assert old.diagnostics_[key]==new.diagnostics_[key]
