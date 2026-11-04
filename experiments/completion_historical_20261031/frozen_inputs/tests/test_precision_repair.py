from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dimensr_mimo import DimenSRMIMORegressor, SearchConfig, repair, sharing, structure
from dimensr_mimo._single.expression import Node
from dimensr_mimo._single.units import ZERO
from dimensr_mimo.evaluation import Candidate, Evaluator
from dimensr_mimo.generation import SystemGrammar
from dimensr_mimo.graph import System, Vertex
from dimensr_mimo.selection import digest


def config(**kwargs):
    return SearchConfig(initialization='compound', sharing_strategy='closure', final_selection='precision',
        acceptance=dict(nrmse=[1e-4]*2, shared_graph_size=50, independent_parameters=20,
                        expanded_total_nodes=60, maximum_output_depth=8), **kwargs)


def candidate(index=0, errors=(.1, .1)):
    return Candidate(System((Vertex('x'+str(index)),), (0, 0)), np.array([]), np.array(errors), 1., 0., 0.)


def fake_evaluator(limit=100, expired=False, errors=(.01, .01)):
    parent = candidate()
    class Evaluation:
        cfg = config(precision_repair=True, population_size=24, max_evaluations=limit)
        X, Y, scales = np.ones((4, 40)), np.ones((4, 2)), np.ones(2)
        deadline = None
        calls = 1
        def __init__(self):
            self.cache = {parent.system:parent}
        def expired(self):
            return expired
        def __call__(self, system, warm):
            assert system not in self.cache  # No cached or failed graph may be refitted.
            self.calls += 1
            child = Candidate(system, warm.copy(), np.array(errors), 1., 0., 0.)
            self.cache[system] = child
            return child
    return parent, Evaluation()


def test_config_is_opt_in_and_strict():
    assert not SearchConfig().precision_repair
    config(precision_repair=True).validate()
    for value in (1, 'true', None):
        with pytest.raises(ValueError, match='boolean'):
            config(precision_repair=value).validate()
    for field, value in (('initialization', 'implicit'), ('sharing_strategy', 'random'), ('final_selection', 'guarded')):
        with pytest.raises(ValueError):
            replace(config(precision_repair=True), **{field:value}).validate()


def test_training_gate_skips_without_initializer_cache_or_grammar_access(monkeypatch):
    parent, evaluator = fake_evaluator()
    parent.errors[:] = 1e-4
    before = deepcopy(parent.to_dict())
    monkeypatch.setattr(repair, 'structured_seeds', lambda *a, **k:pytest.fail('Initializer must not run.'))
    retained, report = repair.precision_repair([parent], evaluator, None)
    assert not retained and not report['triggered'] and report['training_eligible']==1
    assert evaluator.calls==1 and parent.to_dict()==before and len(evaluator.cache)==1


@pytest.mark.parametrize('limit, expired, expected', [(100,False,12), (4,False,3), (1,False,0), (100,True,0)])
def test_repair_joint_new_evaluation_and_global_caps(monkeypatch, limit, expired, expected):
    parent, evaluator = fake_evaluator(limit, expired)
    # Oversupply only the stub to exercise the scheduler's independent 12-call cap.
    def seeds(g, X, y, limit, **kwargs):
        assert limit==12 and kwargs==dict(compound=True, implicit=True, deadline=None)
        return [(Node('x'+str(i)), np.array([])) for i in range(1,21)], dict(implicit_retained_indices=list(range(20)))
    monkeypatch.setattr(repair, 'structured_seeds', seeds)
    retained, report = repair.precision_repair([parent], evaluator, SimpleNamespace(plain=[None,None]))
    assert len(retained)==report['new_evaluations']==report['protected']==expected
    assert evaluator.calls==1+expected
    assert all(e['new_evaluation'] and e['implicit_outputs']==[0,1] for e in report['events'])


def test_parent_protection_cache_failures_nonimplicit_and_missing_roots(monkeypatch):
    parent, evaluator = fake_evaluator(errors=(.09,.12))
    # Root 1 has no proposal and must be cloned from the original parent.
    def seeds(g, *args, **kwargs):
        return ([(Node('x'+str(i)), np.array([])) for i in range(4)], dict(implicit_retained_indices=[0,1,2])) if g==0 else ([], {})
    monkeypatch.setattr(repair, 'structured_seeds', seeds)
    # Builder emits separate roots, so cache the exact emitted graph, including failures.
    from dimensr_mimo.graph import Builder
    for i, value in ((0,parent),(1,None)):
        b = Builder(); b.roots.extend((b.tree(Node('x'+str(i))),b.tree(Node('x0'))))
        evaluator.cache[b.finish()[0]]=value
    before = deepcopy(parent.to_dict())
    retained, report = repair.precision_repair([parent], evaluator, SimpleNamespace(plain=[0,1]))
    assert not retained and report['cache_skips']==2 and report['nonimplicit_skips']==1
    assert report['new_evaluations']==report['valid']==report['protection_rejections']==1
    event = report['events'][0]
    assert event['implicit_outputs']==[0] and event['parent']==before
    assert System.from_dict(event['graph']).nodes[-1].token=='x0'
    assert parent.to_dict()==before


def test_unequal_nonempty_seed_lists_use_fixed_parent_for_missing_index(monkeypatch):
    parent,evaluator=fake_evaluator()
    def seeds(g, *args, **kwargs):
        indices=[1,2] if g==0 else [3]
        return [(Node('x'+str(i)),np.array([])) for i in indices],dict(implicit_retained_indices=list(range(len(indices))))
    monkeypatch.setattr(repair,'structured_seeds',seeds)
    retained,report=repair.precision_repair([parent],evaluator,SimpleNamespace(plain=[0,1]))
    assert len(retained)==2
    assert report['events'][1]['implicit_outputs']==[0]
    graph=retained[1].system
    assert [graph.nodes[r].token for r in graph.roots]==['x2','x0']


def test_zero_seed_limit_is_empty_without_new_evaluations():
    parent, evaluator = fake_evaluator()
    evaluator.cfg.population_size=1
    grammar=SystemGrammar((ZERO,), (ZERO,ZERO), (ZERO,), evaluator.cfg)
    retained, report=repair.precision_repair([parent], evaluator, grammar)
    assert not retained and report['seed_limit']==0 and evaluator.calls==1


@pytest.mark.parametrize('stage', ['admission','product','linear'])
def test_structured_deadline_is_checked_inside_work(monkeypatch, stage):
    cfg=config(max_nodes=15, max_depth=6, operators=('add','mul','div','square'))
    g=SystemGrammar((ZERO,ZERO), (ZERO,ZERO), (ZERO,), cfg).plain[0]
    X=np.random.default_rng(34).uniform(.2,1.2,(16,2)); y=X[:,0]/(.7+X[:,1]**2)
    now=[0.]
    monkeypatch.setattr(structure.time, 'monotonic', lambda:now[0])
    if stage=='admission':
        original=Node.evaluate
        def evaluate(*a,**k):
            now[0]=2.;return original(*a,**k)
        monkeypatch.setattr(Node,'evaluate',evaluate)
    elif stage=='product':
        original=structure._compound_products
        def products(*a,**k):
            now[0]=2.;return original(*a,**k)
        monkeypatch.setattr(structure,'_compound_products',products)
    else:
        original=np.linalg.lstsq
        def solve(*a,**k):
            now[0]=2.;return original(*a,**k)
        monkeypatch.setattr(np.linalg,'lstsq',solve)
    with pytest.raises(TimeoutError) as exc:
        structure.structured_seeds(g,X,y,12,compound=True,implicit=True,deadline=1.)
    counts=exc.value.initialization_diagnostics
    assert counts['feature_evaluations']>0
    if stage=='linear':assert counts['linear_subproblems']==1


def test_repair_deadline_keeps_base_and_partial_diagnostics(monkeypatch):
    parent,evaluator=fake_evaluator()
    def seeds(*a,**k):
        error=TimeoutError('deadline');error.initialization_diagnostics={'linear_subproblems':3}
        raise error
    monkeypatch.setattr(repair,'structured_seeds',seeds)
    pool,report=sharing.protected_closure([parent],evaluator,SimpleNamespace(plain=[None,None]))
    assert pool==[parent] and evaluator.calls==1
    assert report['precision_repair']['stop_reason']=='deadline'
    assert report['precision_repair']['initialization_diagnostics']==[{'linear_subproblems':3}]


def test_real_implicit_repair_is_joint_fitted_without_changing_base():
    cfg=config(precision_repair=True, max_nodes=15,max_depth=6,operators=('add','mul','div','square'),
        population_size=8,constant_max_nfev=12,constant_restarts=1,max_evaluations=40)
    X=np.random.default_rng(92).uniform(.2,1.2,(24,2)); a,b=X.T
    Y=np.column_stack((1.4*b/(.8+a*a),.6*b*b/(.8+a*a)))
    evaluator=Evaluator(X,Y,cfg,input_bounds=[[.2,1.2],[.2,1.2]])
    base=evaluator(candidate().system,np.array([])); before=deepcopy(base.to_dict())
    fingerprint=digest([c.to_dict() for c in evaluator.cache.values() if c is not None])
    grammar=SystemGrammar((ZERO,ZERO),(ZERO,ZERO),(ZERO,),cfg)
    pool,report=sharing.protected_closure([base],evaluator,grammar)
    repair_report=report['precision_repair']
    assert repair_report['triggered'] and repair_report['new_evaluations']>0 and repair_report['protected']>0
    assert min(max(c.errors) for c in pool)<1e-8
    assert report['base_candidate_count']==1 and pool[0].to_dict()==before
    assert report['base_search_fingerprint']==fingerprint
    assert evaluator.calls==report['base_evaluations']+repair_report['new_evaluations']+report['new_evaluations']
    assert len(pool)==1+repair_report['protected']+report['protected']
    assert evaluator.stats['rademacher_systems']>1 and evaluator.stats['constant_nfev']>0


def test_untriggered_complete_pool_selection_and_reload_are_identical(tmp_path):
    X=np.linspace(.2,1.2,20)[:,None];Y=np.column_stack((X[:,0],2*X[:,0]))
    cfg=config(population_size=4,epochs=1,gp_generations=1,selection_generations=2,
        max_evaluations=40,min_candidates=3,polish_candidates=0,constant_max_nfev=8,
        constant_restarts=1,max_nodes=7,operators=('add','mul'),use_rnn=False)
    def fit(c):
        return DimenSRMIMORegressor(c).fit(X,Y,X_val=X[::3],Y_val=Y[::3],input_bounds=[[.2,1.2]])
    base,patched=fit(cfg),fit(replace(cfg,precision_repair=True))
    assert not patched.diagnostics_['sharing_closure']['precision_repair']['triggered']
    assert base.candidates_==patched.candidates_ and base.best_.to_dict()==patched.best_.to_dict()
    assert base.search_fingerprint_==patched.search_fingerprint_
    path=tmp_path/'repair.json';patched.save(path);loaded=DimenSRMIMORegressor.load(path)
    np.testing.assert_array_equal(loaded.predict(X),patched.predict(X))
    assert json.loads(json.dumps(loaded.to_dict()))==json.loads(json.dumps(patched.to_dict()))
    payload=json.loads(path.read_text());payload['diagnostics']['sharing_closure']['base_candidate_sha256']='tampered'
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match='base pool'):
        DimenSRMIMORegressor.load(path)


def test_old_frozen_precision_model_verifies_then_migrates(tmp_path):
    root=Path(__file__).resolve().parents[1]
    path=root/'experiments/implicit_rational_20261028/runs/rational_n0_s702_compound/model.json'
    saved=json.loads(path.read_text(encoding='utf-8'))
    assert 'precision_repair' not in saved['config']
    model=DimenSRMIMORegressor.load(path)
    assert not model.config.precision_repair and model.best_.to_dict()==saved['best']
    model.save(tmp_path/'migrated.json')
    assert DimenSRMIMORegressor.load(tmp_path/'migrated.json').to_dict()==model.to_dict()
    saved['post_selection']['source_model_sha256']='tampered'
    path=tmp_path/'tampered.json';path.write_text(json.dumps(saved))
    with pytest.raises(ValueError,match='Saved selection'):
        DimenSRMIMORegressor.load(path)
