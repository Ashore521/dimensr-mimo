from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dimensr_mimo import DimenSRMIMORegressor, SearchConfig, completion, sharing
from dimensr_mimo._single.expression import Node
from dimensr_mimo._single.units import ZERO, scale, sub, unit
from dimensr_mimo.evaluation import Candidate, Evaluator
from dimensr_mimo.generation import SystemGrammar
from dimensr_mimo.graph import Builder, System, Vertex, in_bounds
from dimensr_mimo.selection import digest
from dimensr_mimo.structure import operator


def config(**kwargs):
    return SearchConfig(initialization='compound', sharing_strategy='closure', final_selection='precision',
        acceptance=dict(nrmse=[1e-4]*2, shared_graph_size=50, independent_parameters=20,
                        expanded_total_nodes=60, maximum_output_depth=8), **kwargs)


def grammar(features=(ZERO, ZERO), target=ZERO, constants=(ZERO,), **kwargs):
    cfg = config(operators=('add', 'mul', 'div', 'square', 'log', 'sqrt'), **kwargs)
    return SystemGrammar(features, (target, target), constants, cfg).plain[0]


def new_proposals(g, X, y):
    counts = completion.completion_seeds(g, X, y, 0)[1]
    proposals = completion._compositional_proposals(g, X, y, counts, None)
    return proposals, counts


def parent(errors=(.1, .1)):
    return Candidate(System((Vertex('x0'),), (0, 0)), np.array([]), np.array(errors), 1., 0., 0.)


def fake_evaluator(limit=100, expired=False, errors=(.01, .01)):
    base = parent()
    class Evaluation:
        cfg = config(compositional_completion=True, population_size=24, max_evaluations=limit)
        X, Y, scales = np.ones((4, 40)), np.ones((4, 2)), np.ones(2)
        deadline = None
        calls = 1
        def __init__(self):
            self.cache = {base.system:base}
        def expired(self):
            return expired
        def __call__(self, system, warm):
            assert system not in self.cache
            self.calls += 1
            child = Candidate(system, warm.copy(), np.array(errors), 1., 0., 0.)
            self.cache[system] = child
            return child
    return base, Evaluation()


def stub_seeds(indices, kinds=None):
    kinds = kinds or ['compositional']*len(indices)
    return [(Node('x'+str(i)), np.array([])) for i in indices], dict(
        returned_sources=[dict(kind=k, returned_index=j) for j, k in enumerate(kinds)],
        compositional_linear_subproblems=2, implicit_linear_subproblems=3, linear_subproblems=5)


def test_opt_in_strict_config_and_mutual_exclusion():
    assert not SearchConfig().compositional_completion
    assert SearchConfig().initialization=='legacy' and SearchConfig().sharing_strategy=='random'
    config(compositional_completion=True).validate()
    for value in (1, 'true', None):
        with pytest.raises(ValueError, match='boolean'):
            config(compositional_completion=value).validate()
    for field, value in [('initialization', 'implicit'), ('sharing_strategy', 'random'),
                         ('final_selection', 'guarded'), ('use_sharing', False),
                         ('use_units', False), ('use_constraints', False), ('precision_repair', True)]:
        with pytest.raises(ValueError):
            replace(config(compositional_completion=True), **{field:value}).validate()


@pytest.mark.parametrize('token,form,x,y,reason', [
    ('log','direct',None,[1000.], 'invalid_inverse'),
    ('log','direct',None,[-1000.], 'invalid_inverse'),
    ('sqrt','direct',None,[-.1], 'negative_sqrt_response'),
    ('sqrt','direct',None,[1e200], 'invalid_inverse'),
    ('log','multiply',[1e-13],[1.], 'near_zero_divisor'),
    ('log','divide',[1.],[1e-13], 'near_zero_divisor'),
    ('sqrt','multiply',[1.],[-1.], 'negative_sqrt_response'),
    ('sqrt','divide',[-1.],[1.], 'negative_sqrt_response'),
    ('log','direct',None,[np.nan], 'nonfinite_latent')])
def test_invalid_inverse_is_rejected_without_clipping(token, form, x, y, reason):
    target, actual = completion._inverse_target(token, form, None if x is None else np.array(x), np.array(y))
    assert target is None and actual==reason


def test_sqrt_zero_and_variable_division_inverse():
    target, reason = completion._inverse_target('sqrt', 'direct', None, np.array([0., 2.]))
    np.testing.assert_array_equal(target, [0., 4.]); assert reason is None
    for form, y in [('multiply', np.array([2., 6.])), ('divide', np.array([.5, 2/3]))]:
        target, reason = completion._inverse_target('sqrt', form, np.array([1., 2.]), y)
        np.testing.assert_allclose(target, [4., 9.]); assert reason is None


def test_square_weight_absorption_and_native_overflow_semantics():
    g = grammar()
    constants = {ZERO:Node('c0')}
    x = Node('x0'); square = operator('square', x)
    X = np.array([[.3], [1.7]])
    for weight in (0., .4, 3.2):
        tree, c = completion._weighted(square, np.array([]), weight, constants, g)
        np.testing.assert_allclose(tree.evaluate(X, c), weight*X[:, 0]**2)
        assert tree.token=='square'
    inner = operator('square', operator('mul', Node('c0'), x))
    tree, c = completion._weighted(inner, np.array([2.3]), .7, constants, g)
    np.testing.assert_allclose(c, [2.3*np.sqrt(.7)])
    np.testing.assert_allclose(tree.evaluate(X, c), .7*(2.3*X[:, 0])**2)
    negative, c = completion._weighted(square, np.array([]), -2., constants, g)
    assert negative.token=='mul'
    np.testing.assert_allclose(negative.evaluate(X, c), -2*X[:, 0]**2)
    old = operator('mul', Node('c0'), square)
    new, c = completion._weighted(square, np.array([]), 0., constants, g)
    extreme = np.array([[1e200]])
    assert np.isnan(old.evaluate(extreme, [0.])).all()
    assert new.evaluate(extreme, c)[0]==0.
    assert completion._score(old, [0.], extreme, np.zeros(1)) is None
    assert completion._score(new, c, extreme, np.zeros(1)) is not None
    from dimensr_mimo._single.intervals import audit_box
    assert audit_box(old, [0.], [[1e200, 2e200]])['status']=='invalid'
    assert audit_box(new, c, [[1e200, 2e200]])['status']=='enclosed'


def test_basis_fixed_order_and_24_cap_with_16_outer_variables():
    g = grammar(features=(ZERO,)*18)
    X = np.random.default_rng(88).uniform(.2, 1.4, (10, 18))
    counts = completion.completion_seeds(g, X, X[:, 0], 0)[1]
    variables, basis = completion._basis(g, X, {ZERO:Node('c0')}, counts, None)
    assert len(variables)==16 and len(basis)==24
    assert [t.token for t, _, _ in basis[:18]]==['x'+str(i) for i in range(18)]
    assert [t.token for t, _, _ in basis[18:]]==['square']*6


def test_dimensioned_fractional_units_and_missing_constants_or_operators():
    length = unit([1, 0, 0, 0, 0, 0, 0])
    half = scale(length, Fraction(1, 2))
    g = grammar((length, length), half, (ZERO,))
    X = np.random.default_rng(9).uniform(.3, 1.4, (12, 2))
    proposals, counts = new_proposals(g, X, np.sqrt(.4*X[:, 0]+.7*X[:, 1]))
    assert proposals and counts['compositional_unit_rejections']>0
    for tree, c, source in proposals:
        assert tree.units==half and source['transform']=='sqrt'
        assert completion._legal(tree, g) and np.isfinite(tree.evaluate(X, c)).all()
    # No inverse-length constant: do not invent dimensionless log arguments.
    g = grammar((length,), length, (ZERO,))
    proposals, _ = new_proposals(g, X[:, :1], X[:, 0]*np.log(1+X[:, 0]**2))
    assert not any(s['transform']=='log' for _, _, s in proposals)
    no_unary = SystemGrammar((ZERO,), (ZERO, ZERO), (), config(operators=('mul',))).plain[0]
    assert new_proposals(no_unary, X[:, :1], X[:, 0])[0]==[]
    no_constants = grammar((ZERO,), ZERO, ())
    proposals, _ = new_proposals(no_constants, X[:, :1], np.sqrt(X[:, 0]))
    assert all(t.n_constants==0 and not s['outer_amplitude'] for t, _, s in proposals)


def test_original_y_ranking_and_actual_winning_constant_source(monkeypatch):
    g = grammar((ZERO,))
    tree = operator('log', Node('c0'))
    X = np.array([[1.], [2.]])
    y = np.log([1., 9.])
    monkeypatch.setattr(completion, 'structured_seeds', lambda *a, **k:
        ([(tree, np.array([5.]))], dict(implicit_retained_indices=[0])))
    monkeypatch.setattr(completion, '_compositional_proposals', lambda *a:
        [(tree, np.array([3.]), dict(kind='compositional', transform='log'))])
    seeds, stats = completion.completion_seeds(g, X, y, 12)
    assert len(seeds)==1 and seeds[0][1][0]==3.
    assert stats['returned_sources'][0]['kind']=='compositional'
    assert stats['implicit_retained_indices']==[]
    # In inverse space 5 is better; strict original Y correctly chooses 3.
    assert np.mean((5-np.exp(y))**2) < np.mean((3-np.exp(y))**2)
    monkeypatch.setattr(completion, '_compositional_proposals', lambda *a:
        [(tree, np.array([30.]), dict(kind='compositional'))])
    seeds, stats = completion.completion_seeds(g, X, y, 12)
    assert seeds[0][1][0]==5. and stats['returned_sources'][0]['kind']=='implicit'
    assert stats['compositional_retained_indices']==[]


def test_bare_and_external_amplitude_variants_and_bounds():
    X = np.random.default_rng(13).uniform(.2, 1.2, (14, 2))
    y = X[:, 0]*np.log(1.3+.4*X[:, 1]**2)
    g = grammar(max_nodes=15, max_depth=6)
    proposals, counts = new_proposals(g, X, y)
    assert proposals and counts['compositional_linear_subproblems']>0
    assert any(not s['outer_amplitude'] and s['form']=='multiply' for _, _, s in proposals)
    amplitudes = [(t,c,s) for t,c,s in proposals if s['outer_amplitude'] and s['form']=='multiply'
                  and s['variable']=='x0' and s['transform']=='log']
    assert amplitudes
    for tree, c, source in amplitudes:
        # Canonical order is explicit, so locate the constant on variable branch.
        branch_index = next(i for i,ch in enumerate(tree.children) if ch.token=='mul')
        variable_branch = tree.children[branch_index]
        assert any(ch.token=='x0' for ch in variable_branch.children)
        coefficient_index = 0
        for path, node in tree.walk():
            if node.token.startswith('c'):
                if path[0]==branch_index:
                    break
                coefficient_index += 1
        scaled = c.copy(); scaled[coefficient_index] = 2.7
        np.testing.assert_allclose(tree.evaluate(X, scaled), 2.7*tree.evaluate(X, c), rtol=1e-12, atol=1e-12)
    tight = grammar(max_nodes=5, max_depth=3)
    small, stats = new_proposals(tight, X, y)
    assert stats['compositional_structure_rejections']>0
    assert all(t.size<=5 and t.depth<=3 and completion._legal(t,tight) for t,_,_ in small)


def test_final_divisor_rejects_near_zero_even_when_inverse_labels_are_finite():
    X=np.column_stack((np.zeros(10),np.linspace(.2,1.2,10)))
    proposals,counts=new_proposals(grammar(),X,np.ones(10))
    for tree,c,source in proposals:
        if source['transform']=='log' and source['form']=='divide' and source['variable']=='x0':
            divisor=tree.children[1].evaluate(X,c[1:] if source['outer_amplitude'] else c)
            assert np.isfinite(divisor).all() and np.all(np.abs(divisor)>1e-12)
    assert counts['compositional_numeric_rejections']>0
    exact=operator('div',Node('x0'),operator('log',Node('c0')))
    assert completion._score(exact,[1.],X,np.ones(10)) is None


def test_nonunit_outer_amplitude_joint_fit_with_declared_inverse_units():
    length=unit([1,0,0,0,0,0,0]); duration=unit([0,0,1,0,0,0,0])
    inverse_duration=sub(ZERO,duration)
    cfg=config(compositional_completion=True,operators=('add','mul','square','log'),
               constant_max_nfev=100,constant_restarts=1,max_evaluations=10)
    g=SystemGrammar((length,duration),(length,length),(ZERO,inverse_duration),cfg)
    x=Node('x0',units=length); z=Node('x1',units=duration)
    inner=operator('add',Node('c0'),operator('square',operator('mul',Node('c1',units=inverse_duration),z)))
    tree=operator('mul',operator('mul',Node('c0'),x),operator('log',inner))
    g.plain[0].trace(tree)
    assert tree.size==11 and tree.depth==6
    X=np.random.default_rng(28).uniform(.2,1.3,(20,2))
    warm=np.array([1.,1.3,.6]); truth=np.array([2.4,1.3,.6])
    Y=np.column_stack((tree.evaluate(X,truth),tree.evaluate(X,truth)*.7))
    builder=Builder()
    builder.roots.extend((builder.tree(tree,warm),builder.tree(tree,warm)))
    graph,values=builder.finish()
    assert in_bounds(graph,cfg)
    from dimensr_mimo import audit
    audit.units(graph,(length,duration),(length,length),(ZERO,inverse_duration))
    evaluator=Evaluator(X,Y,cfg,input_bounds=[[.2,1.3],[.2,1.3]])
    candidate=evaluator(graph,values)
    assert candidate is not None and max(candidate.errors)<1e-8
    assert evaluator.calls==1 and evaluator.stats['constant_nfev']>0


@pytest.mark.parametrize('limit,expired,expected', [(100,False,12),(4,False,3),(1,False,0),(100,True,0)])
def test_complete_system_cap_global_budget_and_always_trigger(monkeypatch, limit, expired, expected):
    base, evaluator = fake_evaluator(limit, expired)
    base.errors[:] = 0.  # Training already meets the gate: completion still runs.
    monkeypatch.setattr(completion, 'completion_seeds', lambda *a, **k:stub_seeds(range(1, 21)))
    retained, report = completion.compositional_completion([base], evaluator, SimpleNamespace(plain=[0,1]))
    assert report['triggered'] and report['training_eligible']==1
    assert report['new_evaluations']==expected and evaluator.calls==1+expected
    assert all(e['compositional_outputs']==[0,1] and e['new_evaluation'] for e in report['events'])
    # The exact parent makes children fail protection; no relaxed threshold.
    assert not retained
    if expected:
        assert report['linear_subproblems']==14 and report['implicit_linear_subproblems']==6


def test_failed_successful_cache_short_lists_parent_protection_and_base_freeze(monkeypatch):
    base, evaluator = fake_evaluator(errors=(.09, .12))
    before = deepcopy(base.to_dict())
    monkeypatch.setattr(completion, 'completion_seeds', lambda g,*a,**k:
        stub_seeds([0,1,2,3], ['implicit','compositional','compositional','structured']) if g==0 else stub_seeds([]))
    for i, value in [(0,base),(1,None)]:
        builder = Builder()
        builder.roots.extend((builder.tree(Node('x'+str(i))), builder.tree(Node('x0'))))
        evaluator.cache[builder.finish()[0]] = value
    pool, report = sharing.protected_closure([base], evaluator, SimpleNamespace(plain=[0,1]))
    record = report['compositional_completion']
    assert record['cache_skips']==2 and record['nonimplicit_skips']==1
    assert record['new_evaluations']==record['valid']==record['protection_rejections']==1
    assert record['missing_roots']==3
    assert record['events'][0]['compositional_outputs']==[0]
    assert record['events'][0]['root_sources'][1]==dict(kind='parent')
    assert pool==[base] and base.to_dict()==before
    assert report['base_candidate_count']==1 and report['base_candidate_sha256']==digest([before])


def test_unequal_nonempty_lists_clone_parent_and_zero_limit(monkeypatch):
    base, evaluator = fake_evaluator()
    monkeypatch.setattr(completion, 'completion_seeds', lambda g,*a,**k:stub_seeds([1,2] if g==0 else [3]))
    pool, report = completion.compositional_completion([base], evaluator, SimpleNamespace(plain=[0,1]))
    assert len(pool)==2 and report['events'][1]['compositional_outputs']==[0]
    graph=pool[1].system
    assert [graph.nodes[r].token for r in graph.roots]==['x2','x0']
    monkeypatch.undo()
    evaluator.cfg.population_size=1
    pool, report=completion.compositional_completion([base],evaluator,SimpleNamespace(plain=[None,None]))
    assert not pool and report['seed_limit']==0


def test_scheduler_structure_rejection_and_deadline_keep_prior_retained(monkeypatch):
    base,evaluator=fake_evaluator()
    too_deep=Node('x2')
    for _ in range(8):
        too_deep=operator('square',too_deep)
    def seeds(g,*a,**k):
        roots,stats=stub_seeds([1,2,3])
        roots[0]=(too_deep,np.array([]))
        return roots,stats
    monkeypatch.setattr(completion,'completion_seeds',seeds)
    evaluator.expired=lambda:evaluator.calls>=2
    pool,report=completion.compositional_completion([base],evaluator,SimpleNamespace(plain=[0,1]))
    assert len(pool)==1 and report['structure_rejections']==1
    assert report['new_evaluations']==1 and report['stop_reason']=='deadline'
    assert base.system in evaluator.cache


@pytest.mark.parametrize('stage', ['basis', 'linear', 'old'])
def test_cooperative_deadline_keeps_partial_linear_work(monkeypatch, stage):
    g=grammar()
    X=np.random.default_rng(14).uniform(.2,1.2,(12,2)); y=np.log(1+X[:,0]**2)
    now=[0.]
    monkeypatch.setattr(completion.time,'monotonic',lambda:now[0])
    if stage=='old':
        def old(*a,**k):
            error=TimeoutError('deadline');error.initialization_diagnostics=dict(linear_subproblems=3,implicit_linear_subproblems=2)
            raise error
        monkeypatch.setattr(completion,'structured_seeds',old)
    else:
        monkeypatch.setattr(completion,'structured_seeds',lambda *a,**k:([],dict(linear_subproblems=4)))
        if stage=='basis':
            original=Node.evaluate
            def evaluate(*a,**k):
                now[0]=2.;return original(*a,**k)
            monkeypatch.setattr(Node,'evaluate',evaluate)
        else:
            original=np.linalg.lstsq
            def solve(*a,**k):
                now[0]=2.;return original(*a,**k)
            monkeypatch.setattr(np.linalg,'lstsq',solve)
    with pytest.raises(TimeoutError) as exc:
        completion.completion_seeds(g,X,y,12,deadline=1.)
    stats=exc.value.initialization_diagnostics
    assert stats['linear_subproblems']==(3 if stage=='old' else 4)
    if stage=='linear':assert stats['compositional_linear_subproblems']==1
    if stage=='basis':assert stats['compositional_feature_evaluations']==1


def test_timeout_keeps_base_and_does_not_swallow_programming_errors(monkeypatch):
    base, evaluator=fake_evaluator()
    def timeout(*a,**k):
        error=TimeoutError('deadline');error.initialization_diagnostics=dict(compositional_linear_subproblems=2,linear_subproblems=3)
        raise error
    monkeypatch.setattr(completion,'completion_seeds',timeout)
    pool,report=sharing.protected_closure([base],evaluator,SimpleNamespace(plain=[0,1]))
    assert pool==[base] and report['compositional_completion']['stop_reason']=='deadline'
    assert report['compositional_completion']['linear_subproblems']==5
    def bug(*a,**k):
        raise RuntimeError('implementation error')
    monkeypatch.setattr(completion,'completion_seeds',bug)
    with pytest.raises(RuntimeError,match='implementation error'):
        completion.compositional_completion([base],evaluator,SimpleNamespace(plain=[0,1]))


def test_real_joint_evaluator_and_native_domain_audit_preserve_base():
    cfg=config(compositional_completion=True,operators=('add','mul','square','sqrt'),population_size=8,
        constant_max_nfev=10,constant_restarts=1,max_evaluations=30)
    X=np.random.default_rng(21).uniform(.3,1.3,(18,2)); a,b=X.T
    Y=np.column_stack((np.sqrt(.6*a*a+.3*b*b),np.sqrt(.9*a*a+.5*b*b)))
    evaluator=Evaluator(X,Y,cfg,input_bounds=[[.3,1.3],[.3,1.3]])
    base=evaluator(parent().system,np.array([])); before=deepcopy(base.to_dict())
    g=SystemGrammar((ZERO,ZERO),(ZERO,ZERO),(ZERO,),cfg)
    pool,report=sharing.protected_closure([base],evaluator,g)
    record=report['compositional_completion']
    assert record['new_evaluations']>0 and record['protected']>0
    assert any(e['compositional_outputs'] for e in record['events'])
    assert base.to_dict()==before and pool[0] is base
    assert evaluator.calls==report['base_evaluations']+record['new_evaluations']+report['new_evaluations']
    assert len(pool)==1+record['protected']+report['protected']
    assert evaluator.stats['rademacher_systems']>1 and evaluator.stats['constant_nfev']>0
    assert evaluator.stats['interval_enclosed']>0


def test_full_base_identity_history_and_native_model_roundtrip(tmp_path):
    X=np.linspace(.2,1.2,18)[:,None];Y=np.column_stack((X[:,0],2*X[:,0]))
    cfg=config(population_size=4,epochs=1,gp_generations=1,selection_generations=2,
        max_evaluations=40,min_candidates=3,polish_candidates=0,constant_max_nfev=8,
        constant_restarts=1,max_nodes=7,operators=('add','mul','sqrt'),use_rnn=False)
    def fit(c):
        return DimenSRMIMORegressor(c).fit(X,Y,X_val=X[::3],Y_val=Y[::3],input_bounds=[[.2,1.2]])
    base, model=fit(cfg),fit(replace(cfg,compositional_completion=True))
    count=model.diagnostics_['sharing_closure']['base_candidate_count']
    base_count=base.diagnostics_['sharing_closure']['base_candidate_count']
    assert model.candidates_[:count]==base.candidates_[:base_count]
    record=model.diagnostics_['sharing_closure']['compositional_completion']
    history=next(h for h in model.history_ if h['phase']=='sharing_closure')
    assert history['completion_new_systems']==record['new_evaluations']
    assert history['completion_protected_systems']==record['protected']
    path=tmp_path/'completion.json';model.save(path)
    loaded=DimenSRMIMORegressor.load(path)
    assert json.loads(json.dumps(loaded.to_dict()))==json.loads(json.dumps(model.to_dict()))
    np.testing.assert_array_equal(loaded.predict(X),model.predict(X))
    payload=json.loads(path.read_text());payload['diagnostics']['sharing_closure']['base_candidate_sha256']='changed'
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match='base pool'):
        DimenSRMIMORegressor.load(path)


@pytest.mark.parametrize('relative', [
    'implicit_rational_20261028/runs/rational_n0_s702_compound/model.json',
    'precision_repair_20261028/runs/rational_n0_s702_repair/model.json'])
def test_old_saved_models_verify_before_default_migration(relative,tmp_path):
    root=Path(__file__).resolve().parents[1]
    path=root/'experiments'/relative
    saved=json.loads(path.read_text(encoding='utf-8'))
    assert 'compositional_completion' not in saved['config']
    model=DimenSRMIMORegressor.load(path)
    assert not model.config.compositional_completion and model.best_.to_dict()==saved['best']
    output=tmp_path/'migrated.json';model.save(output)
    assert DimenSRMIMORegressor.load(output).to_dict()==model.to_dict()
