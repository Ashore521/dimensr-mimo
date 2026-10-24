import importlib.util
from pathlib import Path

import numpy as np
import pytest

from dimensr_mimo import SearchConfig, MIMOSymbolicRegressor
from dimensr_mimo.features import Feature, compose, evaluate_features
from dimensr_mimo.search import make_candidate, predict_candidate
from dimensr_mimo._vendor.expression import Node
from dimensr_mimo._vendor.units import ZERO, unit, scale

directory = Path(__file__).resolve().parents[1]/"experiments/contextual_transfer_v1"
spec = importlib.util.spec_from_file_location("contextual_under_test",directory/"contextual.py")
context = importlib.util.module_from_spec(spec)
spec.loader.exec_module(context)


def head(features, X, y, target=ZERO):
    return make_candidate(features,evaluate_features(features,X),y[:,None],
        context.shared.scales_of(y[:,None]),(target,),(ZERO,))


def test_graft_preserves_constants_outside_the_replaced_occurrence():
    x = Feature(Node("x0"))
    a = compose("mul",Feature(Node("c0"),(2.,)),x)
    b = compose("sin",compose("mul",Feature(Node("c0"),(3.,)),x))
    original = compose("sub",a,b)
    donor = compose("add",Feature(Node("c0"),(7.,)),x)
    changed = context.graft(original,(1,0),donor)
    X = np.array([[.2],[.7]])
    np.testing.assert_allclose(evaluate_features((changed,),X)[:,0],2*X[:,0]-np.sin(7+X[:,0]))
    np.testing.assert_allclose(evaluate_features((original,),X)[:,0],2*X[:,0]-np.sin(3*X[:,0]))
    assert changed.tree.n_constants == len(changed.constants) == 2
    assert context.graft(original,(),donor) == donor
    with pytest.raises(ValueError,match="path"):
        context.graft(original,(9,),donor)


def test_graft_rejects_wrong_units_and_changes_only_one_shared_occurrence():
    x = Feature(Node("x0")); h = compose("square",x)
    full = compose("add",h,h)
    changed = context.graft(full,(0,),x)
    X = np.array([[2.],[3.]])
    np.testing.assert_array_equal(evaluate_features((changed,),X)[:,0],X[:,0]+X[:,0]**2)
    length = unit([1,0,0,0,0,0,0])
    with pytest.raises(ValueError,match="units"):
        context.graft(full,(0,),Feature(Node("x0",units=length)))


@pytest.mark.parametrize("private",[False,True])
def test_internal_and_partial_sharing_recover_missing_nonlinear_context(private,tmp_path):
    rng = np.random.default_rng(6)
    X = rng.uniform(-1,1,(100,3))
    x0,x1,x2 = (Feature(Node(f"x{i}")) for i in range(3))
    donor = compose("add",x0,x1)
    Y = np.column_stack((np.sin(X[:,0]+X[:,1])+(X[:,2]**2 if private else 0),X[:,0]+X[:,1]))
    bases = (compose("sin",x0),)+( (compose("square",x2),) if private else () )
    pools = [[head(bases,X,Y[:,0])],[head((donor,),X,Y[:,1])]]
    before = predict_candidate(pools[0][0],X).copy()
    cfg = SearchConfig(max_terms=len(bases),head_ridge=0.,random_state=1)
    bounds = np.array([[-1.,1.]]*3)
    root,_ = context.transfer(pools,X,Y,bounds,(ZERO,)*2,(ZERO,),cfg,seconds=.3,internal=False)
    result,counts = context.transfer(pools,X,Y,bounds,(ZERO,)*2,(ZERO,),cfg,seconds=.5)
    assert min(c["train_loss"] for c in root[0]) > .001
    best = min(result[0],key=lambda c:c["train_loss"])
    assert best["train_loss"] < 1e-20 and counts["internal_accepted"] > 0
    assert result[0][0] is pools[0][0]
    np.testing.assert_array_equal(predict_candidate(pools[0][0],X),before)
    archives = context.shared.validate_archives(result,X,Y,Y.std(axis=0),bounds,(ZERO,)*2,32)
    combined = context.shared.combine(archives,context.shared.select_indices(archives,"compact"))
    model = context.shared.inference_model(combined,cfg,bounds,Y.std(axis=0),
        ["x0","x1","x2"],["y0","y1"],(ZERO,)*3,(ZERO,)*2,(ZERO,))
    model.save(tmp_path/"model.json")
    np.testing.assert_array_equal(MIMOSymbolicRegressor.load(tmp_path/"model.json").predict(X),model.predict(X))


def test_internal_matching_uses_subtree_units_not_output_units():
    rng = np.random.default_rng(2); X = rng.uniform(1,2,(70,2))
    length = unit([1,0,0,0,0,0,0]); area = scale(length,2)
    x0,x1 = (Feature(Node(f"x{i}",units=length)) for i in range(2))
    donor = compose("add",x0,x1)
    Y = np.column_stack(((X[:,0]+X[:,1])**2,X[:,0]+X[:,1]))
    pools = [[head((compose("square",x0),),X,Y[:,0],area)],[head((donor,),X,Y[:,1],length)]]
    result,counts = context.transfer(pools,X,Y,np.array([[1.,2.]]*2),(area,length),(ZERO,),
        SearchConfig(max_terms=1,head_ridge=0.),seconds=.5)
    assert min(c["train_loss"] for c in result[0]) < 1e-20
    assert counts["internal_accepted"] > 0


def test_legal_donor_cannot_make_an_illegal_denominator_context():
    X = np.array([[-.9],[-.5],[.5],[.9]])
    x = Feature(Node("x0")); one = Feature(Node("c0"),(1.,))
    base = compose("div",one,compose("add",one,compose("square",x)))
    donor = compose("square",x)
    Y = np.column_stack((1/X[:,0]**2,X[:,0]**2))
    pools = [[head((base,),X,Y[:,0])],[head((donor,),X,Y[:,1])]]
    bad = context.graft(base,(1,),donor)
    # All sampled rows are finite; the continuous box includes the hidden zero.
    assert np.isfinite(evaluate_features((bad,),X)).all()
    result,counts = context.transfer(pools,X,Y,np.array([[-1.,1.]]),(ZERO,)*2,(ZERO,),
        SearchConfig(max_terms=1,max_nodes=13),seconds=.5)
    assert counts["domain_rejections"] > 0
    assert all(bad not in c["features"] for c in result[0])
    assert all(context.shared.enclosed(c,np.array([[-1.,1.]]),(ZERO,),32) for c in result[0])


def test_unsafe_source_cannot_donate_and_exhausted_budget_is_noop(monkeypatch):
    X = np.linspace(-1,1,31)[:,None];Y = np.column_stack((X[:,0],X[:,0]**2))
    x = Feature(Node("x0"));bad = compose("div",x,x)
    invalid = dict(head((),X,Y[:,1]),features=(bad,),coefficients=np.ones((1,1)))
    pools = [[head((compose("sin",x),),X,Y[:,0])],[invalid]]
    cfg = SearchConfig(max_terms=1)
    _,counts = context.transfer(pools,X,Y,np.array([[-1.,1.]]),(ZERO,)*2,(ZERO,),cfg,seconds=.2)
    assert counts["donor_counts"][0] == 0 and counts["domain_rejections"] > 0
    from itertools import count
    ticks = count(0.,.1)
    monkeypatch.setattr(context.time,"process_time",lambda:next(ticks))
    result,counts = context.transfer(pools,X,Y,np.array([[-1.,1.]]),(ZERO,)*2,(ZERO,),cfg,seconds=1e-12)
    assert all(a[0] is b[0] for a,b in zip(result,pools)) and counts["head_fits"] == 0


def test_unshared_exact_models_remain_available_without_forced_coupling():
    rng = np.random.default_rng(9);X = rng.uniform(-1,1,(80,2))
    Y = np.column_stack((np.sin(X[:,0]),X[:,1]**2))
    bases = (compose("sin",Feature(Node("x0"))),compose("square",Feature(Node("x1"))))
    pools = [[head((f,),X,Y[:,j])] for j,f in enumerate(bases)]
    bounds = np.array([[-1.,1.]]*2)
    result,_ = context.transfer(pools,X,Y,bounds,(ZERO,)*2,(ZERO,),SearchConfig(max_terms=1),seconds=.3)
    archives = context.shared.validate_archives(result,X,Y,Y.std(axis=0),bounds,(ZERO,)*2,32)
    assert context.shared.select_indices(archives,"compact") == [0,0]


def test_limits_reject_oversize_grafts():
    rng = np.random.default_rng(3);X = rng.uniform(-1,1,(80,2))
    x0,x1 = Feature(Node("x0")),Feature(Node("x1"))
    donor = compose("add",x0,x1)
    Y = np.column_stack((np.sin(X[:,0]+X[:,1]),X[:,0]+X[:,1]))
    pools = [[head((compose("sin",x0),),X,Y[:,0])],[head((donor,),X,Y[:,1])]]
    result,counts = context.transfer(pools,X,Y,np.array([[-1.,1.]]*2),(ZERO,)*2,(ZERO,),
        SearchConfig(max_terms=1,max_nodes=3,max_depth=2),seconds=.3)
    assert counts["size_rejections"] > 0
    assert all(f.tree.size <= 3 and f.tree.depth <= 2 for p in result for c in p for f in c["features"])
