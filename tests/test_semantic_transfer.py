import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from dimensr_mimo import SearchConfig
from dimensr_mimo.features import Feature, compose, evaluate_features
from dimensr_mimo.search import make_candidate, predict_candidate
from dimensr_mimo._vendor.expression import Node
from dimensr_mimo._vendor.units import ZERO, unit

directory=Path(__file__).resolve().parents[1]/"experiments/semantic_transfer_v1"
sys.path.insert(0,str(directory))
try:
    import transfer
    import study
    spec=importlib.util.spec_from_file_location("transfer_analysis",directory/"analyze.py")
    analysis=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis)
finally:
    sys.path.pop(0)


def head(features,X,y):
    return make_candidate(features,evaluate_features(features,X),y[:,None],np.array([y.std()]),(ZERO,),(ZERO,))


def test_subtrees_keep_their_own_constants_and_units():
    x=Feature(Node("x0"))
    a=compose("mul",Feature(Node("c0"),(2.,)),x)
    b=compose("sin",compose("mul",Feature(Node("c0"),(3.,)),x))
    full=compose("add",a,b)
    pieces=transfer.subexpressions(full)
    assert a in pieces and b in pieces and full in pieces
    X=np.array([[.2],[.7]])
    np.testing.assert_allclose(evaluate_features((b,),X)[:,0],np.sin(3*X[:,0]))


def test_semantic_scores_are_sign_and_scale_invariant_and_handle_constants():
    r=np.array([-2.,-1.,1.,2.])
    values=np.column_stack((-3*r+10,np.ones(4),np.array([1.,-1.,-1.,1.])))
    np.testing.assert_allclose(transfer.semantic_scores(values,r),[1.,0.,0.],atol=1e-14)
    np.testing.assert_array_equal(transfer.semantic_scores(values,np.zeros(4)),np.zeros(3))


@pytest.mark.parametrize("rule",["random","semantic"])
def test_optional_transfer_recovers_missing_term_and_preserves_originals(rule):
    rng=np.random.default_rng(4);X=rng.uniform(-1,1,(90,2))
    Y=np.column_stack((X[:,0]+2*X[:,1]**2,3*X[:,1]**2))
    x=Feature(Node("x0"));square=compose("square",Feature(Node("x1")))
    pools=[[head((x,),X,Y[:,0])],[head((square,),X,Y[:,1])]]
    before=[predict_candidate(p[0],X).copy() for p in pools]
    cfg=SearchConfig(max_terms=2,head_ridge=0.,search_seconds=2.)
    result,counts=transfer.transfer(pools,X,Y,np.array([[-1.,1.]]*2),(ZERO,)*2,(ZERO,),cfg,seconds=2.,rule=rule)
    assert counts["accepted_per_output"][0]>0
    for old,new,pred in zip(pools,result,before):
        assert new[0] is old[0]
        np.testing.assert_array_equal(predict_candidate(old[0],X),pred)
    best=min(result[0],key=lambda c:c["train_loss"])
    np.testing.assert_allclose(predict_candidate(best,X)[:,0],Y[:,0],atol=1e-12)
    assert best["donor_outputs"]==[1]


def test_illegal_donor_is_rejected_before_subtree_extraction():
    X=np.linspace(-1,1,51)[:,None];Y=np.column_stack((X[:,0],X[:,0]**2))
    x=Feature(Node("x0"));bad=compose("div",x,x)
    invalid=dict(features=(bad,),coefficients=np.ones((1,1)),intercepts=np.zeros(1),train_loss=1.,complexity=4)
    pools=[[head((x,),X,Y[:,0])],[invalid]]
    result,counts=transfer.transfer(pools,X,Y,np.array([[-1.,1.]]),(ZERO,)*2,(ZERO,),
        SearchConfig(search_seconds=1.),seconds=.5,rule="semantic")
    assert counts["domain_rejections"]>0
    assert counts["donor_counts"][0]==0
    assert result[0]==pools[0]


def test_incompatible_units_do_not_cross_output_boundary():
    X=np.linspace(1,2,30)[:,None];Y=np.column_stack((X[:,0],X[:,0]**2))
    length=unit([1,0,0,0,0,0,0]);time_unit=unit([0,0,1,0,0,0,0])
    donor=compose("square",Feature(Node("x0",units=length)))
    pools=[[head((),X,Y[:,0])],[dict(head((),X,Y[:,1]),features=(donor,),coefficients=np.ones((1,1)))]]
    _,counts=transfer.transfer(pools,X,Y,np.array([[1.,2.]]),(time_unit,donor.tree.units),(ZERO,),
        SearchConfig(search_seconds=1.),seconds=.5,rule="semantic")
    assert counts["donor_counts"][0]==0


def test_exhausted_budget_returns_unmodified_original_candidates():
    X=np.arange(5.)[:,None];Y=np.column_stack((X[:,0],2*X[:,0]))
    pools=[[head((),X,Y[:,j])] for j in range(2)]
    result,counts=transfer.transfer(pools,X,Y,np.array([[0.,4.]]),(ZERO,)*2,(ZERO,),
        SearchConfig(search_seconds=1.),seconds=1e-12,rule="semantic")
    assert all(new[0] is old[0] for new,old in zip(result,pools))
    assert counts["head_fits"]==0


def test_validation_can_reject_every_transfer():
    def saved(feature,loss):
        return dict(features=[feature.to_dict()],coefficients=[[1.]],intercepts=[0.],
            train_loss=.01,validation_loss=loss,complexity=feature.tree.size+1)
    x=Feature(Node("x0"));donor=compose("square",x)
    archives=[[saved(x,.1),saved(donor,.2)] for _ in range(2)]
    assert transfer.shared.select_indices(archives,"compact")==[0,0]


def test_pipeline_freezes_all_choices_before_outer_scores(tmp_path,monkeypatch):
    rng=np.random.default_rng(5);X=rng.uniform(-1,1,(50,2));Y=np.column_stack((X[:,0]+X[:,1]**2,2*X[:,0]))
    split=dict(X=X[:30],Y=Y[:30],X_val=X[30:40],Y_val=Y[30:40],X_eval=X[40:],Y_eval=Y[40:],
        target_scale=Y[:30].std(axis=0),feature_names=["x0","x1"],target_names=["a","b"],metadata={"fold_sha256":"synthetic"})
    monkeypatch.setattr(study,"HERE",tmp_path);monkeypatch.setattr(study,"ROOT",tmp_path)
    old=tmp_path/"experiments/budget_capacity_v1";old.mkdir(parents=True)
    study.save(old/"SPLITS.json",{"concrete_slump_f0":split["metadata"]})
    monkeypatch.setattr(study,"check_freeze",lambda:None);monkeypatch.setattr(study,"sha",lambda p:"synthetic")
    monkeypatch.setattr(study.v3,"prepare_split",lambda *args:split)
    monkeypatch.setattr(study,"BASE_SECONDS",.05);monkeypatch.setattr(study,"EXTRA_SECONDS",.01)
    monkeypatch.setattr(study,"CONTROL_SECONDS",.06)
    out=tmp_path/"run";out.mkdir()
    original=study.v3.metrics;outer=[]
    def metrics(y,pred,scales):
        if y is split["Y_eval"]:
            import json
            assert set(json.loads((out/"CHOICES.json").read_text()))==set(study.METHODS)
            assert (out/"archives.json").exists()
            outer.append(True)
        return original(y,pred,scales)
    monkeypatch.setattr(study.v3,"metrics",metrics)
    result=study.fit_job(("concrete_slump",0,0),out)
    assert result["status"]=="ok" and len(outer)==4


def groups(error,k):
    def group(e,size):
        return dict(complete=True,metrics={"outer_nrmse":{"mean":e},"complexity":{"mean":size}},
            datasets={d:dict(metrics={"outer_nrmse_0":{"mean":e}}) for d in study.DATASETS})
    return dict(semantic25=group(error,k),independent25=group(1.,40.),independent20=group(1.02,40.))


@pytest.mark.parametrize("error,k,passed",[(.99,44.,True),(1.005,36.,True),(.991,44.,False),(1.006,36.,False),(.99,44.1,False)])
def test_primary_gate_boundaries(error,k,passed):
    assert analysis.gate(groups(error,k))["eligible"] is passed


def test_output_damage_incomplete_pairs_and_base_regression_block_advancement():
    g=groups(.98,40.)
    g["semantic25"]["datasets"]["concrete_slump"]["metrics"]["outer_nrmse_0"]["mean"]=1.03
    assert not analysis.gate(g)["eligible"]
    g=groups(.98,40.);g["independent25"]["complete"]=False
    assert not analysis.gate(g)["eligible"]
    g=groups(.98,40.);g["independent20"]["metrics"]["outer_nrmse"]["mean"]=.97
    assert not analysis.gate(g)["eligible"]
    pairs=analysis.paired([],"semantic25","independent25")
    assert len(pairs)==45 and not analysis.aggregate(pairs,45)["complete"]
