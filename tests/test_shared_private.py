import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from dimensr_mimo import MIMOSymbolicRegressor, SearchConfig
from dimensr_mimo.features import Feature, compose
from dimensr_mimo.regressor import _serial_candidate
from dimensr_mimo.search import make_candidate, predict_candidate
from dimensr_mimo._vendor.expression import Node
from dimensr_mimo._vendor.units import ZERO

directory = Path(__file__).resolve().parents[1]/"experiments/shared_private_v1"
sys.path.insert(0,str(directory))
try:
    import hybrid
    spec = importlib.util.spec_from_file_location("shared_private_run",directory/"run.py")
    study = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(study)
finally:
    sys.path.pop(0)


def saved(feature, loss):
    return dict(features=[feature.to_dict()],coefficients=[[1.]],intercepts=[0.],
                train_loss=loss,validation_loss=loss,complexity=hybrid.graph_size([feature])+1)


def test_compact_selection_credits_shared_graph_and_obeys_each_output_tolerance():
    x,y = Feature(Node("x0")),Feature(Node("x1"))
    common = compose("mul",x,y)
    archives = [[saved(compose("square",f),1.),saved(common,1.005)] for f in (x,y)]
    assert hybrid.select_indices(archives,"penalty") == [0,0]
    assert hybrid.select_indices(archives,"compact") == [1,1]
    assert hybrid.combine(archives,[0,0])["complexity"] == 6
    assert hybrid.combine(archives,[1,1])["complexity"] == 5
    archives[0][1]["validation_loss"] = 1.02
    assert hybrid.select_indices(archives,"compact")[0] == 0
    for candidates in archives:
        for candidate in candidates:
            candidate["outer_score"] = -1e9
    assert hybrid.select_indices(archives,"compact")[0] == 0


@pytest.mark.parametrize("token",["div","log","sqrt"])
def test_raw_domain_audit_rejects_unsafe_basis_even_with_zero_weight(token):
    x = Feature(Node("x0"))
    f = compose(token,x,x) if token=="div" else compose(token,x)
    candidate = dict(features=(f,),coefficients=np.array([[0.]]),intercepts=np.array([0.]),train_loss=0.,complexity=3)
    assert not hybrid.enclosed(candidate,np.array([[-1.,1.]]),(ZERO,),32)


def test_shared_private_training_refits_both_components_without_validation(monkeypatch,tmp_path):
    rng = np.random.default_rng(42)
    X = rng.uniform(-1,1,(100,3))
    Y = np.column_stack((2*X[:,0]+3*X[:,1]**2, -X[:,0]+4*np.sin(X[:,2])))
    common = Feature(Node("x0"))
    private = [compose("square",Feature(Node("x1"))),compose("sin",Feature(Node("x2")))]
    calls = []
    def fake_search(X,response,bounds,fu,tu,cu,cfg):
        index = len(calls)
        calls.append(response.copy())
        feature = common if index==0 else private[index-1]
        scale = hybrid.scales_of(response)
        cand = make_candidate((feature,),hybrid.evaluate_features((feature,),X),response,scale,tu,cu)
        return [cand],scale,[],dict(search_cpu_seconds=0.,head_fits=1,budget_overshoot_seconds=0.)
    monkeypatch.setattr(hybrid,"search",fake_search)
    cfg = SearchConfig(max_terms=8,search_seconds=20.,head_ridge=0.)
    bounds = np.array([[-1.,1.]]*3)
    pools,counts = hybrid.search_hybrid(X,Y,bounds,(ZERO,)*3,(ZERO,)*2,(ZERO,),cfg)
    anchor = hybrid.decode(counts["shared_anchor"])
    residual = Y-predict_candidate(anchor,X)
    np.testing.assert_allclose(calls[1],residual[:,:1])
    np.testing.assert_allclose(calls[2],residual[:,1:])
    archives = hybrid.validate_archives(pools,X,Y,hybrid.scales_of(Y),bounds,(ZERO,)*2,32)
    choices = hybrid.select_indices(archives,"compact")
    combined = hybrid.combine(archives,choices)
    np.testing.assert_allclose(predict_candidate(combined,X),Y,atol=1e-12)
    assert np.count_nonzero(np.count_nonzero(combined["coefficients"],axis=0)>1)==1
    assert np.count_nonzero(np.count_nonzero(combined["coefficients"],axis=0)==1)==2
    model = hybrid.inference_model(combined,cfg,bounds,hybrid.scales_of(Y),["x0","x1","x2"],["a","b"],(ZERO,)*3,(ZERO,)*2,(ZERO,))
    path = tmp_path/"model.json"
    model.save(path)
    restored = MIMOSymbolicRegressor.load(path)
    np.testing.assert_array_equal(restored.predict(X),model.predict(X))


@pytest.mark.parametrize("arm",study.ARMS)
def test_full_synthetic_pipeline_saves_choices_before_outer_scores(tmp_path,monkeypatch,arm):
    rng = np.random.default_rng(4)
    X = rng.uniform(-1,1,(50,2));Y = np.column_stack((X[:,0]+X[:,1]**2,2*X[:,0]))
    split = dict(X=X[:30],Y=Y[:30],X_val=X[30:40],Y_val=Y[30:40],X_eval=X[40:],Y_eval=Y[40:],
        target_scale=Y[:30].std(axis=0),feature_names=["x0","x1"],target_names=["a","b"],metadata={"fold_sha256":"synthetic"})
    monkeypatch.setattr(study,"HERE",tmp_path)
    monkeypatch.setattr(study,"ROOT",tmp_path)
    old=tmp_path/"experiments/budget_capacity_v1";old.mkdir(parents=True)
    study.save(old/"SPLITS.json",{"concrete_slump_f0":split["metadata"]})
    monkeypatch.setattr(study,"check_freeze",lambda:None)
    monkeypatch.setattr(study,"sha",lambda p:"synthetic")
    monkeypatch.setattr(study.v3,"prepare_split",lambda *args:split)
    monkeypatch.setattr(study,"SearchConfig",lambda **kw:SearchConfig(**dict(kw,search_seconds=.03)))
    out=tmp_path/"run";out.mkdir()
    r=study.fit_job(("concrete_slump",arm,0,0),out)
    assert r["status"]=="ok"
    assert len(r["variants"])==(1 if arm=="joint" else 2)
    assert (out/"CHOICES.json").exists()
    assert all(v["outer"]["mean_nrmse"]>=0 for v in r["variants"].values())
