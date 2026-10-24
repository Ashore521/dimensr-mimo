import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

directory = Path(__file__).resolve().parents[1]/"experiments/contextual_transfer_v1"
sys.path.insert(0,str(directory))
try:
    import context_study as study
    spec = importlib.util.spec_from_file_location("contextual_analysis_test",directory/"analyze.py")
    analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis)
finally:
    sys.path.pop(0)


def groups(error,k):
    def group(e,size):
        return dict(complete=True,metrics={"outer_nrmse":{"mean":e},"complexity":{"mean":size}},
            datasets={d:dict(metrics={"outer_nrmse_0":{"mean":e}}) for d in study.DATASETS})
    return dict(context25=group(error,k),independent25=group(1.,40.),whole25=group(.97,40.),
        independent20=group(1.02,40.))


@pytest.mark.parametrize("error,k,passed",[(.99,44.,True),(1.005,36.,True),(.991,44.,False),(1.006,36.,False),(.99,44.1,False)])
def test_advancement_gate_boundaries(error,k,passed):
    assert analysis.gate(groups(error,k))["eligible"] is passed


def test_primary_success_does_not_establish_internal_graft_attribution():
    g = groups(.98,40.)
    assert analysis.gate(g)["eligible"]
    assert not analysis.gate(g,"whole25")["eligible"]
    g["context25"]["datasets"]["concrete_slump"]["metrics"]["outer_nrmse_0"]["mean"] = 1.03
    assert not analysis.gate(g)["eligible"]
    g = groups(.98,40.);g["independent25"]["complete"] = False
    assert not analysis.gate(g)["eligible"]
    assert not analysis.aggregate(analysis.paired([],"context25","independent25"),45)["complete"]


def test_freeze_rejects_changed_input_and_writer_refuses_overwrite(tmp_path,monkeypatch):
    monkeypatch.setattr(study,"HERE",tmp_path);monkeypatch.setattr(study,"ROOT",tmp_path)
    source = tmp_path/"source.py";source.write_text("original")
    study.save(tmp_path/"FREEZE.json",{"files":{"source.py":study.sha(source)}})
    study.check_freeze()
    source.write_text("changed")
    with pytest.raises(ValueError,match="Frozen input changed"):
        study.check_freeze()
    with pytest.raises(FileExistsError):
        study.save(tmp_path/"FREEZE.json",{})


def test_all_training_finishes_and_choices_are_saved_before_outer_scoring(tmp_path,monkeypatch):
    rng = np.random.default_rng(7);X = rng.uniform(-1,1,(60,2))
    Y = np.column_stack((np.sin(X[:,0]+X[:,1]),X[:,0]+X[:,1]))
    split = dict(X=X[:30],Y=Y[:30],X_val=X[30:45],Y_val=Y[30:45],X_eval=X[45:],Y_eval=Y[45:],
        target_scale=Y[:30].std(axis=0),feature_names=["x0","x1"],target_names=["a","b"],metadata={"fold_sha256":"synthetic"})
    monkeypatch.setattr(study,"HERE",tmp_path);monkeypatch.setattr(study,"ROOT",tmp_path)
    old = tmp_path/"experiments/budget_capacity_v1";old.mkdir(parents=True)
    study.save(old/"SPLITS.json",{"concrete_slump_f0":split["metadata"]})
    monkeypatch.setattr(study,"check_freeze",lambda:None)
    monkeypatch.setattr(study,"sha",lambda p:"synthetic")
    monkeypatch.setattr(study.v3,"prepare_split",lambda *args:split)
    monkeypatch.setattr(study,"BASE_SECONDS",.06)
    monkeypatch.setattr(study,"EXTRA_SECONDS",.03)
    monkeypatch.setattr(study,"CONTROL_SECONDS",.09)
    calls = []
    original_search = study.search
    def search(*args,**kwargs):
        calls.append("joint-trained")
        return original_search(*args,**kwargs)
    monkeypatch.setattr(study,"search",search)
    validate = study.shared.validate_archives
    def validate_after_training(*args,**kwargs):
        assert calls == ["joint-trained"]
        return validate(*args,**kwargs)
    monkeypatch.setattr(study.shared,"validate_archives",validate_after_training)
    out = tmp_path/"run";out.mkdir()
    original = study.v3.metrics;outer = []
    def metrics(y,pred,scales):
        if y is split["Y_eval"]:
            assert set(json.loads((out/"CHOICES.json").read_text())) == set(study.METHODS)
            assert (out/"archives.json").exists()
            outer.append(True)
        return original(y,pred,scales)
    monkeypatch.setattr(study.v3,"metrics",metrics)
    result = study.fit_job(("concrete_slump",0,0),out)
    assert result["status"] == "ok" and len(outer) == 6
    assert all(v["training_domain"]=="enclosed" for v in result["variants"].values())


def test_incomplete_run_is_preserved(tmp_path,monkeypatch):
    monkeypatch.setattr(study,"HERE",tmp_path)
    job = ("concrete_slump",0,0)
    run = tmp_path/"runs"/study.name(job);run.mkdir(parents=True)
    marker = run/"started.json";marker.write_text("original partial run")
    with pytest.raises(FileExistsError,match="Incomplete run preserved"):
        study.launch(job)
    assert marker.read_text() == "original partial run"
