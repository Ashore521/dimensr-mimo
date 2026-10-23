import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

directory = Path(__file__).resolve().parents[1] / "experiments/real_blind_v3"
spec = importlib.util.spec_from_file_location("holdout_v3", directory / "holdout.py")
holdout = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(directory))
try:
    spec.loader.exec_module(holdout)
finally:
    sys.path.pop(0)


def test_group_bootstrap_uses_paired_errors_and_keeps_seed_means():
    y = np.zeros((12, 2))
    groups = np.repeat(np.arange(3), 4)
    before = [np.ones_like(y), -np.ones_like(y)]
    identical = holdout.paired_interval(y, before, before, np.ones(2), groups)
    better = holdout.paired_interval(y, before, [np.zeros_like(y)], np.ones(2), groups)
    assert identical["after_minus_joint_nrmse_95ci"] == [0., 0.]
    assert better["after_minus_joint_nrmse_95ci"] == [-1., -1.]
    assert better["groups"] == 3 and better["resamples"] == 2000


def test_changed_frozen_artifact_prevents_opening(tmp_path, monkeypatch):
    monkeypatch.setattr(holdout, "ROOT", tmp_path)
    source = tmp_path / "model.json"
    source.write_text("changed")
    freeze = tmp_path / "FREEZE.json"
    holdout.write_json(freeze, {"files": {"model.json": "not-the-current-hash"}})
    destination = tmp_path / "holdout"
    with pytest.raises(ValueError, match="Frozen file changed"):
        holdout.evaluate(freeze, destination)
    assert not destination.exists()


def test_study_cannot_reopen_under_another_output_path(tmp_path, monkeypatch):
    monkeypatch.setattr(holdout, "ROOT", tmp_path)
    monkeypatch.setattr(holdout, "HERE", tmp_path)
    freeze = tmp_path / "FREEZE.json"
    holdout.write_json(freeze, {"files": {}, "models": []})
    monkeypatch.setattr(holdout.subprocess, "check_output", lambda *a, **k: freeze.read_bytes())
    (tmp_path / "HOLDOUT_OPENED.json").write_text("previous opening")
    monkeypatch.setattr(holdout.np, "load", lambda *a, **k: pytest.fail("Must not load sealed arrays"))
    with pytest.raises(FileExistsError):
        holdout.evaluate(freeze, tmp_path / "different-output")
    assert not (tmp_path / "different-output").exists()
