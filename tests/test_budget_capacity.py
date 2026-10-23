import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from dimensr_mimo import SearchConfig
from dimensr_mimo.features import Feature
from dimensr_mimo._vendor.expression import Node

directory = Path(__file__).resolve().parents[1] / "experiments/budget_capacity_v1"
sys.path.insert(0, str(directory))
try:
    import curves
    spec = importlib.util.spec_from_file_location("budget_diagnosis", directory / "diagnose.py")
    diagnosis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diagnosis)
finally:
    sys.path.pop(0)


def saved(features, weights, loss, intercept=0.):
    return dict(features=[f.to_dict() for f in features], coefficients=[weights],
                intercepts=[intercept], train_loss=loss, validation_loss=loss, complexity=0)


def test_independent_frontier_counts_exact_shared_nodes_and_connections():
    x = Feature(Node("x0"))
    a = [saved([], [], 2., 1.), saved([x], [1.], .1)]
    b = [saved([], [], 2., 1.), saved([x], [2.], .2)]
    points, count = curves.validation_frontier([a, b])
    assert count == 4
    assert points == [dict(complexity=2, validation_loss=2., indices=[0, 0]),
                      dict(complexity=3, validation_loss=pytest.approx(.15), indices=[1, 1])]
    assert curves.select_cap(points, 1) is None
    assert curves.select_cap(points, 2)["indices"] == [0, 0]
    assert curves.select_cap(points, 3)["indices"] == [1, 1]


def test_constant_values_prevent_false_sharing():
    archives = [[saved([Feature(Node("c0"), (value,))], [1.], .1)] for value in (1., 2.)]
    points, _ = curves.validation_frontier(archives)
    assert points[0]["complexity"] == 4


def test_outer_scores_cannot_select_curve_or_make_it_monotone():
    x, y = Feature(Node("x0")), Feature(Node("x1"))
    archives = [[saved([], [], 2., 1.), saved([x], [1.], 1.), saved([x, y], [1., 1.], .5)]]
    for candidate, outer in zip(archives[0], (0., 100., 200.)):
        candidate["outer_loss"] = outer
    before, _ = curves.validation_frontier(archives)
    for candidate in archives[0]:
        candidate["outer_loss"] *= -100
    after, _ = curves.validation_frontier(archives)
    assert before == after
    assert after[-1]["indices"] == [2]
    np.testing.assert_array_equal(curves.predict_tuple(archives, [2], np.array([[2., 3.]])), [[5.]])


def test_validation_tie_keeps_lower_complexity():
    archives = [[saved([], [], .2, 1.), saved([Feature(Node("x0"))], [1.], .2)]]
    points, _ = curves.validation_frontier(archives)
    assert len(points) == 1 and points[0]["complexity"] == 1


@pytest.mark.parametrize("arm", diagnosis.ARMS)
def test_synthetic_run_persists_archive_curves_and_model_reload(tmp_path, monkeypatch, arm):
    rng = np.random.default_rng(7)
    X = rng.normal(size=(40, 2))
    Y = np.column_stack((X[:, 0], X[:, 1] ** 2))
    split = dict(X=X[:24], Y=Y[:24], X_val=X[24:32], Y_val=Y[24:32],
                 X_eval=X[32:], Y_eval=Y[32:], target_scale=Y[:24].std(axis=0),
                 feature_names=["x0", "x1"], target_names=["y0", "y1"], metadata={"fold_sha256":"synthetic"})
    monkeypatch.setattr(diagnosis, "HERE", tmp_path)
    monkeypatch.setattr(diagnosis, "check_freeze", lambda: None)
    monkeypatch.setattr(diagnosis, "sha", lambda path: "synthetic")
    monkeypatch.setattr(diagnosis.v3, "prepare_split", lambda *args: split)
    monkeypatch.setattr(diagnosis, "SearchConfig", lambda **kw: SearchConfig(**dict(kw, search_seconds=.03)))
    diagnosis.save(tmp_path / "SPLITS.json", {"concrete_slump_f0": split["metadata"]})
    report = diagnosis.fit_job(("concrete_slump", arm, 20, 0, 0), tmp_path)
    assert report["status"] == "ok" and report["frontier"]
    assert all(p["status"] == "ok" for p in report["frontier"])
    assert len(report["models"]) == (2 if arm == "independent8" else 1)
    assert report["configs"][0]["max_terms"] == (16 if arm == "joint_wide" else 8)
    assert (tmp_path / "archives.json").exists()
