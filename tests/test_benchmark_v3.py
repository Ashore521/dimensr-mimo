import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dimensr_mimo.features import Feature, compose
from dimensr_mimo._vendor.expression import Node


spec = importlib.util.spec_from_file_location("benchmark_v3_runner", Path(__file__).resolve().parents[1]
                                           / "experiments/real_blind_v3/runner.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
summary_spec = importlib.util.spec_from_file_location("benchmark_v3_summary", Path(__file__).resolve().parents[1]
                                                    / "experiments/real_blind_v3/summarize_cv.py")
summary = importlib.util.module_from_spec(summary_spec)
summary_spec.loader.exec_module(summary)


@pytest.mark.parametrize("dataset", ["concrete_slump", "parkinsons", "air_quality"])
def test_nested_splits_keep_scoring_and_selection_separate(dataset):
    groups = np.repeat(np.arange(30), 20)
    outer_scored = []
    for fold in range(3):
        parts = runner.nested_indices(dataset, len(groups), groups, fold)
        train, val, evaluate = (parts[name] for name in ("train", "val", "eval"))
        sets = [set(index) for index in parts.values()]
        assert all(not sets[a] & sets[b] for a in range(3) for b in range(a + 1, 3))
        assert min(map(len, parts.values())) > 0
        if dataset == "parkinsons":
            sets = [set(groups[index]) for index in parts.values()]
            assert all(not sets[a] & sets[b] for a in range(3) for b in range(a + 1, 3))
        if dataset == "air_quality":
            assert val.min() - train.max() - 1 == 24
            assert evaluate.min() - val.max() - 1 == 24
        outer_scored.extend(evaluate)
    assert len(outer_scored) == len(set(outer_scored))
    if dataset != "air_quality":
        assert sorted(outer_scored) == list(range(len(groups)))


def test_prepare_split_uses_frozen_trainval_rows_and_training_only_scales(tmp_path, monkeypatch):
    rng = np.random.default_rng(10)
    X = rng.normal(size=(150, 2))
    Y = np.column_stack((3 * X[:, 0], np.full(len(X), 4.)))
    path = tmp_path / "concrete_slump.npz"
    ids, groups = np.array([f"row:{i}" for i in range(len(X))]), np.arange(len(X))
    np.savez(path, X_train=X[:100], X_val=X[100:], Y_train=Y[:100], Y_val=Y[100:],
             train_ids=ids[:100], val_ids=ids[100:], train_groups=groups[:100], val_groups=groups[100:])
    metadata = dict(feature_names=["a", "b"], target_names=["c", "d"], trainval_sha256=runner.sha(path))
    (tmp_path / "manifest.json").write_text(json.dumps({"datasets": {"concrete_slump": metadata}}))
    monkeypatch.setattr(runner, "DATA_DIR", tmp_path)
    split = runner.prepare_split("concrete_slump", 1)
    repeated = runner.prepare_split("concrete_slump", 1)
    info = split["metadata"]
    chosen = info["pool_indices"]["train"]
    np.testing.assert_allclose(info["input_mean"], X[chosen].mean(axis=0))
    np.testing.assert_allclose(info["input_scale"], X[chosen].std(axis=0))
    np.testing.assert_allclose(split["X"].mean(axis=0), 0., atol=1e-14)
    np.testing.assert_allclose(split["target_scale"], [Y[chosen, 0].std(), 4.])
    assert repeated["metadata"]["fold_sha256"] == info["fold_sha256"]
    assert set(info["row_ids"]["eval"]).isdisjoint(info["row_ids"]["val"])
    assert set(info["row_ids"]["eval"]).isdisjoint(info["row_ids"]["train"])


def test_direct_trainval_caps_rows_before_scaling_and_does_not_need_test(tmp_path):
    rng = np.random.default_rng(9)
    X = rng.normal(size=(1300, 2))
    Y = np.column_stack((X[:, 0], X[:, 1]))
    path = tmp_path / "trainval.npz"
    np.savez(path, X_train=X, Y_train=Y, X_val=np.full((12, 2), 1e6), Y_val=np.full((12, 2), -1e6))
    split = runner.prepare_trainval(path, dict(feature_names=["a", "b"], target_names=["c", "d"]), dataset="new")
    info = split["metadata"]
    chosen = info["fitting_subset_positions"]
    assert len(chosen) == 1200
    assert "X_eval" not in split and "Y_eval" not in split
    np.testing.assert_allclose(info["input_mean"], X[chosen].mean(axis=0))
    np.testing.assert_allclose(split["target_scale"], Y[chosen].std(axis=0))


def test_common_dag_gives_independent_models_the_same_reuse_credit():
    feature = compose("square", Feature(Node("x0")))
    joint = SimpleNamespace(features_=[feature], coef_=np.array([[1.], [2.]]), intercept_=np.array([0., 1.]))
    independent = [SimpleNamespace(features_=[feature], coef_=np.array([[value]]), intercept_=np.array([offset]))
                   for value, offset in ((1., 0.), (2., 1.))]
    a, b = runner.model_complexity([joint]), runner.model_complexity(independent)
    assert a["complexity"] == b["complexity"] == 5
    assert a["storage_complexity"] == 5
    assert b["storage_complexity"] == 7
    constants = [SimpleNamespace(features_=[Feature(Node("c0"), (value,))], coef_=np.ones((1, 1)), intercept_=np.zeros(1))
                 for value in (1., 2.)]
    assert runner.model_complexity(constants)["shared_dag_nodes"] == 2


def synthetic_split():
    rng = np.random.default_rng(21)
    X = rng.normal(size=(35, 2))
    Y = np.column_stack((X[:, 0] + 3., 2 * X[:, 1] - 1., X.sum(axis=1)))
    parts = {name: (X[section], Y[section], np.array([f"{name}:{i}" for i in range(len(X[section]))]),
                    np.arange(len(X[section])))
             for name, section in (("train", slice(0, 20)), ("val", slice(20, 27)), ("eval", slice(27, None)))}
    return runner._standardize(parts, dict(feature_names=["a", "b"], target_names=["c", "d", "e"]),
                               "synthetic", 0, "synthetic")


def test_independent_divides_total_budget_and_shares_training_rows(monkeypatch):
    calls = []

    class MeanEstimator:
        def __init__(self, config, feature_names, target_names):
            self.config = config

        def fit(self, X, Y, *, X_val, Y_val):
            calls.append((self.config, X.copy(), Y.copy(), X_val.copy(), Y_val.copy()))
            self.features_, self.coef_, self.intercept_ = [], np.empty((Y.shape[1], 0)), Y.mean(axis=0)
            self.diagnostics_, self.history_, self.expressions_ = {}, [], ["mean"] * Y.shape[1]
            return self

        def predict(self, X):
            return np.tile(self.intercept_, (len(X), 1))

        def audit_domain(self, bounds=None):
            return dict(status="enclosed")

    monkeypatch.setattr(runner, "MIMOSymbolicRegressor", MeanEstimator)
    split = synthetic_split()
    report, models = runner.train_run(split, "independent", 3, budget=20.)
    assert report["status"] == "ok"
    assert len(models) == len(calls) == 3
    assert report["allocated_search_seconds"] == [20 / 3] * 3
    for j, (config, X, Y, X_val, Y_val) in enumerate(calls):
        assert config.search_seconds == 20 / 3 and config.random_state == 3
        assert config.candidate_quota == 0 and config.prune_tolerance == 0
        np.testing.assert_array_equal(X, split["X"])
        np.testing.assert_array_equal(X_val, split["X_val"])
        np.testing.assert_array_equal(Y, split["Y"][:, j:j + 1])
        np.testing.assert_array_equal(Y_val, split["Y_val"][:, j:j + 1])
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("method", runner.METHODS)
def test_synthetic_single_run_reports_metrics_and_raw_audit(method):
    report, models = runner.train_run(synthetic_split(), method, 0, budget=.12)
    assert report["status"] == "ok", report.get("error")
    assert len(models) == (3 if method == "independent" else 1)
    assert len(report["eval"]["nrmse"]) == 3
    assert all(item["status"] == "enclosed" for item in report["raw_domain"]["training"])
    assert report["search_cpu_seconds"] > 0
    json.dumps(report, allow_nan=False)


@pytest.fixture
def cv_records(tmp_path):
    values = {"joint": (1., 100), "independent": (1.1, 130), "coverage": (.98, 105), "prune": (1.002, 80)}
    for dataset, outputs in summary.DATASETS.items():
        for method in summary.METHODS:
            for fold in range(3):
                for seed in range(5):
                    folder = tmp_path / f"{dataset}_{method}_f{fold}_s{seed}"
                    folder.mkdir()
                    nrmse, complexity = values[method]
                    record = dict(dataset=dataset, method=method, fold=fold, seed=seed, status="ok",
                                  fold_sha256=str(fold) * 64, eval=dict(nrmse=[nrmse] * outputs),
                                  complexity=complexity, storage_complexity=complexity + 10,
                                  search_cpu_seconds=20.01, budget_overshoot_seconds=.01)
                    (folder / "result.json").write_text(json.dumps(record), encoding="utf-8")
    return tmp_path


def test_summary_applies_global_gates_and_never_averages_away_failures(cv_records):
    report = summary.summarize(cv_records)
    assert report["counts"] == dict(ok=180, failed=0, timeout=0, missing=0)
    assert report["selected_method"] == "coverage"
    assert report["gates"]["coverage"]["eligible"] and report["gates"]["prune"]["eligible"]
    assert report["methods"]["coverage"]["S"] == pytest.approx(.98)
    assert report["methods"]["prune"]["K"] == 80
    folder = cv_records / "concrete_slump_coverage_f0_s0"
    timeout = dict(dataset="concrete_slump", method="coverage", fold=0, seed=0, status="timeout")
    (folder / "timeout.json").write_text(json.dumps(timeout), encoding="utf-8")
    report = summary.summarize(cv_records)
    assert report["counts"]["timeout"] == 1
    assert report["methods"]["coverage"]["S"] is None
    assert not report["gates"]["coverage"]["eligible"]
    assert report["selected_method"] == "prune"
    (cv_records / "concrete_slump_joint_f0_s0/result.json").unlink()
    report = summary.summarize(cv_records)
    assert report["counts"]["missing"] == 1
    assert report["selected_method"] is None
    (cv_records / "concrete_slump_joint_f0_s0/result.json").write_text("[]")
    report = summary.summarize(cv_records)
    assert report["counts"]["failed"] == 1
    json.dumps(report, allow_nan=False)


def test_summary_rejects_damaged_outputs_and_mismatched_fold_hashes(cv_records):
    for fold in range(3):
        for seed in range(5):
            path = cv_records / f"concrete_slump_coverage_f{fold}_s{seed}/result.json"
            record = json.loads(path.read_text())
            record["eval"]["nrmse"] = [1.04, .8, .8]
            path.write_text(json.dumps(record))
    report = summary.summarize(cv_records)
    gate = report["gates"]["coverage"]
    assert gate["score_passed"] and gate["complexity_passed"]
    assert not gate["worst_output_passed"] and not gate["eligible"]
    assert report["selected_method"] == "prune"
    path = cv_records / "air_quality_independent_f2_s4/result.json"
    record = json.loads(path.read_text())
    record["fold_sha256"] = "f" * 64
    path.write_text(json.dumps(record))
    report = summary.summarize(cv_records)
    assert report["protocol_errors"][0]["kind"] == "split_mismatch"
    assert report["selected_method"] is None


def test_gate_accepts_roundoff_at_exact_declared_threshold():
    control = dict(complete=True, S=1., K=100., datasets={
        name: dict(per_output_nrmse=[1.] * count) for name, count in summary.DATASETS.items()})
    candidate = dict(complete=True, S=.9900000000000001, K=110., datasets={
        name: dict(per_output_nrmse=[1.02] + [.95] * (count - 1)) for name, count in summary.DATASETS.items()})
    assert summary._gate("coverage", control, candidate, [])["eligible"]
