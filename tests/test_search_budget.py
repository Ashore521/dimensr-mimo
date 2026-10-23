import importlib

import numpy as np
import pytest

from dimensr_mimo import MIMOSymbolicRegressor, SearchConfig
from dimensr_mimo._vendor.units import ZERO

search_module = importlib.import_module("dimensr_mimo.search")


@pytest.mark.parametrize("field,value", [
    ("search_seconds", 0), ("search_seconds", True), ("search_seconds", np.inf),
    ("candidate_quota", -1), ("candidate_quota", 1.5), ("candidate_quota", True),
    ("prune_tolerance", -1), ("prune_tolerance", np.nan), ("prune_tolerance", True),
])
def test_budget_config_rejects_invalid_values(field, value):
    with pytest.raises(ValueError):
        SearchConfig(**{field: value}).validate()


def test_expiry_during_initialization_retains_legal_fallback(monkeypatch):
    ticks = iter([0., 1., 1.1])
    monkeypatch.setattr(search_module.time, "process_time", lambda: next(ticks))
    X = np.arange(12.).reshape(-1, 1)
    Y = np.column_stack((2 * X[:, 0], 3 * X[:, 0]))
    front, _, _, counts = search_module.search(
        X, Y, np.array([[0., 11.]]), [ZERO], [ZERO, ZERO], [ZERO],
        SearchConfig(search_seconds=.1))
    assert counts["budget_exhausted"]
    assert len(front) == 1 and front[0]["features"] == ()
    np.testing.assert_allclose(front[0]["intercepts"], Y.mean(axis=0))
    assert counts["budget_overshoot_seconds"] == pytest.approx(1.)


def test_short_budget_returns_finite_serializable_model(tmp_path):
    rng = np.random.default_rng(6)
    X = rng.uniform(-1, 1, (60, 2))
    Y = np.column_stack((X[:, 0] ** 2, 2 * X[:, 1]))
    model = MIMOSymbolicRegressor(SearchConfig(search_seconds=.03, generations=10000,
        candidate_quota=1, prune_tolerance=.005)).fit(X, Y, X_val=X, Y_val=Y)
    assert model.diagnostics_["budget_exhausted"]
    assert np.isfinite(model.predict(X)).all()
    assert model.audit_domain()["status"] == "enclosed"
    path = tmp_path / "budget.json"
    model.save(path)
    loaded = MIMOSymbolicRegressor.load(path)
    np.testing.assert_array_equal(loaded.predict(X), model.predict(X))


def test_disabled_options_are_deterministic():
    X = np.linspace(-1, 1, 30).reshape(-1, 1)
    Y = np.column_stack((X[:, 0], X[:, 0] ** 2))
    cfg = SearchConfig(generations=0, max_features=24, max_terms=2, polish_max_nfev=0)
    models = [MIMOSymbolicRegressor(cfg).fit(X, Y, X_val=X, Y_val=Y) for _ in range(2)]
    assert models[0].training_archive_ == models[1].training_archive_
    assert not models[0].diagnostics_["budget_exhausted"]
