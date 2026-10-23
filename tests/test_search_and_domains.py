import json
import numpy as np
import pytest
from dimensr_mimo import MIMOSymbolicRegressor, SearchConfig
from dimensr_mimo.features import Feature, compose, evaluate_features, graph_size
from dimensr_mimo.search import fit_heads
from dimensr_mimo._vendor.expression import Node
from dimensr_mimo._vendor.intervals import audit_box
from dimensr_mimo._vendor.units import ZERO, unit


def test_shared_nonlinear_system_and_output_scaling():
    rng = np.random.default_rng(42)
    X = rng.uniform(-2, 2, (120, 3))
    Y = np.column_stack((2*X[:, 0]*X[:, 1]+.5*X[:, 2], 1e6*(-X[:, 0]*X[:, 1]+3*X[:, 2])))
    cfg = SearchConfig(generations=1, max_terms=3, polish_max_nfev=0)
    model = MIMOSymbolicRegressor(cfg).fit(X[:80], Y[:80], X_val=X[80:100], Y_val=Y[80:100], input_bounds=[[-2, 2]]*3)
    np.testing.assert_allclose(model.predict(X[100:]) / [1, 1e6], Y[100:] / [1, 1e6], atol=1e-10)
    assert len(model.features_) == 2
    assert model.diagnostics_["shared_features"] == 2
    assert model.audit_domain()["status"] == "enclosed"
    assert len(model.history_) == 2
    assert all(h["retained_features"] <= cfg.max_features for h in model.history_)


def test_shared_graph_matches_raw_evaluation_with_distinct_constants():
    X = np.array([[.2], [.9]])
    x, c = Feature(Node("x0")), Feature(Node("c0"), (1.,))
    a = compose("cos", compose("add", x, c))
    b = compose("mul", a, a)
    d = Feature(a.tree, (2.,))
    values = evaluate_features([a, b, d], X)
    for k, f in enumerate((a, b, d)):
        np.testing.assert_allclose(values[:, k], f.tree.evaluate(X, f.constants))
    assert graph_size([a, b]) < a.tree.size + b.tree.size
    assert not np.allclose(values[:, 0], values[:, 2])


@pytest.mark.parametrize("case", ["cancelled_pole", "off_grid", "log_zero", "hidden_overflow", "negative_root"])
def test_illegal_raw_domains_are_never_enclosed(case):
    x, c = Feature(Node("x0")), Feature(Node("c0"), (.37123456789,))
    if case == "cancelled_pole":
        f, bounds = compose("div", compose("square", x), x), [[-1, 1]]
    elif case == "off_grid":
        f, bounds = compose("div", c, compose("sub", x, c)), [[0, 1]]
    elif case == "log_zero":
        f, bounds = compose("log", x), [[0, 1]]
    elif case == "hidden_overflow":
        f, bounds = compose("div", x, compose("exp", x)), [[800, 900]]
    else:
        f, bounds = compose("sqrt", x), [[-1, 1]]
    assert audit_box(f.tree, f.constants, bounds)["status"] != "enclosed"
    if case == "hidden_overflow":
        assert np.isnan(evaluate_features([f], np.array([[850.]]))).all()


def test_valid_zero_root_and_unresolved_budget():
    x, c = Feature(Node("x0")), Feature(Node("c0"), (1.,))
    f = compose("sqrt", compose("square", x))
    assert audit_box(f.tree, (), [[-1, 1]])["status"] == "enclosed"
    f = compose("div", c, compose("add", compose("sub", compose("square", x), x), c))
    assert audit_box(f.tree, f.constants, [[0, 1]], max_boxes=1)["status"] == "unresolved"
    assert audit_box(f.tree, f.constants, [[0, 1]], max_boxes=32)["status"] == "enclosed"


def test_coefficients_require_explicit_dimensional_permission():
    length = unit([1, 0, 0, 0, 0, 0, 0])
    X = np.linspace(0, 1, 20)[:, None]
    f = Feature(Node("x0"))
    Y = 3*X
    w, _ = fit_heads([f], X, Y, [length], [ZERO])
    assert w[0, 0] == 0
    w, b = fit_heads([f], X, Y, [length], [ZERO, length])
    np.testing.assert_allclose(X @ w.T+b, Y, atol=1e-12)


def test_load_rejects_illegal_raw_tree_even_if_sympy_cancels(tmp_path):
    X = np.linspace(.1, 1, 20)[:, None]
    model = MIMOSymbolicRegressor(SearchConfig(generations=0, polish_max_nfev=0)).fit(X, X)
    path = tmp_path / "model.json"
    model.save(path)
    data = json.loads(path.read_text())
    x = Feature(Node("x0"))
    f = compose("div", compose("square", x), x)
    data["model"]["features"] = [f.to_dict()]
    data["model"]["coefficients"] = [[1.]]
    data["input_bounds"] = [[0., 1.]]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="domain"):
        MIMOSymbolicRegressor.load(path)
