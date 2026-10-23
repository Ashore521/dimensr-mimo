import numpy as np
import pytest
from dimensr_mimo import MIMOSymbolicRegressor, SearchConfig
from dimensr_mimo.features import Feature
from dimensr_mimo.search import fit_heads
from dimensr_mimo._vendor.expression import Node
from dimensr_mimo._vendor.units import ZERO


def test_ridge_stabilizes_collinear_noisy_heads_without_penalizing_intercept():
    rng = np.random.default_rng(81)
    x = rng.normal(size=40)
    X = np.column_stack((x, x + .001*rng.normal(size=40)))
    Y = np.column_stack((2*x + .2*rng.normal(size=40) + 7, np.full(40, 5.)))
    features = [Feature(Node("x0")), Feature(Node("x1"))]
    ols, offset = fit_heads(features, X, Y, [ZERO]*2, [ZERO])
    ridge, bias = fit_heads(features, X, Y, [ZERO]*2, [ZERO], .001)
    test_x = np.linspace(-2, 2, 100)
    test = np.column_stack((test_x, test_x + .1*np.sin(4*test_x)))
    truth = 2*test_x + 7
    assert np.mean((test @ ridge[0]+bias[0]-truth)**2) < np.mean((test @ ols[0]+offset[0]-truth)**2)
    assert np.linalg.norm(ridge[0]) < np.linalg.norm(ols[0])
    np.testing.assert_allclose(X @ ridge[1]+bias[1], 5., atol=1e-12)


@pytest.mark.parametrize("value", [-1., float("nan"), float("inf"), True, "0.1"])
def test_invalid_head_ridge(value):
    with pytest.raises(ValueError, match="head_ridge"):
        SearchConfig(head_ridge=value).validate()


def test_regularized_model_roundtrip_and_validation_isolation(tmp_path):
    rng = np.random.default_rng(3)
    X = rng.normal(size=(35, 2))
    Y = np.column_stack((2*X[:, 0]+.1*rng.normal(size=35), -X[:, 1]))
    cfg = SearchConfig(generations=0, max_features=32, max_terms=2, polish_max_nfev=0, head_ridge=.001)
    a = MIMOSymbolicRegressor(cfg).fit(X[:25], Y[:25], X_val=X[25:], Y_val=Y[25:])
    b = MIMOSymbolicRegressor(cfg).fit(X[:25], Y[:25], X_val=X[25:], Y_val=Y[25:]+100)
    assert a.training_archive_ == b.training_archive_
    path = tmp_path / "model.json"
    a.save(path)
    restored = MIMOSymbolicRegressor.load(path)
    assert restored.config_.head_ridge == .001
    np.testing.assert_array_equal(restored.predict(X), a.predict(X))
