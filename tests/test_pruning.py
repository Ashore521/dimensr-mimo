import numpy as np
import pytest

from dimensr_mimo.features import Feature, graph_size
from dimensr_mimo.pruning import prune_candidate
from dimensr_mimo._vendor.expression import Node
from dimensr_mimo._vendor.units import ZERO, unit


def candidate(weights, intercepts, units=None):
    weights = np.array(weights, dtype=float)
    features = tuple(Feature(Node(f"x{i}", units=u))
                     for i, u in enumerate(units or [ZERO] * weights.shape[1]))
    return dict(features=features, coefficients=weights,
                intercepts=np.array(intercepts, dtype=float), train_loss=0.,
                complexity=graph_size(features) + np.count_nonzero(weights) + np.count_nonzero(intercepts))


def orthogonal_inputs():
    return np.array([[-1., -1., 1.], [-1., 1., -1.],
                     [1., -1., -1.], [1., 1., 1.]])


def test_weak_connection_is_deleted_and_remaining_head_is_refitted():
    X = orthogonal_inputs()[:, :2]
    X[:, 1] = X[:, 0] + .1 * X[:, 1]
    original = candidate([[2., .01]], [3.])
    Y = X @ original["coefficients"].T + original["intercepts"]
    counts = {}
    result = prune_candidate(original, X, Y, np.ones(1), [ZERO], [ZERO],
                             tolerance=1e-5, counts=counts)
    assert result["features"] == original["features"][:1]
    np.testing.assert_allclose(result["coefficients"], [[2.01]])
    np.testing.assert_allclose(result["intercepts"], [3.])
    assert result["train_loss"] == pytest.approx(1e-6)
    assert counts["prune_head_fits"] == 2


def test_important_connections_are_retained():
    X = orthogonal_inputs()[:, :2]
    original = candidate([[2., 3.]], [0.])
    Y = X @ original["coefficients"].T
    result = prune_candidate(original, X, Y, np.ones(1), [ZERO], [ZERO])
    assert result["features"] == original["features"]
    np.testing.assert_array_equal(result["coefficients"], original["coefficients"])
    assert result["train_loss"] == 0.


def test_tolerance_is_anchored_to_original_loss_not_accumulated():
    X = orthogonal_inputs()
    original = candidate([[3., .05, .05]], [0.])
    Y = X @ original["coefficients"].T
    result = prune_candidate(original, X, Y, np.ones(1), [ZERO], [ZERO], tolerance=.003)
    # Either weak term costs .0025 MSE; deleting both would cost .005.
    assert result["features"] == (original["features"][0], original["features"][2])
    np.testing.assert_allclose(result["coefficients"], [[3., .05]])
    assert result["train_loss"] == pytest.approx(.0025)


def test_refits_preserve_units_and_other_output_connections():
    length = unit([1, 0, 0, 0, 0, 0, 0])
    time = unit([0, 0, 1, 0, 0, 0, 0])
    X = orthogonal_inputs()
    original = candidate([[2., 0., .01], [0., 3., 0.]], [0., 0.], [length, time, length])
    Y = X @ original["coefficients"].T
    result = prune_candidate(original, X, Y, np.ones(2), [length, time], [ZERO], tolerance=.001)
    assert result["features"] == original["features"][:2]
    np.testing.assert_allclose(result["coefficients"], [[2., 0.], [0., 3.]])
    np.testing.assert_array_equal(result["intercepts"], [0., 0.])


def test_candidate_is_unchanged_and_globally_unused_features_are_removed():
    X = orthogonal_inputs()
    original = candidate([[2., 0., 0.]], [0.])
    original_features = original["features"]
    original_weights = original["coefficients"].copy()
    Y = X @ original_weights.T
    result = prune_candidate(original, X, Y, np.ones(1), [ZERO], [ZERO])
    assert result["features"] == original_features[:1]
    assert result["complexity"] == 2
    assert original["features"] == original_features
    np.testing.assert_array_equal(original["coefficients"], original_weights)
    np.testing.assert_array_equal(original["intercepts"], [0.])
    assert original["complexity"] == 4


def test_budget_interrupt_leaves_candidate_unchanged_after_local_deletion():
    X = orthogonal_inputs()
    original = candidate([[3., .05, .05]], [0.])
    Y = X @ original["coefficients"].T
    calls, counts = [], {}

    def check_budget():
        calls.append(None)
        if len(calls) == 2:
            raise RuntimeError("budget exhausted")

    with pytest.raises(RuntimeError, match="budget exhausted"):
        prune_candidate(original, X, Y, np.ones(1), [ZERO], [ZERO],
                        tolerance=.003, check_budget=check_budget, counts=counts)
    assert counts["prune_head_fits"] == 1
    assert len(original["features"]) == 3
    np.testing.assert_array_equal(original["coefficients"], [[3., .05, .05]])
    np.testing.assert_array_equal(original["intercepts"], [0.])
