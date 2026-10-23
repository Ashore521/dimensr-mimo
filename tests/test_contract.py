"""Public API checks that do not depend on stochastic formula recovery."""
import json
from fractions import Fraction

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from dimensr_mimo import MIMOSymbolicRegressor, SearchConfig


ZERO = [0, 0, 0, 0, 0, 0, 0]
LENGTH = [1, 0, 0, 0, 0, 0, 0]
TIME = [0, 0, 1, 0, 0, 0, 0]


def small_config():
    return SearchConfig(generations=0, max_features=32, max_terms=2,
                        polish_max_nfev=0, random_state=17)


def data():
    rng = np.random.default_rng(61)
    X = rng.uniform(0.2, 1.2, (32, 2))
    Y = np.column_stack((2 * X[:, 0], -3 * X[:, 1]))
    return X, Y


def regressor(**kwargs):
    return MIMOSymbolicRegressor(config=small_config(), **kwargs)


@pytest.fixture
def fitted():
    X, Y = data()
    return regressor().fit(X, Y), X, Y


def test_clone_preserves_constructor_metadata():
    model = regressor(
        feature_names=["distance", "time"],
        target_names=["distance_out", "time_out"],
        feature_units=[LENGTH, TIME],
        target_units=[LENGTH, TIME],
        coefficient_units=[ZERO],
    )
    copied = clone(model)
    assert copied.get_params(deep=False) == model.get_params(deep=False)
    assert copied is not model
    with pytest.raises(NotFittedError):
        copied.predict([[1.0, 2.0]])


def test_sklearn_tags_advertise_matrix_targets():
    model = regressor()
    try:
        from sklearn.utils import get_tags
    except ImportError:
        from sklearn.utils._tags import _safe_tags
        tags = _safe_tags(model)
        assert tags["multioutput"] and tags["multioutput_only"]
    else:
        tags = get_tags(model).target_tags
        assert tags.multi_output and not tags.single_output


@pytest.mark.parametrize("bad", [
    "x_rank", "y_rank", "row_count", "nonfinite_x", "nonfinite_y", "zero_outputs",
])
def test_fit_rejects_invalid_array_contract(bad):
    X, Y = data()
    if bad == "x_rank":
        X = X[:, 0]
    elif bad == "y_rank":
        Y = Y[:, 0]
    elif bad == "row_count":
        Y = Y[:-1]
    elif bad == "nonfinite_x":
        X[0, 0] = np.nan
    elif bad == "nonfinite_y":
        Y[0, 1] = np.inf
    else:
        Y = np.empty((len(X), 0))
    with pytest.raises(ValueError):
        regressor().fit(X, Y)


@pytest.mark.parametrize("metadata", [
    {"feature_names": ["only_one"]},
    {"target_names": ["only_one"]},
    {"feature_names": ["duplicate", "duplicate"]},
    {"target_names": ["duplicate", "duplicate"]},
    {"feature_units": [ZERO]},
    {"target_units": [ZERO]},
    {"feature_units": [[0, 0], ZERO]},
    {"target_units": [ZERO, [0, 0]]},
])
def test_metadata_matches_all_input_and_output_columns(metadata):
    X, Y = data()
    with pytest.raises(ValueError):
        regressor(**metadata).fit(X, Y)


@pytest.mark.parametrize("validation", ["x_only", "y_only", "wrong_outputs", "wrong_inputs"])
def test_validation_arrays_must_match_training_contract(validation):
    X, Y = data()
    kwargs = {"X_val": X[:8], "Y_val": Y[:8]}
    if validation == "x_only":
        kwargs.pop("Y_val")
    elif validation == "y_only":
        kwargs.pop("X_val")
    elif validation == "wrong_outputs":
        kwargs["Y_val"] = Y[:8, :1]
    else:
        kwargs["X_val"] = X[:8, :1]
    with pytest.raises(ValueError):
        regressor().fit(X, Y, **kwargs)


def test_predict_shape_and_feature_validation(fitted):
    model, X, _ = fitted
    assert model.predict(X[:3]).shape == (3, 2)
    assert len(model.expressions_) == 2
    with pytest.raises(ValueError):
        model.predict(np.ones((3, 1)))
    with pytest.raises(ValueError):
        model.predict([[np.nan, 1.0]])


def test_different_output_units_are_supported():
    X, Y = data()
    model = regressor(feature_units=[LENGTH, TIME], target_units=[LENGTH, TIME])
    model.fit(X, Y)
    assert model.predict(X[:4]).shape == (4, 2)
    assert np.isfinite(model.predict(X)).all()
    assert len(model.expressions_) == 2
    for candidate in model.training_archive_:
        assert candidate["intercepts"] == [0.0, 0.0]
        for j, target in enumerate((LENGTH, TIME)):
            for k, feature in enumerate(candidate["features"]):
                if candidate["coefficients"][j][k] != 0:
                    assert [Fraction(v) for v in feature["tree"]["units"]] == target


def test_dimensionful_output_coefficient_requires_explicit_opt_in():
    X, _ = data()
    X = X[:, :1]
    Y = 3 * X
    conversion_units = [-1, 0, 1, 0, 0, 0, 0]
    model = regressor(feature_units=[LENGTH], target_units=[TIME],
                      coefficient_units=[conversion_units]).fit(X, Y)
    np.testing.assert_allclose(model.predict(X), Y, rtol=1e-10, atol=1e-10)
    assert all(candidate["intercepts"] == [0.0] for candidate in model.training_archive_)


def test_single_output_stays_a_matrix():
    X, Y = data()
    model = regressor().fit(X, Y[:, :1])
    assert model.predict(X[:4]).shape == (4, 1)


def test_finite_large_features_do_not_overflow_head_normalization():
    X = np.linspace(1.0, 2.0, 32)[:, None] * 1e160
    Y = np.column_stack((X[:, 0] * 3e-160, X[:, 0] * -2e-160))
    config = SearchConfig(generations=0, max_features=24, max_terms=1,
                          polish_max_nfev=0, operators=("add", "mul"))
    model = MIMOSymbolicRegressor(config=config).fit(X, Y, X_val=X, Y_val=Y)
    np.testing.assert_allclose(model.predict(X), Y, rtol=1e-10, atol=1e-10)


def test_validation_labels_do_not_change_training_search():
    X, Y = data()
    first = regressor().fit(X[:24], Y[:24], X_val=X[24:], Y_val=Y[24:])
    second = regressor().fit(X[:24], Y[:24], X_val=X[24:], Y_val=Y[24:] + [500, -300])
    assert first.training_archive_
    assert first.training_archive_ == second.training_archive_


def test_failed_refit_discards_previous_model(fitted):
    model, X, Y = fitted
    bad_y = Y.copy()
    bad_y[0, 0] = np.nan
    with pytest.raises(ValueError):
        model.fit(X, bad_y)
    with pytest.raises(NotFittedError):
        model.predict(X)


def test_json_roundtrip_preserves_predictions(fitted, tmp_path):
    model, X, _ = fitted
    path = tmp_path / "model.json"
    model.save(path)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(saved, dict)
    restored = MIMOSymbolicRegressor.load(path)
    np.testing.assert_array_equal(restored.predict(X), model.predict(X))
    assert [str(item) for item in restored.expressions_] == [str(item) for item in model.expressions_]


@pytest.mark.parametrize("text", [
    "{ not valid JSON",
    "[]",
    "null",
    '{"format": "not-this-package", "schema_version": 1}',
    '{"format": "dimensr-mimo", "schema_version": 999}',
])
def test_bad_json_is_rejected_as_invalid_input(text, tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        MIMOSymbolicRegressor.load(path)


@pytest.mark.parametrize("tamper", [
    "version", "weights", "intercepts", "constants", "target_units",
    "feature_units", "scales", "negative_loss", "unknown_operator", "feature_index",
])
def test_tampered_model_metadata_and_parameters_are_rejected(fitted, tmp_path, tamper):
    model, _, _ = fitted
    path = tmp_path / "tampered.json"
    model.save(path)
    saved = json.loads(path.read_text(encoding="utf-8"))
    if tamper == "version":
        saved["version"] = 999
    elif tamper == "weights":
        saved["model"]["coefficients"] = [[]]
    elif tamper == "intercepts":
        saved["model"]["intercepts"] = [0.0]
    elif tamper == "constants":
        saved["model"]["features"][0]["constants"].append(1.0)
    elif tamper == "target_units":
        saved["target_units"][0] = TIME
    elif tamper == "feature_units":
        saved["feature_units"][0] = LENGTH
    elif tamper == "scales":
        saved["target_scales"] = [1.0]
    elif tamper == "negative_loss":
        saved["model"]["train_loss"] = -1.0
    elif tamper == "unknown_operator":
        saved["model"]["features"][0]["tree"]["token"] = "unknown"
    else:
        pending = [entry["tree"] for entry in saved["model"]["features"]]
        while pending:
            node = pending.pop()
            if node["token"].startswith("x"):
                node["token"] = "x999"
                break
            pending.extend(node["children"])
        else:
            pytest.fail("The deterministic linear fixture must contain an input feature.")
    path.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(ValueError):
        MIMOSymbolicRegressor.load(path)


@pytest.mark.parametrize("kind", ["cancelled_hole", "off_grid_pole"])
def test_load_reaudits_raw_program_on_saved_bounds(fitted, tmp_path, kind):
    model, _, _ = fitted
    path = tmp_path / "invalid-domain.json"
    model.save(path)
    saved = json.loads(path.read_text(encoding="utf-8"))

    def node(token, *children):
        return {"token": token, "units": ZERO, "children": list(children)}

    x = node("x0")
    if kind == "cancelled_hole":
        tree, constants = node("div", node("square", x), x), []
    else:
        c = node("c0")
        tree = node("div", c, node("sub", x, c))
        constants = [1.0, 0.37123456789]
    saved["model"]["features"][0] = {"tree": tree, "constants": constants}
    saved["input_bounds"][0] = [0.0, 1.0]
    path.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(ValueError, match="domain"):
        MIMOSymbolicRegressor.load(path)
