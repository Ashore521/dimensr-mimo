"""Public estimator and versioned, non-executable JSON inference models."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted
from .config import SearchConfig
from .features import Feature, compose, evaluate_features
from .search import search, predict_candidate
from ._vendor.expression import Node
from ._vendor.intervals import audit_box
from ._vendor.units import ZERO, unit, encode, sub


def _names(values, count, prefix):
    result = list(values) if values is not None else [f"{prefix}{i}" for i in range(count)]
    if len(result) != count or any(not isinstance(n, str) or not n for n in result) or len(set(result)) != count:
        raise ValueError(f"{prefix} names must be distinct nonempty strings matching the column count.")
    return result


def _units(values, count):
    result = tuple(unit(u) for u in values) if values is not None else (ZERO,) * count
    if len(result) != count:
        raise ValueError("Unit metadata must match the column count.")
    return result


def _bounds(bounds, n_features):
    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape != (n_features, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 0] > bounds[:, 1]):
        raise ValueError("input_bounds must contain one finite ordered pair per feature.")
    return bounds


def _data(X, Y):
    X = check_array(X, dtype=float)
    Y = check_array(Y, dtype=float)
    if len(X) != len(Y):
        raise ValueError("X and Y must have the same sample count.")
    return X, Y


def _serial_candidate(candidate):
    return dict(features=[f.to_dict() for f in candidate["features"]],
                coefficients=candidate["coefficients"].tolist(), intercepts=candidate["intercepts"].tolist(),
                train_loss=candidate["train_loss"], complexity=candidate["complexity"])


def _output_features(candidate, targets):
    result = []
    for j, target in enumerate(targets):
        terms = []
        for k, feature in enumerate(candidate["features"]):
            coefficient = candidate["coefficients"][j, k]
            if coefficient != 0:
                scalar = Feature(Node("c0", units=sub(target, feature.tree.units)), (float(coefficient),))
                terms.append(compose("mul", scalar, feature))
        intercept = candidate["intercepts"][j]
        if intercept != 0 or not terms:
            terms.append(Feature(Node("c0", units=target), (float(intercept),)))
        tree = terms[0]
        for term in terms[1:]:
            tree = compose("add", tree, term)
        result.append(tree)
    return result


class MIMOSymbolicRegressor(RegressorMixin, BaseEstimator):
    """Joint symbolic dictionary search for X=(samples, inputs), Y=(samples, outputs).

    Units are seven SI exponents in L,M,T,I,Theta,N,J order. Coefficients are
    dimensionless by default; explicitly supply coefficient_units to admit
    dimensional output coefficients. Internal feature constants are dimensionless.
    """

    def __init__(self, config=None, feature_names=None, target_names=None,
                 feature_units=None, target_units=None, coefficient_units=None):
        self.config = config
        self.feature_names = feature_names
        self.target_names = target_names
        self.feature_units = feature_units
        self.target_units = target_units
        self.coefficient_units = coefficient_units

    def _more_tags(self):
        return {"multioutput": True, "multioutput_only": True}

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.multi_output = True
        tags.target_tags.single_output = False
        return tags

    def fit(self, X, Y, *, X_val=None, Y_val=None, input_bounds=None):
        for name in list(vars(self)):
            if name.endswith("_"):
                delattr(self, name)
        cfg = replace(self.config) if self.config is not None else SearchConfig()
        cfg.validate()
        X, Y = _data(X, Y)
        if len(X) < 5:
            raise ValueError("At least five samples are required.")
        n_features, n_outputs = X.shape[1], Y.shape[1]
        names = _names(self.feature_names, n_features, "x")
        targets = _names(self.target_names, n_outputs, "y")
        feature_units = _units(self.feature_units, n_features)
        target_units = _units(self.target_units, n_outputs)
        coefficient_units = tuple(unit(u) for u in self.coefficient_units) if self.coefficient_units is not None else (ZERO,)
        if not coefficient_units:
            raise ValueError("coefficient_units must not be empty.")
        if (X_val is None) != (Y_val is None):
            raise ValueError("Provide both X_val and Y_val.")
        if X_val is None:
            rng = np.random.default_rng(np.random.SeedSequence([cfg.random_state, 911]))
            order = rng.permutation(len(X))
            count = max(1, int(np.ceil(len(X) * cfg.validation_fraction)))
            if len(X) - count < 2:
                raise ValueError("validation_fraction leaves fewer than two training samples.")
            validation_indices, training_indices = order[:count], order[count:]
            X_val, Y_val = X[validation_indices], Y[validation_indices]
            X, Y = X[training_indices], Y[training_indices]
        else:
            X_val, Y_val = _data(X_val, Y_val)
            if X_val.shape[1] != n_features or Y_val.shape[1] != n_outputs:
                raise ValueError("Validation feature and output counts must match training data.")
            training_indices, validation_indices = np.arange(len(X)), None
        bounds = _bounds(input_bounds, n_features) if input_bounds is not None else np.column_stack((X.min(axis=0), X.max(axis=0)))
        if np.any(X < bounds[:, 0]) or np.any(X > bounds[:, 1]):
            raise ValueError("input_bounds must contain all training inputs.")

        candidates, scales, history, counts = search(X, Y, bounds, feature_units, target_units, coefficient_units, cfg)
        # Validation labels are first consulted here, after every training parameter is frozen.
        front = []
        for candidate in candidates:
            outputs = _output_features(candidate, target_units)
            if any(audit_box(f.tree, f.constants, bounds, max_boxes=cfg.interval_max_boxes)["status"] != "enclosed" for f in outputs):
                continue
            prediction = predict_candidate(candidate, X_val)
            with np.errstate(all="ignore"):
                loss = float(np.mean(((prediction - Y_val) / scales) ** 2))
            if np.isfinite(prediction).all() and np.isfinite(loss):
                front.append((loss + cfg.complexity_weight * candidate["complexity"], loss, candidate))
        if not front:
            raise ValueError("No candidate is finite on validation data and enclosed on the training domain.")
        _, validation_loss, selected = min(front, key=lambda item: (item[0], item[2]["complexity"]))
        self.n_features_in_, self.n_outputs_ = n_features, n_outputs
        self.feature_names_, self.target_names_ = names, targets
        self.feature_units_, self.target_units_, self.coefficient_units_ = feature_units, target_units, coefficient_units
        self.input_bounds_, self.target_scales_ = bounds, scales
        self.training_indices_, self.validation_indices_ = training_indices, validation_indices
        self.training_archive_ = [_serial_candidate(c) for c in candidates]
        self.pareto_front_ = [dict(_serial_candidate(c), validation_loss=loss) for _, loss, c in front]
        self.history_, self.config_ = history, cfg
        self.diagnostics_ = dict(counts, training_rows=len(X), validation_rows=len(X_val),
                                 selected_train_loss=selected["train_loss"], selected_validation_loss=validation_loss,
                                 shared_features=int(sum(np.count_nonzero(selected["coefficients"][:, k]) > 1
                                                     for k in range(len(selected["features"])))),
                                 domain_source="declared_box" if input_bounds is not None else "training_box")
        self._set_model(selected)
        return self

    def _set_model(self, candidate):
        self.model_ = candidate
        self.features_ = candidate["features"]
        self.coef_, self.intercept_ = candidate["coefficients"], candidate["intercepts"]
        outputs = _output_features(candidate, self.target_units_)
        self.expressions_ = [f.tree.sympy(f.constants, self.feature_names_) for f in outputs]

    def predict(self, X):
        check_is_fitted(self, "model_")
        X = check_array(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Prediction feature count does not match fitted data.")
        prediction = predict_candidate(self.model_, X)
        if not np.isfinite(prediction).all():
            raise ValueError("Raw expression is undefined or overflows at supplied prediction inputs.")
        return prediction

    def audit_domain(self, input_bounds=None):
        check_is_fitted(self, "model_")
        bounds = self.input_bounds_ if input_bounds is None else _bounds(input_bounds, self.n_features_in_)
        features = [audit_box(f.tree, f.constants, bounds, max_boxes=self.config_.interval_max_boxes) for f in self.features_]
        outputs = [audit_box(f.tree, f.constants, bounds, max_boxes=self.config_.interval_max_boxes)
                   for f in _output_features(self.model_, self.target_units_)]
        statuses = [r["status"] for r in features + outputs]
        status = "invalid" if "invalid" in statuses else "unresolved" if "unresolved" in statuses else "enclosed"
        return dict(status=status, features=features, outputs=outputs, input_bounds=bounds.tolist(),
                    guarantee="Numerical interval screening, not a formal mathematical certificate.")

    def save(self, path):
        check_is_fitted(self, "model_")
        data = dict(format="dimensr-mimo", version=1, config=asdict(self.config_),
                    feature_names=self.feature_names_, target_names=self.target_names_,
                    feature_units=[encode(u) for u in self.feature_units_],
                    target_units=[encode(u) for u in self.target_units_],
                    coefficient_units=[encode(u) for u in self.coefficient_units_],
                    input_bounds=self.input_bounds_.tolist(), target_scales=self.target_scales_.tolist(),
                    model=_serial_candidate(self.model_))
        Path(path).write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        path = Path(path)
        if path.stat().st_size > 10_000_000:
            raise ValueError("Model JSON exceeds 10 MB.")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data["format"] != "dimensr-mimo" or data["version"] != 1:
                raise ValueError("Unsupported model format or version.")
            cfg = SearchConfig(**data["config"])
            cfg.validate()
            model = cls(cfg, data["feature_names"], data["target_names"], data["feature_units"],
                        data["target_units"], data["coefficient_units"])
            n, m = len(model.feature_names), len(model.target_names)
            if n < 1 or m < 1:
                raise ValueError("Empty feature or target metadata.")
            model.feature_names_, model.target_names_ = _names(model.feature_names, n, "x"), _names(model.target_names, m, "y")
            model.feature_units_, model.target_units_ = _units(model.feature_units, n), _units(model.target_units, m)
            model.coefficient_units_ = tuple(unit(u) for u in model.coefficient_units)
            if not model.coefficient_units_:
                raise ValueError("Empty coefficient units.")
            model.input_bounds_ = _bounds(data["input_bounds"], n)
            scales = np.asarray(data["target_scales"], dtype=float)
            if scales.shape != (m,) or not np.isfinite(scales).all() or np.any(scales <= 0):
                raise ValueError("Invalid target scales.")
            saved = data["model"]
            features = []

            def validate_tree(tree, depth=0):
                if depth > 64:
                    raise ValueError("Expression nesting exceeds 64 levels.")
                children = tree["children"]
                if not isinstance(children, list) or len(children) > 2:
                    raise ValueError("Invalid tree children.")
                for child in children:
                    validate_tree(child, depth + 1)

            if len(saved["features"]) > cfg.max_terms:
                raise ValueError("Too many basis features.")
            for entry in saved["features"]:
                validate_tree(entry["tree"])
                tree = Node.from_dict(entry["tree"])
                constants = np.asarray(entry["constants"], dtype=float)
                if constants.shape != (tree.n_constants,) or not np.isfinite(constants).all():
                    raise ValueError("Invalid feature constants.")
                if tree.size > cfg.max_nodes or tree.depth > cfg.max_depth:
                    raise ValueError("Tree exceeds saved search limits.")
                for _, node in tree.walk():
                    if node.token.startswith("x"):
                        index = int(node.token[1:])
                        if index >= n or node.units != model.feature_units_[index]:
                            raise ValueError("Invalid feature index or units.")
                    if not node.children and node.token.startswith("c") and node.units != ZERO:
                        raise ValueError("Internal constants must be dimensionless.")
                features.append(Feature(tree, tuple(constants)))
            weights, intercepts = np.asarray(saved["coefficients"], dtype=float), np.asarray(saved["intercepts"], dtype=float)
            if weights.shape != (m, len(features)) or intercepts.shape != (m,) or not np.isfinite(weights).all() or not np.isfinite(intercepts).all():
                raise ValueError("Invalid coefficient or intercept shape/values.")
            for j, target in enumerate(model.target_units_):
                if intercepts[j] != 0 and target not in model.coefficient_units_:
                    raise ValueError("Intercept units are not permitted.")
                for k, feature in enumerate(features):
                    if weights[j, k] != 0 and sub(target, feature.tree.units) not in model.coefficient_units_:
                        raise ValueError("Coefficient units are not permitted.")
            loss, complexity = float(saved["train_loss"]), int(saved["complexity"])
            if not np.isfinite(loss) or loss < 0 or complexity < 0:
                raise ValueError("Invalid saved training metrics.")
            model.n_features_in_, model.n_outputs_, model.config_, model.target_scales_ = n, m, cfg, scales
            model._set_model(dict(features=tuple(features), coefficients=weights, intercepts=intercepts,
                                  train_loss=loss, complexity=complexity))
            if model.audit_domain()["status"] != "enclosed":
                raise ValueError("Saved raw model fails its declared domain audit.")
            return model
        except (KeyError, TypeError, IndexError, OverflowError, RecursionError, StopIteration, ZeroDivisionError) as exc:
            raise ValueError("Malformed dimensr-mimo model JSON.") from exc
