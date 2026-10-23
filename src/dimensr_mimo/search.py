"""Joint residual-guided evolution of a typed, shared feature dictionary."""
from itertools import product
import numpy as np
from scipy.optimize import least_squares
from .features import Feature, compose, evaluate_features, graph_size
from ._vendor.expression import ARITY, Node
from ._vendor.units import ZERO, sub
from ._vendor.intervals import audit_box


def fit_heads(features, values, Y, target_units, coefficient_units):
    weights = np.zeros((Y.shape[1], len(features)))
    intercepts = np.zeros(Y.shape[1])
    for j, target in enumerate(target_units):
        indices = [k for k, f in enumerate(features) if sub(target, f.tree.units) in coefficient_units]
        offset = target in coefficient_units
        design = values[:, indices]
        if offset:
            design = np.column_stack((design, np.ones(len(Y))))
        if design.shape[1]:
            norm = np.maximum(np.max(np.abs(design), axis=0), np.finfo(float).tiny)
            coef = np.linalg.lstsq(design / norm, Y[:, j], rcond=None)[0] / norm
            weights[j, indices] = coef[:len(indices)]
            if offset:
                intercepts[j] = coef[-1]
    return weights, intercepts


def make_candidate(features, values, Y, scales, targets, coefficient_units):
    weights, intercepts = fit_heads(features, values, Y, targets, coefficient_units)
    prediction = values @ weights.T + intercepts
    loss = float(np.mean(((prediction - Y) / scales) ** 2))
    complexity = graph_size(features) + int(np.count_nonzero(weights)) + int(np.count_nonzero(intercepts))
    return dict(features=tuple(features), coefficients=weights, intercepts=intercepts,
                train_loss=loss, complexity=complexity)


def predict_candidate(candidate, X):
    with np.errstate(all="ignore"):
        return evaluate_features(candidate["features"], X) @ candidate["coefficients"].T + candidate["intercepts"]


def pareto(candidates):
    front = []
    for candidate in sorted(candidates, key=lambda c: (c["complexity"], c["train_loss"])):
        if not front or candidate["train_loss"] < front[-1]["train_loss"] - 1e-14:
            front.append(candidate)
    return front


def search(X, Y, bounds, feature_units, target_units, coefficient_units, cfg):
    rng = np.random.default_rng(cfg.random_state)
    std = np.std(Y, axis=0)
    # Constant outputs use a finite magnitude scale, without suppressing small varying outputs.
    scales = np.where(std > 0, std, np.maximum(np.abs(Y).max(axis=0), 1.))
    if not np.isfinite(scales).all():
        raise ValueError("Target magnitudes exceed stable numerical scaling.")
    counts = dict(feature_attempts=0, feature_evaluations=0, domain_rejections=0,
                  head_fits=0, polish_residual_calls=0)
    cache, seen = {}, set()
    base = [Feature(Node(f"x{i}", units=u)) for i, u in enumerate(feature_units)]
    base += [Feature(Node("c0"), (v,)) for v in (1., 2.)]
    if cfg.max_features < len(base):
        raise ValueError("max_features must allow all input features and two constant terminals.")

    def admit(feature):
        counts["feature_attempts"] += 1
        if feature in seen:
            return False
        seen.add(feature)
        if feature.tree.size > cfg.max_nodes or feature.tree.depth > cfg.max_depth:
            return False
        counts["feature_evaluations"] += 1
        if audit_box(feature.tree, feature.constants, bounds, max_boxes=cfg.interval_max_boxes)["status"] != "enclosed":
            counts["domain_rejections"] += 1
            return False
        values = feature.tree.evaluate(X, feature.constants)
        if not np.isfinite(values).all():
            counts["domain_rejections"] += 1
            return False
        cache[feature] = values
        return True

    for feature in base:
        admit(feature)
    operators = tuple(cfg.operators)
    initial = list(base)
    # Interleave unary and binary operators so small budgets do not starve either family.
    combinations = {op: iter(product(base, repeat=ARITY[op])) for op in operators}
    for _ in range(2 * cfg.max_features):
        available = False
        for op in operators:
            args = next(combinations[op], None)
            if args is None:
                continue
            available = True
            try:
                feature = compose(op, *args)
            except ValueError:
                continue
            if admit(feature):
                initial.append(feature)
        if not available or counts["feature_attempts"] >= 2 * cfg.max_features:
            break

    # Bounded generic scaled-unary and common-denominator productions.
    enriched = []
    for x in base[:-2]:
        if "mul" in operators:
            scaled = compose("mul", base[-1], x)
            for op in operators:
                if ARITY[op] == 1:
                    try:
                        enriched.append(compose(op, scaled))
                    except ValueError:
                        pass
        if {"add", "square", "div"} <= set(operators):
            try:
                denominator = compose("add", base[-2], compose("square", x))
                enriched.append(denominator)
                enriched.extend(compose("div", numerator, denominator) for numerator in base[:-2])
            except ValueError:
                pass
    for feature in enriched[:cfg.max_features]:
        if admit(feature):
            initial.append(feature)

    def relevance(feature, residual):
        v = cache[feature]
        score = 0.
        for j, target in enumerate(target_units):
            if sub(target, feature.tree.units) not in coefficient_units:
                continue
            centered = v - v.mean() if target in coefficient_units else v
            magnitude = np.max(np.abs(centered))
            centered = centered / magnitude if magnitude > 0 else centered
            norm = np.linalg.norm(centered)
            if norm > 0 and np.isfinite(norm):
                projection = np.dot(centered / norm, residual[:, j] / scales[j])
                score += projection ** 2 / len(Y)
        return score

    def trim(pool, protected=()):
        unique = list(dict.fromkeys(pool))
        residual = Y - np.where([t in coefficient_units for t in target_units], Y.mean(axis=0), 0.)
        ranked = sorted(unique, key=lambda f: (-relevance(f, residual), f.tree.size))
        # Keep intermediates and a random slice, including units with no direct output connection.
        small = sorted(unique, key=lambda f: f.tree.size)[:max(1, cfg.max_features // 8)]
        random = [unique[i] for i in rng.permutation(len(unique))[:max(1, cfg.max_features // 8)]]
        return list(dict.fromkeys(list(base) + list(protected) + small + random + ranked))[:cfg.max_features]

    archive, history = [], []
    pool = trim(initial)
    selected = ()
    for generation in range(cfg.generations + 1):
        if generation:
            additions = []
            for attempt in range(cfg.proposals_per_generation):
                parent = pool[int(rng.integers(len(pool)))]
                if parent.constants and attempt % 5 == 0:
                    c = np.array(parent.constants)
                    c[int(rng.integers(len(c)))] += rng.normal(0, .5)
                    proposal = Feature(parent.tree, tuple(c))
                else:
                    op = operators[int(rng.integers(len(operators)))]
                    donor_pool = list(selected) + base if attempt % 2 == 0 else pool
                    args = (parent,) if ARITY[op] == 1 else (parent, donor_pool[int(rng.integers(len(donor_pool)))])
                    try:
                        proposal = compose(op, *args)
                    except ValueError:
                        continue
                if admit(proposal):
                    additions.append(proposal)
            pool = trim(pool + additions, selected)

        current = make_candidate((), np.empty((len(X), 0)), Y, scales, target_units, coefficient_units)
        counts["head_fits"] += 1
        archive.append(current)
        beam = [current]
        for _ in range(cfg.max_terms):
            branches, seen_bases = [], set()
            for parent in beam:
                residual = Y - predict_candidate(parent, X)
                remaining = [f for f in pool if f not in parent["features"]]
                shortlist = sorted(remaining, key=lambda f: (-relevance(f, residual), f.tree.size))[:12]
                for feature in shortlist:
                    features = parent["features"] + (feature,)
                    key = frozenset(features)
                    if key in seen_bases:
                        continue
                    seen_bases.add(key)
                    values = np.column_stack([cache[f] for f in features])
                    proposal = make_candidate(features, values, Y, scales, target_units, coefficient_units)
                    counts["head_fits"] += 1
                    if proposal["train_loss"] < parent["train_loss"] - 1e-14:
                        branches.append(proposal)
            if not branches:
                break
            archive.extend(branches)
            beam = sorted(branches, key=lambda c: (round(c["train_loss"], 12), c["complexity"], c["train_loss"]))[:4]
            current = beam[0]
        selected = current["features"]

        if cfg.polish_max_nfev and any(f.constants for f in selected):
            sizes = [len(f.constants) for f in selected]
            start = np.concatenate([f.constants for f in selected])

            def unpack(values):
                parts = np.split(values, np.cumsum(sizes)[:-1])
                return tuple(Feature(f.tree, tuple(c)) for f, c in zip(selected, parts))

            def residual(values):
                counts["polish_residual_calls"] += 1
                features = unpack(values)
                matrix = evaluate_features(features, X)
                if not np.isfinite(matrix).all():
                    return np.full(Y.size, 1e10)
                try:
                    weights, offsets = fit_heads(features, matrix, Y, target_units, coefficient_units)
                    error = ((matrix @ weights.T + offsets - Y) / scales).ravel()
                    return error if np.isfinite(error).all() else np.full(Y.size, 1e10)
                except np.linalg.LinAlgError:
                    return np.full(Y.size, 1e10)

            result = least_squares(residual, start, max_nfev=cfg.polish_max_nfev)
            polished = unpack(result.x)
            if all(audit_box(f.tree, f.constants, bounds, max_boxes=cfg.interval_max_boxes)["status"] == "enclosed" for f in polished):
                candidate = make_candidate(polished, evaluate_features(polished, X), Y, scales, target_units, coefficient_units)
                counts["head_fits"] += 1
                if candidate["train_loss"] < current["train_loss"]:
                    archive.append(candidate)
                    for f in polished:
                        admit(f)
                        cache[f] = f.tree.evaluate(X, f.constants)
                    selected = polished
                    current = candidate
                    pool = trim(pool + list(polished), selected)
        history.append(dict(generation=generation, train_loss=current["train_loss"],
                            complexity=current["complexity"], retained_features=len(pool), **counts))
        # Values from discarded candidates need not survive the next generation.
        cache = {f: cache[f] for f in pool}
    return pareto(archive), scales, history, counts
