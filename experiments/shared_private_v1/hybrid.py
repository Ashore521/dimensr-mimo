"""Experimental shared dictionary plus output-specific residual dictionaries.

Training and validation selection are separate functions. No holdout data is
accepted here. Reuses the unchanged typed search, raw-domain audit and JSON API.
"""
from dataclasses import replace
from pathlib import Path
import sys
import time

import numpy as np

from dimensr_mimo import MIMOSymbolicRegressor
from dimensr_mimo.features import evaluate_features, graph_size
from dimensr_mimo.regressor import _output_features, _serial_candidate
from dimensr_mimo.search import search, make_candidate, predict_candidate, pareto
from dimensr_mimo._vendor.intervals import audit_box

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "budget_capacity_v1"))
try:
    from curves import decode, validation_frontier
finally:
    sys.path.pop(0)


def scales_of(Y):
    std = Y.std(axis=0)
    return np.where(std > 0, std, np.maximum(np.abs(Y).max(axis=0), 1.))


def enclosed(candidate, bounds, targets, max_boxes):
    expressions = list(candidate["features"]) + _output_features(candidate, targets)
    return all(audit_box(f.tree, f.constants, bounds, max_boxes=max_boxes)["status"] == "enclosed"
               for f in expressions)


def project(candidate, output, X, Y, scales):
    weights = candidate["coefficients"][output:output+1]
    used = np.flatnonzero(weights[0] != 0)
    features = tuple(candidate["features"][k] for k in used)
    weights = weights[:, used]
    intercepts = candidate["intercepts"][output:output+1].copy()
    pred = evaluate_features(features, X) @ weights.T + intercepts
    return dict(features=features, coefficients=weights.copy(), intercepts=intercepts,
                train_loss=float(np.mean(((pred - Y[:, output:output+1])/scales[output])**2)),
                complexity=graph_size(features)+int(np.count_nonzero(weights))+int(np.count_nonzero(intercepts)),
                origin="shared")


def search_hybrid(X, Y, bounds, feature_units, targets, coefficient_units, cfg):
    """40% shared search, 40% private search, remainder for training-only refits.

Every per-output head uses at most cfg.max_terms bases. Shared bases may have
different coefficients in each head. Unused allowance is not reallocated.
"""
    cfg.validate()
    if cfg.search_seconds is None or cfg.max_terms < 2:
        raise ValueError("Hybrid requires a finite search_seconds and max_terms >= 2.")
    started = time.process_time()
    deadline = started + cfg.search_seconds
    scales = scales_of(Y)
    shared_terms = cfg.max_terms // 2
    private_terms = cfg.max_terms - shared_terms
    zero = make_candidate((), np.empty((len(X), 0)), Y, scales, targets, coefficient_units, cfg.head_ridge)
    shared = [zero]
    stages, refits = [], 0

    def run(response, units, nominal, terms, seed):
        allowance = min(nominal, deadline-time.process_time())
        if allowance <= 0:
            return []
        front, _, _, counts = search(X, response, bounds, feature_units, units, coefficient_units,
            replace(cfg, search_seconds=allowance, max_terms=terms, random_state=seed))
        stages.append(dict(allocation=allowance, **counts))
        return front

    shared += run(Y, targets, .4*cfg.search_seconds, shared_terms, cfg.random_state)
    # Anchor selection consults training loss and raw-domain legality only.
    anchor = zero
    for candidate in sorted(shared, key=lambda c: (c["train_loss"]+cfg.complexity_weight*c["complexity"], c["complexity"])):
        if time.process_time() >= deadline:
            break
        if enclosed(candidate, bounds, targets, cfg.interval_max_boxes):
            anchor = candidate
            break
    residual = Y - predict_candidate(anchor, X)
    private = [run(residual[:, j:j+1], targets[j:j+1], .4*cfg.search_seconds/Y.shape[1],
                   private_terms, cfg.random_state) for j in range(Y.shape[1])]
    pools = [[project(c, j, X, Y, scales) for c in shared] for j in range(Y.shape[1])]
    # Round-robin refitting avoids spending the remaining allowance on output 0.
    for index in range(max(map(len, private), default=0)):
        for j, front in enumerate(private):
            if index >= len(front):
                continue
            bases = front[index]["features"]
            for features, origin in ((tuple(dict.fromkeys(anchor["features"]+bases)), "shared+private"),
                                     (bases, "private-only")):
                if time.process_time() >= deadline:
                    break
                candidate = make_candidate(features, evaluate_features(features, X), Y[:, j:j+1],
                    scales[j:j+1], targets[j:j+1], coefficient_units, cfg.head_ridge)
                candidate = project(candidate, 0, X, Y[:, j:j+1], scales[j:j+1])
                candidate["origin"] = origin
                pools[j].append(candidate)
                refits += 1
    pools = [pareto(pool) for pool in pools]
    elapsed = time.process_time()-started
    return pools, dict(search_cpu_seconds=elapsed, budget_overshoot_seconds=max(0.,elapsed-cfg.search_seconds),
        stages=stages, refit_head_fits=refits, shared_anchor=_serial_candidate(anchor),
        shared_terms_limit=shared_terms, private_terms_limit=private_terms,
        head_fits=sum(s["head_fits"] for s in stages)+refits)


def validate_archives(pools, X_val, Y_val, scales, bounds, targets, max_boxes):
    """Attach inner losses after all training parameters are fixed."""
    archives = []
    for j, pool in enumerate(pools):
        valid = []
        for candidate in pool:
            if not enclosed(candidate, bounds, targets[j:j+1], max_boxes):
                continue
            pred = predict_candidate(candidate, X_val)
            with np.errstate(all="ignore"):
                loss = float(np.mean(((pred-Y_val[:, j:j+1])/scales[j])**2))
            if np.isfinite(pred).all() and np.isfinite(loss):
                valid.append(dict(_serial_candidate(candidate), validation_loss=loss,
                                  origin=candidate.get("origin", "independent")))
        if not valid:
            raise ValueError(f"No domain-safe finite inner-validation candidate for output {j}.")
        archives.append(valid)
    return archives


def select_indices(archives, rule, weight=1e-4):
    if rule == "penalty":
        return [min(range(len(a)), key=lambda i: (
            a[i]["validation_loss"]+weight*a[i]["complexity"], a[i]["complexity"], i)) for a in archives]
    if rule != "compact":
        raise ValueError("Expected penalty or compact selection.")
    eligible = []
    for candidates in archives:
        best = min(c["validation_loss"] for c in candidates)
        eligible.append([i for i,c in enumerate(candidates) if c["validation_loss"] <= 1.01*best + 1e-12])
    # One-percent tolerance is a deterministic rule, not a confidence interval.
    front, _ = validation_frontier([[a[i] for i in keep] for a,keep in zip(archives,eligible)])
    return [keep[i] for keep,i in zip(eligible,front[0]["indices"])]


def combine(archives, indices):
    heads = [decode(archives[j][i]) for j,i in enumerate(indices)]
    features = tuple(dict.fromkeys(f for c in heads for f in c["features"]))
    positions = {f:k for k,f in enumerate(features)}
    weights = np.zeros((len(heads),len(features)))
    for j,c in enumerate(heads):
        for k,f in enumerate(c["features"]):
            weights[j,positions[f]] += c["coefficients"][0,k]
    intercepts = np.array([c["intercepts"][0] for c in heads])
    return dict(features=features, coefficients=weights, intercepts=intercepts,
        train_loss=float(np.mean([c["train_loss"] for c in heads])),
        complexity=graph_size(features)+int(np.count_nonzero(weights))+int(np.count_nonzero(intercepts)))


def inference_model(candidate, cfg, bounds, scales, feature_names, target_names,
                    feature_units, target_units, coefficient_units):
    """Export through the unchanged validated v1 interchange format."""
    model = MIMOSymbolicRegressor(cfg, feature_names, target_names, feature_units, target_units, coefficient_units)
    model.config_ = replace(cfg, max_terms=cfg.max_terms*len(target_names))
    model.feature_names_, model.target_names_ = list(feature_names), list(target_names)
    model.feature_units_, model.target_units_, model.coefficient_units_ = feature_units, target_units, coefficient_units
    model.n_features_in_, model.n_outputs_ = len(feature_names), len(target_names)
    model.input_bounds_, model.target_scales_ = np.asarray(bounds), np.asarray(scales)
    model._set_model(candidate)
    if model.audit_domain()["status"] != "enclosed":
        raise ValueError("Combined raw model does not pass its training-box audit.")
    return model
