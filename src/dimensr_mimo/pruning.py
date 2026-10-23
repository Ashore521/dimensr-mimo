"""Conservative training-only pruning of individual output connections."""
import numpy as np
from .features import evaluate_features, graph_size


def prune_candidate(candidate, X, Y, scales, targets, coefficient_units,
                    ridge=0.0, tolerance=0.005, check_budget=lambda: None,
                    counts=None):
    """Refit after weak-connection deletion without exceeding an output's loss budget.

    Each output's normalized training MSE is anchored to the original candidate,
    so successive deletions cannot accumulate additional tolerance. Try the
    original connection ranking once; this greedy training criterion can miss
    jointly redundant connections and does not establish validation improvement.
    The input model is unchanged, including when ``check_budget`` interrupts.
    """
    from .search import fit_heads

    features = candidate["features"]
    values = evaluate_features(features, X)
    weights = candidate["coefficients"].copy()
    intercepts = candidate["intercepts"].copy()
    original_loss = np.mean(((values @ weights.T + intercepts - Y) / scales) ** 2, axis=0)
    feature_std = np.std(values, axis=0)
    connections = sorted(
        zip(*np.nonzero(weights)),
        key=lambda connection: abs(weights[connection]) * feature_std[connection[1]] / scales[connection[0]],
    )
    for output, feature in connections:
        if weights[output, feature] == 0:
            continue
        active = [i for i in np.flatnonzero(weights[output]) if i != feature]
        check_budget()
        if counts is not None:
            counts["prune_head_fits"] = counts.get("prune_head_fits", 0) + 1
        fitted, offset = fit_heads(
            [features[i] for i in active], values[:, active], Y[:, output:output + 1],
            [targets[output]], coefficient_units, ridge,
        )
        prediction = values[:, active] @ fitted[0] + offset[0]
        loss = np.mean(((prediction - Y[:, output]) / scales[output]) ** 2)
        if (loss <= original_loss[output] + tolerance
                and np.count_nonzero(fitted) < np.count_nonzero(weights[output])):
            weights[output] = 0.
            weights[output, active] = fitted[0]
            intercepts[output] = offset[0]

    used = np.flatnonzero(np.any(weights != 0, axis=0))
    features = tuple(features[i] for i in used)
    weights = weights[:, used]
    prediction = values[:, used] @ weights.T + intercepts
    loss = float(np.mean(((prediction - Y) / scales) ** 2))
    complexity = graph_size(features) + int(np.count_nonzero(weights)) + int(np.count_nonzero(intercepts))
    return dict(candidate, features=features, coefficients=weights, intercepts=intercepts,
                train_loss=loss, complexity=complexity)
