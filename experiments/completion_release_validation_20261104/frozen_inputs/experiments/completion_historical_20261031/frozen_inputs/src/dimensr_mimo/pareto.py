"""Constrained two-objective NSGA-II using DimenSR sorting/crowding/size density."""
import numpy as np
from ._single.pareto import selection_order, fronts, crowding, size_density


def violations(errors, reference, cfg):
    if not cfg.use_constraints or reference is None:
        return 0.0
    threshold = np.asarray(reference) * (1 + cfg.degradation_relative) + cfg.degradation_absolute
    return float(np.maximum(np.asarray(errors) - threshold, 0).sum())


def ordered(candidates, cfg, reference=None):
    if not candidates:
        return []
    violation = [violations(c.errors, reference, cfg) for c in candidates]
    result = []
    # Deb constraint-domination. Violations are NOT a third objective.
    for amount in sorted(set(violation)):
        group = [i for i, v in enumerate(violation) if v == amount]
        order = selection_order([(candidates[i].error, candidates[i].complexity) for i in group],
                                [candidates[i].system.size for i in group],
                                cfg.use_size_diversity, cfg.size_radius)
        result.extend(candidates[group[k]] for k in order)
    return result


def diverse(candidates, limit, cfg):
    ranked = sorted({c.system: c for c in candidates}.values(), key=lambda c: (c.error, c.complexity))
    best = {}
    for c in ranked:
        best.setdefault(c.system.size, c)
    sizes = [best[k] for k in sorted(best)]
    count = min(len(sizes), max(1, int(limit * cfg.elite_fraction))) if cfg.use_size_diversity else 0
    protect = [sizes[i] for i in np.linspace(0, len(sizes) - 1, count, dtype=int)] if count else []
    return list({c.system: c for c in protect + ranked}.values())[:limit]
