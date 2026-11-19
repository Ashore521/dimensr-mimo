"""Constrained two-objective NSGA-II using DimenSR sorting/crowding/size density."""
import numpy as np
from ._single.pareto import selection_order, fronts, crowding, size_density
from .evaluation import search_score


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
        order = selection_order([(search_score(candidates[i].errors, cfg), candidates[i].complexity) for i in group],
                                [candidates[i].system.size for i in group],
                                cfg.use_size_diversity, cfg.size_radius)
        if cfg.search_objective == 'worst':
            champion = min(range(len(group)), key=lambda k: (search_score(candidates[group[k]].errors, cfg),
                candidates[group[k]].error, candidates[group[k]].complexity))
            order = [champion] + [k for k in order if k != champion]
        result.extend(candidates[group[k]] for k in order)
    return result


def retain_parameters(ranked, limit, cfg, reference=None):
    """Protect compact parameter-count representatives without dropping the best score."""
    if not cfg.parameter_diversity or limit <= 0 or not ranked:
        return ranked[:limit]
    best_violation = min(violations(c.errors, reference, cfg) for c in ranked)
    eligible = [c for c in ranked if violations(c.errors, reference, cfg) == best_violation]
    key = lambda c: (search_score(c.errors, cfg), c.complexity)
    protected = [min(eligible, key=key)]
    cap = cfg.acceptance['independent_parameters']
    slots = min(4, max(1, limit // 4))
    for index in np.linspace(0, 3, slots, dtype=int):
        bound = int(np.ceil(cap * (index + 1) / 4))
        compact = [c for c in eligible if c.system.n_parameters <= bound]
        if compact:
            protected.append(min(compact, key=key))
    return list({c.system: c for c in protected + ranked}.values())[:limit]


def diverse(candidates, limit, cfg, reference=None):
    if limit <= 0:
        return []
    key = lambda c: (violations(c.errors, reference, cfg), search_score(c.errors, cfg), c.error, c.complexity)
    unique = {}
    for c in sorted(candidates, key=key):
        unique.setdefault(c.system, c)
    ranked, kept = list(unique.values()), []
    # Diversity is subordinate to constraint-domination, including small graphs.
    for violation in sorted({violations(c.errors, reference, cfg) for c in ranked}):
        group = [c for c in ranked if violations(c.errors, reference, cfg) == violation]
        slots = limit - len(kept)
        if slots <= 0:
            break
        best = {}
        for c in group:
            best.setdefault(c.system.size, c)
        sizes = [best[k] for k in sorted(best)]
        count = min(len(sizes), max(1, int(slots * cfg.elite_fraction))) if cfg.use_size_diversity else 0
        protect = [sizes[i] for i in np.linspace(0, len(sizes) - 1, count, dtype=int)] if count else []
        priority = ([group[0]] if cfg.search_objective == 'worst' else []) + protect + group
        kept.extend(retain_parameters(list({c.system: c for c in priority}.values()), slots, cfg, reference))
    return kept
