"""Validation-only post-selection of one immutable candidate pool.

The guard is relative to the original weighted selector's chosen outputs.
It guarantees the declared validation bounds only, not test-set bounds.
"""
from copy import deepcopy
import hashlib
import json
import math

FIELDS = ('system', 'constants', 'errors', 'complexity', 'rademacher', 'rademacher_complexity')
POLICY = dict(degradation_relative=0.10, degradation_absolute=1e-8,
              equivalent_relative=0.01, equivalent_absolute=1e-8)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def choose(candidates, method, policy=POLICY):
    if method not in ('original', 'no_penalty', 'guarded') or not candidates:
        raise ValueError('Unknown method or empty candidate pool.')
    if set(policy) != set(POLICY) or any(not math.isfinite(v) or v < 0 for v in policy.values()):
        raise ValueError('Selection tolerances must be finite and nonnegative.')
    q = len(candidates[0]['validation_errors'])
    for c in candidates:
        values = c['validation_errors'] + [c['violation'], c['score'], c['complexity']]
        if not q or len(c['validation_errors']) != q or not all(math.isfinite(v) and v >= 0 for v in values):
            raise ValueError('Invalid candidate metrics.')
    errors = [sum(c['validation_errors']) / q for c in candidates]
    anchor = min(range(len(candidates)), key=lambda i: (candidates[i]['violation'], candidates[i]['score'], candidates[i]['complexity']))
    tier = [i for i, c in enumerate(candidates) if c['violation'] == candidates[anchor]['violation']]
    bounds = [v * (1 + policy['degradation_relative']) + policy['degradation_absolute']
              for v in candidates[anchor]['validation_errors']]
    protected = [i for i in tier if all(v <= bound for v, bound in zip(candidates[i]['validation_errors'], bounds))]
    assert anchor in protected
    best_error = min(errors[i] for i in protected)
    near_bound = best_error * (1 + policy['equivalent_relative']) + policy['equivalent_absolute']
    near = [i for i in protected if errors[i] <= near_bound]
    index = (anchor if method == 'original' else
             min(tier, key=lambda i: (errors[i], candidates[i]['complexity'])) if method == 'no_penalty' else
             min(near, key=lambda i: (candidates[i]['complexity'], errors[i], i)))
    return dict(method=method, candidate_index=index, anchor_index=anchor, policy=dict(policy),
                validation_bounds=bounds, minimum_violation_tier=len(tier), protected_count=len(protected),
                near_optimal_count=len(near), near_optimal_mean_bound=near_bound,
                chosen_validation_errors=candidates[index]['validation_errors'],
                chosen_within_bounds=all(v <= bound for v, bound in zip(candidates[index]['validation_errors'], bounds)),
                candidate_sha256=digest({k: candidates[index][k] for k in FIELDS}))


def derive(source, method):
    """Export a prediction-compatible model with explicit post-selection metadata."""
    from dimensr_mimo.graph import System
    record = choose(source['candidates'], method)
    pool = [{k: c[k] for k in FIELDS} for c in source['candidates']]
    if digest(pool) != source['diagnostics']['frozen_candidate_sha256']:
        raise ValueError('Candidate pool differs from its training freeze.')
    anchor = source['candidates'][record['anchor_index']]
    if any(source['best'][k] != anchor[k] for k in FIELDS):
        raise ValueError('Source model is not the original weighted choice.')
    model = deepcopy(source)
    candidate = model['candidates'][record['candidate_index']]
    model['best'] = {k: candidate[k] for k in FIELDS}
    model['validation_nrmse'] = candidate['validation_errors']
    model['diagnostics'].update(validation_feasible=candidate['violation'] == 0,
                                validation_violation=candidate['violation'], selected_score=candidate['score'],
                                sharing=System.from_dict(candidate['system']).sharing)
    # Config and candidate scores describe the original search/anchor. The new
    # top-level record, retained in the exported JSON, defines post-selection.
    model['post_selection'] = dict(record, source_model_sha256=digest(source),
                                   search_reused=True, training_configuration_unchanged=True)
    return model
