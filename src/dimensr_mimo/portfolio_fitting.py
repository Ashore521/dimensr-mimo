"""Explicit, journal-bound fit profiles for the experimental portfolio search."""
from contextlib import contextmanager
from dataclasses import replace

import numpy as np

from .capacity import parameter_partition


def affected_parameters(system, outputs):
    """Shared parameters are active once; all affected predictions remain scored."""
    used = set()
    for output in outputs:
        for index in system.reachable(system.roots[output]):
            parameter = system.nodes[index].parameter
            if parameter is not None:
                used.add(parameter)
    return tuple(sorted(used))


def _request(system, cfg, level, outputs):
    if cfg.search_schedule != 'portfolio':
        raise ValueError('Fit profiles require the portfolio schedule.')
    current = getattr(cfg, '_portfolio_request', {})
    level = current.get('level', 'screen') if level is None else level
    outputs = current.get('outputs', ()) if outputs is None else outputs
    if level not in cfg.portfolio_settings['profiles']:
        raise ValueError('Unknown portfolio fit profile.')
    if not isinstance(outputs, (list, tuple)) or any(
            type(j) is not int or not 0 <= j < len(system.roots) for j in outputs):
        raise ValueError('Affected outputs must be valid integer output indices.')
    if len(set(outputs)) != len(outputs):
        raise ValueError('Affected outputs must be unique.')
    return level, tuple(sorted(outputs))


def resolve_fit_config(system, cfg, level=None, outputs=None):
    level, outputs = _request(system, cfg, level, outputs)
    settings = cfg.portfolio_settings
    profile = settings['profiles'][level]
    linear, _ = parameter_partition(system)
    active = affected_parameters(system, outputs)
    mode = settings['solver']
    if mode == 'auto':
        local = (level == 'screen' and 0 < len(outputs) < len(system.roots) and
                 0 < len(active) <= settings['local_parameter_limit'] and len(active) < system.n_parameters)
        mode = 'local' if local else 'separable' if linear else 'joint'
    result = replace(cfg, constant_fit_mode=mode,
                     constant_max_nfev=profile['max_nfev'],
                     constant_prediction_limit=profile['prediction_limit'],
                     constant_jacobian=cfg.constant_jacobian if mode == 'joint' else 'dense')
    result._portfolio_request = dict(level=level, outputs=list(outputs))
    if mode == 'local':
        result._portfolio_active_parameters = active
    return result


def request_identity(system, cfg):
    resolved = resolve_fit_config(system, cfg)
    return dict(**resolved._portfolio_request, mode=resolved.constant_fit_mode,
                max_nfev=resolved.constant_max_nfev,
                prediction_limit=resolved.constant_prediction_limit,
                active_parameters=list(getattr(resolved, '_portfolio_active_parameters', ())))


@contextmanager
def fit_profile(cfg, level, outputs):
    sentinel = object()
    previous = getattr(cfg, '_portfolio_request', sentinel)
    cfg._portfolio_request = dict(level=level, outputs=list(outputs))
    try:
        yield
    finally:
        if previous is sentinel:
            del cfg._portfolio_request
        else:
            cfg._portfolio_request = previous


def fit_request(evaluator, system, warm, *, level='screen', outputs=(), refine=False):
    level, outputs = _request(system, evaluator.cfg, level, outputs)
    warm = np.asarray(warm, dtype=float)
    if warm.shape != (system.n_parameters,) or not np.isfinite(warm).all():
        raise ValueError('Fit start must match the graph and contain finite constants.')
    if type(refine) is not bool or (refine and system not in evaluator.cache):
        raise ValueError('Refinement requires a previously evaluated graph.')
    with fit_profile(evaluator.cfg, level, outputs):
        return evaluator.evaluate_many([(system, warm)], refine=refine)[0]


def outcome_state(outcome):
    """Evidence labels do not assert global infeasibility of a failed structure."""
    if outcome['status'] == 'worker_failure':
        return 'worker_failure'
    if outcome['candidate'] is not None:
        return 'accepted'
    stats = outcome['stats']
    if any(stats.get(key, 0) for key in ('optimizer_status_0', 'constant_prediction_limit_hits',
                                        'linear_solver_limit_hits', 'deadline_interruptions')):
        return 'limited_no_candidate'
    return 'no_accepted_candidate'


def promotion_evidence(work, operation):
    """Classify one durable native operation, independent of cached champions."""
    result = dict(operation=operation, eligible=False, reason='missing_native_outcome',
                  limit_evidence=[], rejection_evidence=[])
    if work is None:
        return result
    if work['kind'] != 'fit':
        result['reason'] = work['kind']
        return result
    outcome = work.get('outcome')
    if outcome is None:
        raise RuntimeError('Unresolved native reservation cannot be promoted.')
    if outcome['status'] == 'worker_failure':
        raise RuntimeError('Fatal native outcome cannot be promoted.')
    if outcome['candidate'] is not None:
        result['reason'] = 'native_accepted'
        return result
    if outcome['status'] != 'fit_failed':
        result['reason'] = 'unknown_native_status'
        return result
    stats = outcome['stats']
    result['limit_evidence'] = [key for key in (
        'optimizer_status_0', 'constant_prediction_limit_hits',
        'linear_solver_limit_hits', 'deadline_interruptions') if stats.get(key, 0)]
    result['rejection_evidence'] = [key for key, value in stats.items() if value and (
        key in ('point_rejections', 'structure_bound_rejections', 'resource_screening_rejections',
                'nonfinite_candidate_rejections', 'optimizer_exceptions', 'fatal_evaluation_errors') or
        (key.startswith('interval_') and key not in (
            'interval_enclosed', 'interval_boxes', 'interval_audit_calls', 'interval_audit_cpu_seconds')))]
    if result['rejection_evidence']:
        result['reason'] = 'mixed_limit_and_rejection' if result['limit_evidence'] else 'observed_rejection'
    elif result['limit_evidence']:
        result.update(eligible=True, reason='solver_allowance_exhausted')
    else:
        result['reason'] = 'no_limit_evidence'
    return result
