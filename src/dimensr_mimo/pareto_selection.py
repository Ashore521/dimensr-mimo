"""Deterministic accuracy/complexity selection from a frozen candidate pool."""
from fractions import Fraction
import math

from .complexity import mixed_complexity
from .graph import System
from .selection import digest


RELATIVE_FACTOR = 1.05


def _finite_nonnegative(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError('Metrics must be finite and nonnegative.')
    return value


def _bound(reference):
    try:
        return float(reference * Fraction.from_float(RELATIVE_FACTOR))
    except OverflowError:
        return None


def _within(value, reference):
    # Exact ratios avoid overflow and rounding a tiny nonzero bound up to a candidate.
    return value == 0 if reference == 0 else value / reference <= RELATIVE_FACTOR


def _candidate(candidate, index, weights):
    system = System.from_dict(candidate['system'])
    constants = [float(value) for value in candidate['constants']]
    if len(constants) != system.n_parameters or not all(map(math.isfinite, constants)):
        raise ValueError('Constants must match the graph and be finite.')
    training = [_finite_nonnegative(value) for value in candidate['errors']]
    validation = [_finite_nonnegative(value) for value in candidate['validation_errors']]
    if len(training) != len(system.roots) or len(validation) != len(system.roots):
        raise ValueError('Error vectors must match the output roots.')
    raw = _finite_nonnegative(candidate['rademacher'])
    scaled = _finite_nonnegative(candidate['rademacher_complexity'])
    stored = _finite_nonnegative(candidate['complexity'])
    total = _finite_nonnegative(mixed_complexity(system, scaled, weights))
    identity = dict(system=system.to_dict(), constants=constants, errors=training,
                    validation_errors=validation, complexity=total,
                    rademacher=raw, rademacher_complexity=scaled)
    exact_errors = tuple(map(Fraction.from_float, validation))
    # Summing exact frozen floats also preserves order below the float subnormal range.
    mean = sum(exact_errors, Fraction()) / len(exact_errors)
    metrics = dict(candidate_index=index, candidate_id=digest(identity),
                   mean_validation_nrmse=float(mean), validation_nrmse=validation,
                   training_nrmse=training,
                   mean_validation_nrmse_underflow=bool(mean and float(mean) == 0),
                   complexity=dict(size=system.size, operator_types=system.n_operator_types,
                       rademacher_raw=raw, rademacher_scaled=scaled,
                       weighted_size=weights[0] * system.size,
                       weighted_operator_types=weights[1] * system.n_operator_types,
                       weighted_rademacher=weights[2] * scaled,
                       total=total, stored_total=stored))
    return dict(index=index, mean=mean, errors=exact_errors, complexity=total,
                candidate_id=metrics['candidate_id'], metrics=metrics)


def select_pareto(candidates, complexity_weights):
    """Return a JSON-compatible report without evaluating or changing candidates.

    Candidate dictionaries use ``Candidate.to_dict()`` plus ``validation_errors``.
    The configured mixed complexity is recomputed from their frozen graph and
    scaled Rademacher metric. Stored score/violation fields have no effect.
    Malformed/nonfinite candidates are reported and excluded. Valid candidates
    must agree on their output count. Empty or wholly invalid pools raise
    ValueError. The float multiplier 1.05 is applied without additive tolerance,
    with exact-zero protection. Bounds that exceed float range are reported as
    None and flagged, while exact-ratio comparisons remain finite and correct.
    """
    weights = tuple(_finite_nonnegative(value) for value in complexity_weights)
    if len(weights) != 3:
        raise ValueError('complexity_weights must contain exactly three weights.')
    candidates = list(candidates)
    if not candidates:
        raise ValueError('Cannot select from an empty candidate pool.')
    valid, invalid = [], []
    for index, candidate in enumerate(candidates):
        try:
            valid.append(_candidate(candidate, index, weights))
        except (KeyError, TypeError, ValueError, OverflowError, IndexError,
                AttributeError, RecursionError) as error:
            invalid.append(dict(candidate_index=index, reason=str(error)))
    if not valid:
        raise ValueError('Candidate pool contains no valid finite candidates.')
    if len({len(candidate['errors']) for candidate in valid}) != 1:
        raise ValueError('Valid candidates must have the same output count.')
    best = min(valid, key=lambda c: (c['mean'], c['complexity'], c['candidate_id']))
    checks, eligible = [], []
    for candidate in valid:
        mean_ok = _within(candidate['mean'], best['mean'])
        output_ok = [_within(value, reference)
                     for value, reference in zip(candidate['errors'], best['errors'])]
        qualifies = mean_ok and all(output_ok)
        checks.append(dict(candidate_index=candidate['index'], mean_within_bound=mean_ok,
                           per_output_within_bounds=output_ok,
                           all_outputs_within_bounds=all(output_ok), eligible=qualifies))
        if qualifies:
            eligible.append(candidate)
    recommended = min(eligible, key=lambda c: (c['complexity'], c['mean'], c['candidate_id']))
    # A sorted sweep keeps equal-objective candidates and runs in O(n log n).
    front, minimum_complexity, previous_mean = [], math.inf, None
    previous_group_minimum = math.inf
    for candidate in sorted(valid, key=lambda c: (c['mean'], c['complexity'], c['candidate_id'])):
        if candidate['mean'] != previous_mean:
            previous_group_minimum = minimum_complexity
            previous_mean = candidate['mean']
        if (candidate['complexity'] < previous_group_minimum and
                candidate['complexity'] <= minimum_complexity):
            front.append(candidate)
        minimum_complexity = min(minimum_complexity, candidate['complexity'])
    mean_bound = _bound(best['mean'])
    output_bounds = [_bound(value) for value in best['errors']]
    selected = {best['index'], recommended['index'], *(candidate['index'] for candidate in front)}
    return dict(method='accuracy_pareto', version=1,
                relative_factor=RELATIVE_FACTOR, complexity_weights=list(weights),
                pool_size=len(candidates), valid_indices=[c['index'] for c in valid],
                invalid_candidates=invalid, best_accuracy_index=best['index'],
                best_accuracy_id=best['candidate_id'], recommended_index=recommended['index'],
                recommended_id=recommended['candidate_id'],
                recommended_is_best_accuracy=recommended['index'] == best['index'],
                pareto_indices=[c['index'] for c in front],
                pareto_ids=[c['candidate_id'] for c in front],
                mean_validation_bound=mean_bound, per_output_validation_bounds=output_bounds,
                mean_validation_bound_overflow=mean_bound is None,
                per_output_validation_bound_overflow=[value is None for value in output_bounds],
                candidate_checks=checks, eligible_indices=[c['index'] for c in eligible],
                candidate_metrics=[c['metrics'] for c in valid if c['index'] in selected])
