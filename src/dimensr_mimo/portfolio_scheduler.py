"""Deterministic training-progress/cost scheduling with a minimum exploration share."""
from copy import deepcopy
import math


PRODUCERS = ('grammar', 'gp', 'roots', 'features')
DEFAULTS = dict(
    profiles=dict(screen=dict(max_nfev=8, prediction_limit=128),
                  standard=dict(max_nfev=32, prediction_limit=512),
                  refine=dict(max_nfev=128, prediction_limit=2048)),
    solver='auto', local_parameter_limit=32, proposal_limit=4096,
    empty_patience=8, root_capacity=8, feature_limit=128,
    minimum_share=0.1, refine_every=8, policy_batch_size=4,
    coefficient_completion=False, completion_scheduler='priority', failure_promotion='legacy',
    structural_discovery=False,
    producer_weights=dict.fromkeys(PRODUCERS, 1.0))


def settings(value=None):
    """Return independent, fully specified and strictly bounded controls."""
    if value is not None and (not isinstance(value, dict) or set(value) - set(DEFAULTS)):
        raise ValueError('Invalid portfolio settings keys.')
    result = deepcopy(DEFAULTS)
    for key, item in (value or {}).items():
        if key in ('profiles', 'producer_weights'):
            if not isinstance(item, dict):
                raise ValueError('Invalid portfolio ' + key)
            if key == 'profiles':
                if set(item) != set(result[key]):
                    raise ValueError('Portfolio profiles require screen, standard and refine.')
                for name, profile in item.items():
                    if not isinstance(profile, dict) or set(profile) != {'max_nfev', 'prediction_limit'}:
                        raise ValueError('Each portfolio profile requires max_nfev and prediction_limit.')
                    if any(type(v) is not int or not 1 <= v <= 1_000_000 for v in profile.values()):
                        raise ValueError('Portfolio profile limits must be bounded positive integers.')
            elif (set(item) != set(PRODUCERS) or any(isinstance(v, bool) or
                    not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 < v <= 1000
                    for v in item.values())):
                raise ValueError('Each portfolio producer needs a bounded positive weight.')
        result[key] = deepcopy(item)
    for key, limit in dict(local_parameter_limit=100_000, proposal_limit=10_000_000,
                           empty_patience=10_000, root_capacity=1024,
                           feature_limit=4096, refine_every=1_000_000,
                           policy_batch_size=128).items():
        if type(result[key]) is not int or not 1 <= result[key] <= limit:
            raise ValueError('Invalid bounded portfolio ' + key)
    if result['solver'] not in ('auto', 'joint', 'separable'):
        raise ValueError('Unknown portfolio solver.')
    if type(result['coefficient_completion']) is not bool:
        raise ValueError('coefficient_completion must be boolean.')
    if type(result['structural_discovery']) is not bool:
        raise ValueError('structural_discovery must be boolean.')
    if result['completion_scheduler'] not in ('priority', 'balanced'):
        raise ValueError('Unknown coefficient completion scheduler.')
    if result['completion_scheduler'] == 'balanced' and not result['coefficient_completion']:
        raise ValueError('Balanced completion scheduling requires coefficient_completion.')
    if result['failure_promotion'] not in ('legacy', 'evidence'):
        raise ValueError('Unknown failure promotion strategy.')
    share = result['minimum_share']
    if (isinstance(share, bool) or not isinstance(share, (int, float)) or
            not math.isfinite(share) or not 0 < share <= 0.25):
        raise ValueError('Portfolio minimum_share must be in (0, 0.25].')
    if result['completion_scheduler'] == 'balanced' and share > 0.2:
        raise ValueError('Five producers require minimum_share <= 0.2.')
    return result


def producer_names(controls):
    return PRODUCERS + (('coefficients',) if controls['completion_scheduler'] == 'balanced' else ())


def initial_state(controls=None):
    return {name: dict(selections=0, empty=0, paused_version=None, improvement=0., cost=0.)
            for name in (PRODUCERS if controls is None else producer_names(controls))}


def choose(state, controls, version):
    """Paused empty producers reactivate when accepted training context changes."""
    names = producer_names(controls)
    active = [name for name in names if state[name]['paused_version'] != version]
    if not active:
        return None
    total = sum(row['selections'] for row in state.values())
    underserved = [name for name in active
                   if state[name]['selections'] < controls['minimum_share'] * (total + 1)]
    if underserved:
        return min(underserved, key=lambda name: (state[name]['selections'], names.index(name)))
    def priority(name):
        row = state[name]
        credit = row['improvement'] / max(1., row['cost'])
        exploration = math.sqrt(math.log(total + 2) / (row['selections'] + 1))
        return controls['producer_weights'].get(name, 1.) * (credit + exploration)
    return max(active, key=priority)


def feedback(state, name, controls, version, *, empty, improvement=0., cost=0.):
    row = state[name]
    row['selections'] += 1
    row['improvement'] += max(0., improvement)
    row['cost'] += max(0., cost)
    row['empty'] = row['empty'] + 1 if empty else 0
    if row['empty'] >= controls['empty_patience']:
        row['paused_version'] = version
