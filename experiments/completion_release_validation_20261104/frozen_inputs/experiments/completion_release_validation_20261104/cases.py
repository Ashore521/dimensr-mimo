"""Locked-candidate synthetic output pairs authored for the 1.7.0 release."""
from pathlib import Path

import numpy as np

from dimensr_mimo._single.units import ZERO, encode, scale, unit

HERE = Path(__file__).resolve().parent
L = unit([1, 0, 0, 0, 0, 0, 0])
T = unit([0, 0, 1, 0, 0, 0, 0])
FAMILIES = ('log_affine_ratio', 'reciprocal_quadratic_norm',
            'oscillatory_affine_saturation', 'exponential_affine_saturation')
CASES = tuple(f'{family}_n{noise}' for family in FAMILIES for noise in (0, 1))
SEEDS = (10241, 10242)


def seeds(name):
    if name not in CASES:
        raise ValueError('Outside the frozen release-validation matrix.')
    return SEEDS


def design(name):
    seeds(name)
    family, noise = name.rsplit('_n', 1)
    if family == 'log_affine_ratio':
        features, targets, constants = (ZERO, ZERO), (ZERO, ZERO), (ZERO,)
        formulas = ['log((1+0.4*x0)/(1+0.7*x1))',
                    'x0*log((1+0.4*x0)/(1+0.7*x1))']
    elif family == 'reciprocal_quadratic_norm':
        features, targets, constants = (ZERO, L), (ZERO, L), (ZERO,)
        formulas = ['1/sqrt(1+x0**2)', 'x1/sqrt(1+x0**2)']
    elif family == 'oscillatory_affine_saturation':
        features, targets, constants = (T, ZERO), (ZERO, ZERO), (ZERO, scale(T, -1))
        formulas = ['sin(0.6*x0)/(1+0.5*x1)', 'cos(0.6*x0)/(1+0.5*x1)']
    else:
        features, targets, constants = (T, L), (ZERO, L), (ZERO, scale(T, -1))
        formulas = ['exp(-0.4*x0)/(1+0.6*x0)', 'x1*exp(-0.4*x0)/(1+0.6*x0)']
    return dict(features=[encode(u) for u in features], targets=[encode(u) for u in targets],
                constants=[encode(u) for u in constants], feature_names=['quantity0', 'quantity1'],
                target_names=['response0', 'response1'], formulas=formulas,
                bounds=[[.2, 1.4], [.2, 1.4]], family=family, noise_fraction=.01*int(noise),
                input_transform='identity in declared SI units',
                output_transform='identity; normalization only in error metric',
                acceptance=dict(nrmse=[.03 if int(noise) else 1e-4]*2,
                                shared_graph_size=24, independent_parameters=6,
                                expanded_total_nodes=30, maximum_output_depth=6),
                clean_test_nrmse=[.02 if int(noise) else 1e-4]*2)


def latent(name, X):
    family = design(name)['family']
    a, b = np.asarray(X).T
    if family == 'log_affine_ratio':
        response = np.log((1+.4*a)/(1+.7*b))
        return np.column_stack((response, a*response))
    if family == 'reciprocal_quadratic_norm':
        denominator = np.sqrt(1+a*a)
        return np.column_stack((1/denominator, b/denominator))
    if family == 'oscillatory_affine_saturation':
        denominator = 1+.5*b
        return np.column_stack((np.sin(.6*a)/denominator, np.cos(.6*a)/denominator))
    response = np.exp(-.4*a)/(1+.6*a)
    return np.column_stack((response, b*response))


def inputs(name, split, seed):
    if seed not in seeds(name):
        raise ValueError('Outside the frozen release-validation seeds.')
    code = {'train': 111, 'validation': 222, 'test': 333}[split]
    count = {'train': 96, 'validation': 48, 'test': 192}[split]
    rng = np.random.default_rng(np.random.SeedSequence([20261104, seed, code, CASES.index(name)]))
    bounds = np.asarray(design(name)['bounds'])
    return rng.uniform(bounds[:, 0], bounds[:, 1], (count, 2))


def generate(name, split, seed):
    if split == 'test' and not (HERE/'ALL_MODELS_FROZEN.json').exists():
        raise RuntimeError('Test labels require the current all-model seal.')
    X = inputs(name, split, seed)
    Y = latent(name, X)
    scales = latent(name, inputs(name, 'train', seed)).std(axis=0)
    code = {'train': 411, 'validation': 422, 'test': 433}[split]
    rng = np.random.default_rng(np.random.SeedSequence([20261104, seed, code, CASES.index(name)]))
    return X, Y+design(name)['noise_fraction']*scales*rng.normal(size=Y.shape)


def training_data(name, split, seed):
    if split not in ('train', 'validation'):
        raise ValueError('Training cannot request test data.')
    return generate(name, split, seed)


def truth_graph(name):
    """Static unit, grammar and domain audit; never consumed by search."""
    from dimensr_mimo._single.expression import Node
    from dimensr_mimo.graph import System, Vertex
    from dimensr_mimo.structure import operator

    spec = design(name)
    nodes, values = [], []

    def x(index):
        nodes.append(Vertex('x'+str(index), units=unit(spec['features'][index])))
        return len(nodes)-1

    def c(value, units=ZERO):
        token = next(i for i, encoded in enumerate(spec['constants']) if unit(encoded) == units)
        parameter = len(values)
        values.append(value)
        nodes.append(Vertex('c'+str(token), units=units, parameter=parameter))
        return len(nodes)-1

    def op(token, *children):
        units = operator(token, *[Node('x0', units=nodes[i].units) for i in children]).units
        nodes.append(Vertex(token, children, units))
        return len(nodes)-1

    a, b = x(0), x(1)
    one = c(1.)
    if spec['family'] == 'log_affine_ratio':
        numerator = op('add', one, op('mul', c(.4), a))
        denominator = op('add', one, op('mul', c(.7), b))
        response = op('log', op('div', numerator, denominator))
        roots = (response, op('mul', a, response))
    elif spec['family'] == 'reciprocal_quadratic_norm':
        denominator = op('sqrt', op('add', one, op('square', a)))
        roots = (op('div', one, denominator), op('div', b, denominator))
    elif spec['family'] == 'oscillatory_affine_saturation':
        argument = op('mul', c(.6, scale(T, -1)), a)
        denominator = op('add', one, op('mul', c(.5), b))
        roots = (op('div', op('sin', argument), denominator),
                 op('div', op('cos', argument), denominator))
    else:
        numerator = op('exp', op('mul', c(-.4, scale(T, -1)), a))
        denominator = op('add', one, op('mul', c(.6, scale(T, -1)), a))
        response = op('div', numerator, denominator)
        roots = (response, op('mul', b, response))
    return System(tuple(nodes), roots), np.asarray(values)
