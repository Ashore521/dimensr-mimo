"""Historical families, declared coefficient/unit variants and two regressions."""
from copy import deepcopy
import numpy as np
from dimensr_mimo._single.units import ZERO, unit, scale, add, encode
from experiments.noise_families_20261026 import cases as old
from experiments.typed_coverage_20261026 import cases as regression

VARIANTS = ('damped_variant', 'rational_variant', 'polynomial_variant')
REGRESSIONS = ('products', 'boundary')
CASES = old.CASES + VARIANTS + REGRESSIONS


def seeds(name):
    return old.SEEDS if name in old.CASES else (711,) if name in VARIANTS else (601,)


def design(name):
    if name in old.CASES:
        return old.design(name)
    if name in REGRESSIONS:
        spec = regression.design(name)
        return dict(spec, clean_test_nrmse=[1e-4]*2, family=name, noise_fraction=0.)
    if name not in VARIANTS:
        raise ValueError('Outside the frozen matrix.')
    family = name.split('_')[0]
    spec = deepcopy(old.design(family+'_n1'))
    first, second = (old.M, old.T) if family == 'polynomial' else (old.L, old.M)
    targets = (add(first, second), add(first, scale(second, 2))) if family == 'polynomial' else (second, second if family == 'damped' else scale(second, 2))
    spec.update(features=[encode(first), encode(second)], targets=[encode(u) for u in targets],
        constants=[encode(ZERO), encode(scale(second, -2) if family == 'polynomial' else scale(first, -1))],
        bounds=[[.15, 1.4], [.3, 1.2]], feature_names=['quantity0', 'quantity1'])
    spec['formulas'] = {
        'damped': ['x1*exp(-0.65*x0)*cos(1.6*x0)', 'x1*exp(-0.65*x0)*sin(1.6*x0)'],
        'rational': ['x1/(1+(0.45*x0)**2)', 'x1**2/(1+(0.45*x0)**2)'],
        'polynomial': ['x0*x1*(1+0.55*x1**2)', '0.8*x0*x1**2*(1+0.55*x1**2)']}[family]
    return spec


def latent(name, X):
    if name in old.CASES:
        return old.latent(name, X)
    a, b = X.T
    if name == 'products':
        return np.column_stack((a*b, .5*a*b*b))
    if name == 'boundary':
        return np.column_stack((np.log(a+b), np.sqrt(a+b)))
    if name == 'damped_variant':
        envelope = b*np.exp(-.65*a)
        return np.column_stack((envelope*np.cos(1.6*a), envelope*np.sin(1.6*a)))
    if name == 'rational_variant':
        return np.column_stack((b/(1+(.45*a)**2), b*b/(1+(.45*a)**2)))
    if name == 'polynomial_variant':
        return np.column_stack((a*b*(1+.55*b*b), .8*a*b*b*(1+.55*b*b)))
    raise ValueError('Unknown case.')


def inputs(name, split, seed):
    if seed not in seeds(name):
        raise ValueError('Outside the frozen seeds.')
    if name in old.CASES:
        return old.inputs(name, split, seed)
    if name in REGRESSIONS:
        return regression.generate(name, split, seed)[0]
    spec = design(name)
    rng = np.random.default_rng(np.random.SeedSequence([20261027, seed, {'train':111, 'validation':222, 'test':333}[split], VARIANTS.index(name)]))
    bounds = np.asarray(spec['bounds'])
    return rng.uniform(bounds[:, 0], bounds[:, 1], ({'train':96, 'validation':48, 'test':192}[split], 2))


def generate(name, split, seed):
    if seed not in seeds(name):
        raise ValueError('Outside the frozen seeds.')
    if name in old.CASES:
        return old.generate(name, split, seed)
    if name in REGRESSIONS:
        return regression.generate(name, split, seed)
    X = inputs(name, split, seed)
    scales = latent(name, inputs(name, 'train', seed)).std(axis=0)
    rng = np.random.default_rng(np.random.SeedSequence([20261027, seed, {'train':411, 'validation':422, 'test':433}[split], VARIANTS.index(name)]))
    Y = latent(name, X)
    return X, Y + .01*scales*rng.normal(size=Y.shape)


def training_data(name, split, seed):
    if split not in ('train', 'validation'):
        raise ValueError('Fitting cannot request test data.')
    return generate(name, split, seed)
