"""Declared synthetic development cases. No external or private data arrays."""
import numpy as np

CASES = ('shared', 'private', 'partial', 'shared_parameter', 'scales', 'units', 'boundary')
ZERO = [0] * 7


def design(name):
    if name not in CASES:
        raise ValueError(name)
    return dict(bounds=[[.2, 1.2], [.3, 1.3]] if name == 'boundary' else [[-.9, .9], [-.9, .9]],
                features=[ZERO, ZERO], targets=[ZERO, ZERO], constants=[ZERO]) if name != 'units' else dict(
                    bounds=[[.2, 2.], [.3, 1.5]], features=[[0, 1, 0, 0, 0, 0, 0], [1, 0, -1, 0, 0, 0, 0]],
                    targets=[[1, 1, -1, 0, 0, 0, 0], [2, 1, -2, 0, 0, 0, 0]], constants=[ZERO])


def generate(name, split, seed=0):
    """Independent RNG streams; workers request train/validation only."""
    spec = design(name)
    code = {'train': 111, 'validation': 222, 'test': 333}[split]
    rng = np.random.default_rng(np.random.SeedSequence([20261025, seed, code, CASES.index(name)]))
    n = {'train': 96, 'validation': 48, 'test': 192}[split]
    bounds = np.asarray(spec['bounds'])
    X = rng.uniform(bounds[:, 0], bounds[:, 1], (n, 2))
    a, b = X.T
    if name == 'shared':
        z = a * b
        Y = np.column_stack((z + a, z - b))
    elif name == 'private':
        Y = np.column_stack((a ** 2 + a, np.sin(b)))
    elif name == 'partial':
        z = a ** 2
        Y = np.column_stack((z + b, z + np.sin(b)))
    elif name == 'shared_parameter':
        Y = np.column_stack((np.sin(1.7 * a), np.cos(1.7 * b)))
    elif name == 'scales':
        Y = np.column_stack((.01 * (a + b), 100 * (a - b)))
    elif name == 'units':
        Y = np.column_stack((a * b, .5 * a * b ** 2))
    else:
        Y = np.column_stack((np.log(a + b), np.sqrt(a + b)))
    return X, Y
