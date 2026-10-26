"""Local vector-valued tangent family; adapted from DimenSR complexity.py."""
import numpy as np
from ._single.complexity import local_rademacher as scalar_tangent_ball


def local_rademacher(system, constants, X, scales, signs, radius):
    n = len(X) * len(system.roots)
    # Each unique parameter is perturbed once and affects every bound occurrence.
    # Flatten sample-major: normalization is per output, population size is n*q.
    evaluate = lambda c: (system.evaluate(X, c) / scales).ravel()
    value = scalar_tangent_ball(system, constants, np.empty((n, 0)), 1.0, signs,
                                radius=radius, evaluate=evaluate)
    scaled = value * np.sqrt(n) / radius if radius else 0.0
    return value, scaled


def mixed_complexity(system, rad, weights):
    return float(weights[0] * system.size + weights[1] * system.n_operator_types + weights[2] * rad)
