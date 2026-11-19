"""Bounded variable projection over a structurally proven affine subset."""
import numpy as np
import time
from scipy.optimize import lsq_linear


class LinearSolveLimit(Exception):
    pass


def design(system, constants, X, linear, stats):
    """One topological pass, propagating only nonzero affine coefficients."""
    values, coefficients = [], []
    linear = set(linear)
    operators = dict(add=np.add, sub=np.subtract, mul=np.multiply, div=np.divide,
                     neg=np.negative, square=np.square, sqrt=np.sqrt, sin=np.sin,
                     cos=np.cos, exp=np.exp, log=np.log)
    with np.errstate(all='ignore'):
        for node in system.nodes:
            columns = {}
            if node.parameter is not None:
                value = np.full(len(X), 0. if node.parameter in linear else constants[node.parameter])
                if node.parameter in linear:
                    columns[node.parameter] = np.ones(len(X))
            elif node.token.startswith('x'):
                value = X[:, int(node.token[1:])]
            else:
                args = [values[c] for c in node.children]
                parts = [coefficients[c] for c in node.children]
                value = operators[node.token](*args)
                value = np.where(np.logical_and.reduce([np.isfinite(a) for a in args]) &
                                 np.isfinite(value), value, np.nan)
                if node.token in ('add', 'sub'):
                    sign = 1 if node.token == 'add' else -1
                    columns = dict(parts[0])
                    for p, column in parts[1].items():
                        columns[p] = columns.get(p, 0.) + sign * column
                elif node.token == 'neg':
                    columns = {p: -column for p, column in parts[0].items()}
                elif node.token == 'mul':
                    if parts[0] and parts[1]:
                        raise ValueError('Linear subset contains a product conflict.')
                    columns = {p: column * args[1] for p, column in parts[0].items()}
                    columns.update({p: column * args[0] for p, column in parts[1].items()})
                elif node.token == 'div':
                    if parts[1]:
                        raise ValueError('Linear subset occurs in a denominator.')
                    columns = {p: column / args[1] for p, column in parts[0].items()}
                elif any(parts):
                    raise ValueError('Linear subset occurs under a nonlinear operator.')
            stats['affine_coefficient_vectors'] += len(columns)
            values.append(value)
            coefficients.append(columns)
    base = np.column_stack([values[r] for r in system.roots])
    ids = sorted(linear)
    matrix = np.zeros((len(X), len(system.roots), len(ids)))
    index = {p: k for k, p in enumerate(ids)}
    for j, root in enumerate(system.roots):
        for p, column in coefficients[root].items():
            matrix[:, j, index[p]] = column
    return base, matrix.reshape(base.size, len(ids))


def project(system, constants, X, Y, scales, linear, bound, stats):
    started, cpu = time.monotonic(), time.process_time()
    stats['separable_linear_attempts'] += 1
    try:
        return _project(system, constants, X, Y, scales, linear, bound, stats)
    finally:
        stats['separable_linear_wall_seconds'] += time.monotonic() - started
        stats['separable_linear_cpu_seconds'] += time.process_time() - cpu


def _project(system, constants, X, Y, scales, linear, bound, stats):
    base, matrix = design(system, constants, X, linear, stats)
    weighted = matrix / np.broadcast_to(scales, Y.shape).ravel()[:, None]
    target = ((Y - base) / scales).ravel()
    if not np.isfinite(weighted).all() or not np.isfinite(target).all():
        raise ValueError('Nonfinite affine design.')
    norms = np.linalg.norm(weighted, axis=0)
    norms[norms == 0] = 1.
    scaled = weighted / norms
    solution, _, rank, singular = np.linalg.lstsq(scaled, target, rcond=None)
    solution = solution / norms
    stats['affine_rank_sum'] += rank
    stats['affine_columns_sum'] += len(linear)
    stats['affine_rank_deficient_solves'] += int(rank < len(linear))
    if np.any(np.abs(solution) > bound):
        fit = lsq_linear(scaled, target, bounds=(-bound * norms, bound * norms), method='bvls', max_iter=100)
        stats['bounded_linear_solves'] += 1
        stats['bounded_linear_iterations'] += fit.nit
        if not fit.success:
            raise LinearSolveLimit()
        # Reimpose physical bounds after undoing column scaling.
        solution = np.clip(fit.x / norms, -bound, bound)
    result = constants.copy()
    result[list(linear)] = solution
    stats['separable_linear_solves'] += 1
    return result, (matrix @ solution).reshape(Y.shape) + base
