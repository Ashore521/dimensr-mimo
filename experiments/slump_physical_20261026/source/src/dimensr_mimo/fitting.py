"""Joint raw-system coefficient fitting; DimenSR affine and domain-start mechanisms."""
import time
import numpy as np
from scipy.optimize import least_squares, lsq_linear
from ._single.fitting import affine_constants, repair_domains


class Deadline(Exception):
    pass


def fit_constants(system, warm, X, Y, scales, probes, cfg, rng, accept, stats, deadline):
    best, best_mse = None, np.inf
    start = time.monotonic()
    def evaluate(c):
        if deadline is not None and time.monotonic() >= deadline:
            raise Deadline()
        stats['constant_residual_calls'] += 1
        return system.evaluate(X, c)
    def residual(c):
        with np.errstate(all='ignore'):
            r = ((evaluate(c) - Y) / scales).ravel()
        # Optimizer penalty only. accept() always checks unmodified raw outputs.
        return np.clip(np.nan_to_num(r, nan=1e6, posinf=1e6, neginf=-1e6), -1e6, 1e6)
    def consider(c):
        nonlocal best, best_mse
        pred = evaluate(c)
        with np.errstate(all='ignore'):
            mse = float(np.mean(((pred - Y) / scales) ** 2))
        if np.isfinite(mse) and mse < best_mse and accept(c, pred):
            best, best_mse = c.copy(), mse
    def repair(c):
        c = np.array(c, copy=True)
        for root in system.roots:
            tree, expanded = system.expanded(root, c)
            repaired = repair_domains(tree, probes, expanded, cfg.constant_bound)
            ids = []
            def visit(i):
                node = system.nodes[i]
                if node.parameter is not None:
                    ids.append(node.parameter)
                for ch in node.children:
                    visit(ch)
            visit(root)
            for p in set(ids):
                # One explicit bound parameter has one repaired start, not copies.
                c[p] = np.mean(repaired[np.asarray(ids) == p])
        return c
    try:
        count = system.n_parameters
        affine = count > 0 and all(affine_constants(system.expanded(r, warm)[0]) for r in system.roots)
        if affine:
            base = evaluate(np.zeros(count))
            matrix = np.column_stack([((evaluate(row) - base) / scales).ravel() for row in np.eye(count)])
            if np.isfinite(matrix).all() and np.isfinite(base).all():
                norm = np.linalg.norm(matrix, axis=0)
                norm[norm == 0] = 1
                scaled = matrix / norm
                target = ((Y - base) / scales).ravel()
                solution = np.linalg.lstsq(scaled, target, rcond=None)[0] / norm
                if np.any(np.abs(solution) > cfg.constant_bound):
                    solution = lsq_linear(scaled, target, bounds=(-cfg.constant_bound * norm, cfg.constant_bound * norm)).x / norm
                stats['linear_fits'] += 1
                consider(solution)
        starts = [np.clip(warm, -cfg.constant_bound, cfg.constant_bound)]
        starts += [rng.uniform(-min(2., cfg.constant_bound * .9), min(2., cfg.constant_bound * .9), count)
                   for _ in range(cfg.constant_restarts - 1)] if count else []
        feasible = []
        for initial in starts:
            repaired = repair(initial)
            stats['domain_repairs'] += int(not np.array_equal(initial, repaired))
            if np.isfinite(evaluate(repaired)).all():
                feasible.append(repaired)
        if not feasible and count:
            for _ in range(cfg.domain_restarts):
                c = repair(np.clip(rng.normal(size=count) * 10 ** rng.uniform(-3, 3), -cfg.constant_bound, cfg.constant_bound))
                if np.isfinite(evaluate(c)).all():
                    feasible.append(c)
        for initial in feasible:
            consider(initial)
            if count and not affine:
                try:
                    fit = least_squares(residual, initial, bounds=(-cfg.constant_bound, cfg.constant_bound),
                                        max_nfev=cfg.constant_max_nfev, x_scale='jac')
                    stats['constant_nfev'] += fit.nfev
                    stats['optimizer_status_' + str(fit.status)] += 1
                    consider(fit.x)
                except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                    stats['optimizer_exceptions'] += 1
    except Deadline:
        stats['deadline_interruptions'] += 1
    finally:
        stats['constant_fit_seconds'] += time.monotonic() - start
    return best
