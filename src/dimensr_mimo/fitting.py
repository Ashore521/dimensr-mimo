"""Joint raw-system coefficient fitting; DimenSR affine and domain-start mechanisms."""
import time
import numpy as np
from scipy.optimize import least_squares, lsq_linear
from ._single.fitting import affine_constants, repair_domains
from .capacity import parameter_partition, output_sparsity, local_parameters
from .separable import project, LinearSolveLimit


class Deadline(Exception):
    pass


class PredictionLimit(Exception):
    pass


def fit_constants(system, warm, X, Y, scales, probes, cfg, rng, accept, stats, deadline):
    warm = np.asarray(warm, dtype=float)
    best, best_mse = None, np.inf
    trial, trial_mse = None, np.inf
    interrupted = False
    start = time.monotonic()
    cpu_start = time.process_time()
    fit_scales = np.array(scales, copy=True)
    prediction_calls = 0
    def objective(prediction):
        mse = np.mean(((prediction - Y) / scales) ** 2, axis=0)
        return float(mse.max() if cfg.search_objective == 'worst' else mse.mean())
    def charge():
        nonlocal prediction_calls
        if deadline is not None and time.monotonic() >= deadline:
            raise Deadline()
        if cfg.constant_prediction_limit is not None and prediction_calls >= cfg.constant_prediction_limit:
            raise PredictionLimit()
        prediction_calls += 1
        stats['constant_residual_calls'] += 1
        stats['constant_node_visits'] += len(system.nodes)
        stats['constant_sample_node_values'] += len(X) * len(system.nodes)
    def evaluate(c):
        charge()
        stats['constant_raw_training_predictions'] += 1
        return system.evaluate(X, c, stats=stats)
    def residual(c):
        nonlocal trial, trial_mse
        stats['constant_solver_residual_calls'] += 1
        with np.errstate(all='ignore'):
            prediction = evaluate(c)
            r = ((prediction - Y) / fit_scales).ravel()
            mse = objective(prediction)
        if (np.isfinite(mse) and mse < trial_mse and np.isfinite(c).all()
                and np.all(np.abs(c) <= cfg.constant_bound)):
            trial, trial_mse = (c.copy(), prediction.copy()), mse
            stats['constant_trial_improvements'] += 1
        # Optimizer penalty only. accept() always checks unmodified raw outputs.
        return np.clip(np.nan_to_num(r, nan=1e6, posinf=1e6, neginf=-1e6), -1e6, 1e6)
    def consider(c):
        nonlocal best, best_mse
        pred = evaluate(c)
        with np.errstate(all='ignore'):
            mse = objective(pred)
        if np.isfinite(mse) and mse < best_mse and accept(c, pred):
            best, best_mse = c.copy(), mse
    def solve(fun, initial, **options):
        stats['optimizer_starts'] += 1
        try:
            fit = least_squares(fun, initial, bounds=(-cfg.constant_bound, cfg.constant_bound),
                                max_nfev=cfg.constant_max_nfev, x_scale='jac', **options)
        except (Deadline, PredictionLimit, LinearSolveLimit, ValueError, FloatingPointError, np.linalg.LinAlgError):
            stats['optimizer_incomplete_starts'] += 1
            raise
        stats['optimizer_returns'] += 1
        stats['constant_nfev'] += fit.nfev
        stats['optimizer_status_' + str(fit.status)] += 1
        return fit
    def repair(c):
        c = np.array(c, copy=True)
        def record_tree_prediction(node, samples):
            stats['domain_tree_predictions'] += 1
            stats['domain_tree_sample_node_values'] += samples * node.size
        for root in system.roots:
            tree, expanded = system.expanded(root, c)
            repaired = repair_domains(tree, probes, expanded, cfg.constant_bound,
                                      work_callback=record_tree_prediction)
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
        if cfg.constant_fit_mode == 'local':
            c[frozen] = np.clip(warm[frozen], -cfg.constant_bound, cfg.constant_bound)
        return c
    try:
        count = system.n_parameters
        linear, nonlinear = parameter_partition(system)
        active = (getattr(cfg, '_portfolio_active_parameters', None)
                  if cfg.search_schedule == 'portfolio' and cfg.constant_fit_mode == 'local' else None)
        if active is None:
            active = local_parameters(system, cfg.constant_local_parameters) if cfg.constant_fit_mode == 'local' else tuple(range(count))
        if any(type(p) is not int or not 0 <= p < count for p in active) or len(set(active)) != len(active):
            raise ValueError('Invalid active parameter dependency set.')
        jacobian_options = (dict(jac_sparsity=output_sparsity(system, len(X)))
                            if cfg.constant_jacobian == 'output_sparse' and count else {})
        stats['fitted_parameters'] += count
        stats['structurally_linear_parameters'] += len(linear)
        if cfg.constant_focus_worst:
            clipped = np.clip(warm, -cfg.constant_bound, cfg.constant_bound)
            prediction = evaluate(clipped)
            errors = np.sqrt(np.mean(((prediction - Y) / scales) ** 2, axis=0))
            if np.isfinite(errors).all():
                # Fixed weights preserve a linear inner problem; exact worst-output
                # retention below remains authoritative, not this LS surrogate.
                weights = np.maximum((errors / max(float(errors.max()), 1e-12)) ** 2, .25)
                fit_scales = scales / np.sqrt(weights)
                if accept(clipped, prediction):
                    best, best_mse = clipped.copy(), objective(prediction)
        affine = count > 0 and cfg.constant_fit_mode != 'local' and all(
            affine_constants(system.expanded(r, warm)[0]) for r in system.roots)
        dimension = len(nonlinear) if cfg.constant_fit_mode == 'separable' else len(active)
        stats['nonlinear_dimensions'] += 0 if affine else dimension
        if affine:
            base = evaluate(np.zeros(count))
            matrix = np.column_stack([((evaluate(row) - base) / fit_scales).ravel() for row in np.eye(count)])
            if np.isfinite(matrix).all() and np.isfinite(base).all():
                norm = np.linalg.norm(matrix, axis=0)
                norm[norm == 0] = 1
                scaled = matrix / norm
                target = ((Y - base) / fit_scales).ravel()
                solution = np.linalg.lstsq(scaled, target, rcond=None)[0] / norm
                if np.any(np.abs(solution) > cfg.constant_bound):
                    scaled_solution = lsq_linear(scaled, target, bounds=(-cfg.constant_bound * norm, cfg.constant_bound * norm)).x
                    solution = np.clip(scaled_solution / norm, -cfg.constant_bound, cfg.constant_bound)
                stats['linear_fits'] += 1
                consider(solution)
        starts = [np.clip(warm, -cfg.constant_bound, cfg.constant_bound)]
        starts += [rng.uniform(-min(2., cfg.constant_bound * .9), min(2., cfg.constant_bound * .9), count)
                   for _ in range(cfg.constant_restarts - 1)] if count else []
        if cfg.constant_fit_mode == 'local':
            frozen = sorted(set(range(count)) - set(active))
            for initial in starts:
                initial[frozen] = starts[0][frozen]
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
                    if cfg.constant_fit_mode == 'separable' and linear:
                        def projected(values):
                            nonlocal trial, trial_mse
                            c = initial.copy()
                            c[list(nonlinear)] = values
                            charge()
                            stats['affine_design_predictions'] += 1
                            c, _ = project(system, c, X, Y, fit_scales, linear, cfg.constant_bound, stats)
                            prediction = evaluate(c)
                            value = objective(prediction)
                            if np.isfinite(value) and value < trial_mse and np.all(np.abs(c) <= cfg.constant_bound):
                                trial, trial_mse = (c.copy(), prediction.copy()), value
                                stats['constant_trial_improvements'] += 1
                            return c, prediction
                        def reduced_residual(values):
                            stats['constant_solver_residual_calls'] += 1
                            _, prediction = projected(values)
                            r = ((prediction - Y) / fit_scales).ravel()
                            return np.clip(np.nan_to_num(r, nan=1e6, posinf=1e6, neginf=-1e6), -1e6, 1e6)
                        if not nonlinear:
                            c, _ = projected(np.empty(0))
                            consider(c)
                            continue
                        fit = solve(reduced_residual, initial[list(nonlinear)])
                        c, _ = projected(fit.x)
                    elif cfg.constant_fit_mode == 'local' and len(active) < count:
                        def local_residual(values):
                            c = initial.copy()
                            c[list(active)] = values
                            return residual(c)
                        fit = solve(local_residual, initial[list(active)])
                        c = initial.copy()
                        c[list(active)] = fit.x
                    else:
                        fit = solve(residual, initial, **jacobian_options)
                        c = fit.x
                    consider(c)
                except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                    stats['optimizer_exceptions'] += 1
    except Deadline:
        interrupted = True
        stats['deadline_interruptions'] += 1
    except PredictionLimit:
        interrupted = True
        stats['constant_prediction_limit_hits'] += 1
    except LinearSolveLimit:
        interrupted = True
        stats['linear_solver_limit_hits'] += 1
    finally:
        # One terminal raw/domain check reuses the already charged prediction.
        # A numerical-difference trial is eligible only after this same check.
        if (interrupted or cfg.search_objective == 'worst') and trial is not None and trial_mse < best_mse:
            stats['constant_trial_validation_checks'] += 1
            if accept(*trial):
                best, best_mse = trial[0], trial_mse
                stats['constant_trial_retained'] += 1
        stats['constant_fit_seconds'] += time.monotonic() - start
        stats['constant_fit_cpu_seconds'] += time.process_time() - cpu_start
        stats['constant_fits'] += 1
        stats['constant_fit_interrupted'] += int(interrupted)
    return best
