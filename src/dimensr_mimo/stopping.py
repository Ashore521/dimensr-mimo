"""Training-only target checks and persistent first-hit epoch accounting."""
from numbers import Real
import numpy as np


def target_vector(metric, target, outputs=None):
    if metric is None and target is None:
        return None
    if metric not in ('nrmse', 'r2') or target is None:
        raise ValueError('Provide both stop_metric (nrmse or r2) and stop_target, or neither.')
    raw = np.asarray(target)
    if raw.ndim > 1 or not raw.size:
        raise ValueError('stop_target must be a scalar or a nonempty output vector.')
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real) for v in raw.reshape(-1)):
        raise ValueError('stop_target entries must be real numbers.')
    values = raw.astype(float).reshape(-1)
    if not np.isfinite(values).all() or (metric == 'nrmse' and np.any(values < 0)) or (metric == 'r2' and np.any(values > 1)):
        raise ValueError('Targets must be finite, with NRMSE >= 0 or R2 <= 1.')
    if outputs is not None:
        if raw.ndim == 0:
            values = np.full(outputs, values[0])
        elif len(values) != outputs:
            raise ValueError('stop_target vector must match the number of outputs; only a scalar broadcasts.')
    return values


def metric_values(candidate, metric, scales, variance):
    errors = np.asarray(candidate.errors, float)
    if metric == 'nrmse':
        return errors
    mse = (errors*np.asarray(scales))**2
    values = np.zeros_like(mse)
    varying = np.asarray(variance) > 0
    values[varying] = 1-mse[varying]/np.asarray(variance)[varying]
    values[~varying] = (mse[~varying] == 0).astype(float)
    return values


def observe(state, candidates, epoch, cfg, scales, variance):
    """Record the first single-candidate hit; later improvements keep its deadline."""
    if cfg.stop_metric is None or state['first_hit_epoch'] is not None:
        return False
    target = target_vector(cfg.stop_metric, cfg.stop_target, len(scales))
    for candidate in candidates:
        if candidate is None:
            continue
        values = metric_values(candidate, cfg.stop_metric, scales, variance)
        reached = values <= target if cfg.stop_metric == 'nrmse' else values >= target
        if np.isfinite(values).all() and bool(np.all(reached)):
            from .selection import digest
            state.update(first_hit_epoch=epoch, finish_after_epoch=epoch+cfg.stop_after_n_epochs,
                         hit_values=values.tolist(), candidate_sha256=digest(candidate.to_dict()),
                         trigger='initialization' if epoch == 0 else 'basic_search')
            return True
    return False


def completed(state):
    return state['first_hit_epoch'] is not None and state['completed_epochs'] >= state['finish_after_epoch']


def select_training(candidates, scales, variance, *, metric='r2', target=None):
    """Choose one complete training candidate without validation labels.

    Among target hits use complexity; otherwise use the worst output, with
    mean error and complexity as deterministic tie breakers.
    """
    from .selection import digest
    if metric not in ('r2', 'nrmse'):
        raise ValueError('Training selection metric must be r2 or nrmse.')
    scales, variance = np.asarray(scales, float), np.asarray(variance, float)
    if (scales.ndim != 1 or not scales.size or variance.shape != scales.shape or
            not np.isfinite(scales).all() or not np.isfinite(variance).all() or
            np.any(scales <= 0) or np.any(variance < 0)):
        raise ValueError('Finite positive scales and nonnegative training variances are required.')
    targets = None if target is None else target_vector(metric, target, len(scales))
    rows = []
    for candidate in candidates:
        if candidate is None:
            continue
        if (np.asarray(candidate.errors).shape != scales.shape or len(candidate.system.roots) != len(scales) or
                not np.isfinite(candidate.complexity) or np.any(np.asarray(candidate.errors) < 0)):
            raise ValueError('Candidate outputs and training scales must agree.')
        values = metric_values(candidate, metric, scales, variance)
        if not np.isfinite(values).all():
            continue
        hit = targets is not None and bool(np.all(values >= targets if metric == 'r2' else values <= targets))
        score = -float(values.min()) if metric == 'r2' else float(values.max())
        rows.append((candidate, values, hit, score, digest(candidate.to_dict())))
    if not rows:
        raise ValueError('No finite complete training candidates.')
    hits = [row for row in rows if row[2]]
    chosen = min(hits or rows, key=(lambda r: (r[0].complexity, r[3], r[4])) if hits else
                 (lambda r: (r[3], r[0].error, r[0].complexity, r[4])))
    candidate, values, hit, _, identity = chosen
    return candidate, dict(selection_data='training_only', metric=metric,
        target=None if targets is None else targets.tolist(), values=values.tolist(),
        target_met=hit, target_hits=len(hits), candidates=len(rows), candidate_id=identity,
        policy='minimum_complexity_target_hit' if hits else 'best_worst_output')
