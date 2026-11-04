"""Rank every frozen candidate in original physical coordinates without refitting."""
from copy import copy, deepcopy
import numpy as np

from .physical import AffineCoordinates, PhysicalModel
from .acceptance import validate_limits, STRUCTURE
from .evaluation import Candidate
from .graph import System
from .pareto import violations
from .selection import choose, digest, FIELDS
from ._single.evaluation import target_scale
from .audit import data_digest


def select_physical(model, *, physical_limits, training_data, validation_data,
                    coordinates=None, input_bounds=None):
    """Return (PhysicalModel or None, report), using original training/validation data.

    New models require exact search-coordinate data hashes. Older models require
    matching normalization and all stored candidate errors. A valid fallback is
    explicitly not_met; if every physical graph is invalid, the model is None.
    """
    if not model.config_.use_units or not model.config_.use_constraints:
        raise ValueError('Physical selection requires units and per-output constraints.')
    validate_limits(physical_limits, model.n_outputs_)
    coordinates = coordinates or AffineCoordinates.identity(model.n_features_in_, model.n_outputs_)
    if len(coordinates.x_offset) != model.n_features_in_ or len(coordinates.y_offset) != model.n_outputs_:
        raise ValueError('Coordinate and model dimensions differ.')
    if input_bounds is None:
        bounds = np.asarray(model.diagnostics_['domain']['input_bounds'])
        input_bounds = (bounds*np.asarray(coordinates.x_scale)[:, None] + np.asarray(coordinates.x_offset)[:, None]).tolist()
    coordinates.input_bounds(input_bounds)
    bounds = np.asarray(input_bounds, float)
    transformed = []
    for name, (X, Y) in zip(('training', 'validation'), (training_data, validation_data)):
        X, Y = np.asarray(X, float), np.asarray(Y, float)
        coordinates.inputs(X)
        if not len(X) or Y.shape != (len(X), model.n_outputs_) or not np.isfinite(Y).all():
            raise ValueError('Provide finite paired physical training and validation arrays.')
        if np.any(X < bounds[:, 0]) or np.any(X > bounds[:, 1]):
            raise ValueError('Physical data lie outside the declared domain.')
        if len(X) != model.diagnostics_.get(name+'_rows', len(X)):
            raise ValueError('Physical data row counts differ from the original search.')
        transformed.append((coordinates.inputs(X), coordinates.outputs(Y)))
    hashes = model.diagnostics_.get('data_sha256')
    if hashes is not None:
        if set(hashes) != {'training', 'validation'} or any(
                data_digest(X, Y) != hashes[name] for name, (X, Y) in zip(('training', 'validation'), transformed)):
            raise ValueError('Physical data differ from the original search data hashes.')
    expected_scales = np.array([target_scale(y) for y in transformed[0][1].T])
    if not np.allclose(expected_scales, model.scales_, rtol=1e-7, atol=1e-10):
        raise ValueError('Physical training normalization differs from the original search.')
    pool = model.candidates_
    pool_hash = digest([{k: c[k] for k in FIELDS} for c in pool])
    if pool_hash != model.diagnostics_['frozen_candidate_sha256']:
        raise ValueError('Candidate pool differs from its training freeze.')
    guard = choose(pool, 'original')
    systems = [System.from_dict(c['system']) for c in pool]
    # Older saved models lack array hashes. Check every stored metric before
    # reusing their reference bounds; this is consistency, not byte identity.
    for system, candidate in zip(systems, pool):
        for (X, Y), key in zip(transformed, ('errors', 'validation_errors')):
            error = np.sqrt(np.mean(((system.evaluate(X, candidate['constants'])-Y)/model.scales_)**2, axis=0))
            if not np.allclose(error, candidate[key], rtol=1e-7, atol=1e-10):
                raise ValueError('Physical data do not reproduce the frozen candidate metrics.')
    rows, valid, feasible = [], {}, []
    for i, candidate in enumerate(pool):
        row = dict(candidate_index=i, candidate_sha256=digest({k: candidate[k] for k in FIELDS}),
                   numerically_valid=False, accepted=False)
        try:
            proxy = copy(model)
            proxy.best_ = Candidate(systems[i], np.asarray(candidate['constants']),
                np.asarray(candidate['errors']), candidate['complexity'], candidate['rademacher'], candidate['rademacher_complexity'])
            proxy.validation_nrmse_ = np.asarray(candidate['validation_errors'])
            proxy.post_selection_ = dict(validation_bounds=guard['validation_bounds'])
            proxy.diagnostics_ = dict(model.diagnostics_)
            proxy.diagnostics_.pop('application_acceptance', None)
            physical = PhysicalModel.from_estimator(proxy, coordinates, input_bounds,
                training_data=training_data, validation_data=validation_data)
            accuracy = physical.source['physical_accuracy']
            train, validation = np.asarray(accuracy['train_nrmse']), np.asarray(accuracy['validation_nrmse'])
            amount = (violations(train, model.reference_train_, model.config_) +
                      violations(validation, model.reference_validation_, model.config_) +
                      float(np.maximum(validation - guard['validation_bounds'], 0).sum()))
            protected = amount == 0
            absolute = bool(np.all(train <= physical_limits['nrmse']) and np.all(validation <= physical_limits['nrmse']))
            simple = all(physical.structure[k] <= physical_limits[k] for k in STRUCTURE)
            physical.source['relative_constraints_ok'] = protected
            shape = tuple(physical.structure[k] for k in STRUCTURE)
            row.update(numerically_valid=True, physical_structure=physical.structure,
                source_structure=physical.source_structure, physical_accuracy=deepcopy(accuracy),
                relative_violation=amount, relative_constraints_pass=protected,
                accuracy_pass=absolute, structure_pass=simple, accepted=absolute and simple and protected)
            valid[i] = physical
            if row['accepted']:
                feasible.append((shape + (float(validation.mean()), i), i))
        except (ValueError, FloatingPointError) as error:
            row['error'] = str(error)
        rows.append(row)
    if not valid:
        index = None
    elif feasible:
        index = min(feasible)[1]
    else:
        index = min(valid, key=lambda i: (rows[i]['relative_violation'],
            float(np.mean(rows[i]['physical_accuracy']['validation_nrmse'])),
            tuple(rows[i]['physical_structure'][k] for k in STRUCTURE), i))
    report = dict(method='physical_pool_precision_then_structure', status='accepted' if feasible else 'not_met',
        candidate_count=len(pool), valid_count=len(valid), feasible_count=len(feasible), candidate_index=index,
        original_selected_index=getattr(model, 'post_selection_', {}).get('candidate_index', guard['anchor_index']),
        weighted_anchor_index=guard['anchor_index'], validation_bounds=guard['validation_bounds'],
        relative_reference=dict(training=model.reference_train_.tolist(), validation=model.reference_validation_.tolist(),
            degradation_relative=model.config_.degradation_relative, degradation_absolute=model.config_.degradation_absolute),
        limits=deepcopy(physical_limits), complexity_priority=list(STRUCTURE),
        pool_sha256=pool_hash, search_fingerprint=model.search_fingerprint_, rows=rows,
        source_data_check=dict(method='all_candidate_metrics_and_training_normalization', rtol=1e-7, atol=1e-10,
            byte_identity_verified=hashes is not None, search_data_sha256=deepcopy(hashes)),
        fallback_method=None if feasible or not valid else 'physical_relative_violation_then_validation_error',
        reason=None if feasible else 'No valid physical graph.' if not valid else 'No candidate satisfies every frozen gate.',
        test_data_used=False, coefficients_refitted=False,
        scope='Frozen-pool training/validation selection; finite domain screening, not continuous proof.')
    if index is None:
        return None, report
    selected = valid[index]
    selected.source['physical_pool_selection'] = dict(candidate_index=index, pool_sha256=pool_hash,
        report_sha256=digest(report), method=report['method'])
    result = PhysicalModel(selected.source, coordinates, input_bounds, physical_limits)
    assert result.acceptance['status'] == report['status']
    return result, report
