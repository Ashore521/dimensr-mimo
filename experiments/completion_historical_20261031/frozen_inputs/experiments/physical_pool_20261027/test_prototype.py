from copy import deepcopy
import numpy as np
import pytest

from dimensr_mimo import AffineCoordinates, DimenSRMIMORegressor, PhysicalModel, SearchConfig
from dimensr_mimo._single.units import ZERO
from dimensr_mimo.evaluation import Candidate
from dimensr_mimo.graph import System, Vertex
from dimensr_mimo.selection import digest, FIELDS
from experiments.physical_pool_20261027.prototype import select_physical


def fixture():
    simple = System((Vertex('x0'),), (0,))
    longer = System((Vertex('x1'), Vertex('c0', parameter=0), Vertex('mul', (0, 1))), (2,))
    model = DimenSRMIMORegressor()
    model.config_ = SearchConfig()
    model.n_features_in_, model.n_outputs_ = 2, 1
    model.feature_units_, model.target_units_, model.constant_units_ = (ZERO, ZERO), (ZERO,), (ZERO,)
    model.feature_names_ = ['stretched', 'plain']
    model.reference_train_ = model.reference_validation_ = np.ones(1)
    model.validation_nrmse_ = np.zeros(1)
    model.scales_ = np.ones(1)
    model.search_fingerprint_ = 'constructed-rank-reversal-fixture'
    model.candidates_ = []
    for i, (system, constants) in enumerate(((simple, []), (longer, [1.]))):
        candidate = Candidate(system, np.asarray(constants), np.zeros(1), float(system.size), 0., 0.)
        model.candidates_.append(dict(candidate.to_dict(), validation_errors=[0.], violation=0., score=float(i)))
        if i == 0:
            model.best_ = candidate
    model.diagnostics_ = dict(domain=dict(input_bounds=[[-1., 1.]]*2),
        frozen_candidate_sha256=digest([{k:c[k] for k in FIELDS} for c in model.candidates_]))
    t = np.linspace(-1., 1., 32)
    X, Y = np.column_stack((10.+2.*t, t)), t[:, None]
    model.scales_ = Y.std(axis=0)
    coordinates = AffineCoordinates((10., 0.), (2., 1.), (0.,), (1.,))
    limits = dict(nrmse=[1e-4], shared_graph_size=20, independent_parameters=6, expanded_total_nodes=30, maximum_output_depth=8)
    return model, dict(physical_limits=limits, training_data=(X, Y), validation_data=(X, Y),
        coordinates=coordinates, input_bounds=[[8., 12.], [-1., 1.]])


def test_physical_complexity_reverses_search_order_and_reloads(tmp_path):
    model, kwargs = fixture()
    original = deepcopy(model.candidates_)
    physical, report = select_physical(model, **kwargs)
    assert report['status'] == 'accepted' and report['candidate_index'] == 1
    assert [r['source_structure']['shared_graph_size'] for r in report['rows']] == [1, 3]
    assert [r['physical_structure']['shared_graph_size'] for r in report['rows']] == [5, 3]
    assert model.candidates_ == original and model.best_.system.roots == (0,)
    X, Y = kwargs['training_data']
    np.testing.assert_array_equal(physical.predict(X), Y)
    physical.save(tmp_path/'physical.json')
    np.testing.assert_array_equal(PhysicalModel.load(tmp_path/'physical.json').predict(X), Y)


def test_accuracy_recomputed_and_invalid_domain_stays_in_denominator():
    model, kwargs = fixture()
    wrong = deepcopy(model.candidates_[1]);wrong['constants'] = [2.]
    invalid = Candidate(System((Vertex('x0'), Vertex('div', (0, 0))), (1,)), np.array([]), np.zeros(1), 3., 0., 0.)
    model.candidates_.extend([wrong, dict(invalid.to_dict(), validation_errors=[0.], violation=0., score=3.)])
    X, Y = kwargs['training_data']
    for candidate in model.candidates_[2:]:
        prediction = System.from_dict(candidate['system']).evaluate(kwargs['coordinates'].inputs(X), candidate['constants'])
        error = np.sqrt(np.mean(((prediction-Y)/model.scales_)**2, axis=0)).tolist()
        candidate['errors'] = candidate['validation_errors'] = error
    model.diagnostics_['frozen_candidate_sha256'] = digest([{k:c[k] for k in FIELDS} for c in model.candidates_])
    _, report = select_physical(model, **kwargs)
    assert report['candidate_count'] == 4 and report['valid_count'] == 3
    assert report['rows'][2]['accuracy_pass'] is False
    assert report['rows'][3]['numerically_valid'] is False and 'error' in report['rows'][3]


def test_no_feasible_candidate_and_pool_drift_are_explicit():
    model, kwargs = fixture()
    kwargs['physical_limits']['shared_graph_size'] = 0
    physical, report = select_physical(model, **kwargs)
    assert physical.acceptance['status'] == report['status'] == 'not_met'
    assert report['feasible_count'] == 0
    model.candidates_[1]['constants'][0] = 2.
    with pytest.raises(ValueError, match='training freeze'):
        select_physical(model, **kwargs)


def test_changed_data_cannot_reuse_old_reference_constraints():
    model, kwargs = fixture()
    X, Y = kwargs['validation_data']
    kwargs['validation_data'] = (X, Y+.1)
    with pytest.raises(ValueError, match='candidate metrics'):
        select_physical(model, **kwargs)
    model, kwargs = fixture()
    X, Y = kwargs['training_data']
    kwargs['training_data'] = (X, Y*2.)
    with pytest.raises(ValueError, match='normalization'):
        select_physical(model, **kwargs)
