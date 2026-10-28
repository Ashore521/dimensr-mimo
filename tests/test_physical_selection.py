from copy import deepcopy
import numpy as np
import pytest

from dimensr_mimo import AffineCoordinates, DimenSRMIMORegressor, PhysicalModel, SearchConfig
from dimensr_mimo._single.units import ZERO
from dimensr_mimo.evaluation import Candidate
from dimensr_mimo.graph import System, Vertex
from dimensr_mimo.selection import digest, FIELDS
from dimensr_mimo import select_physical


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
    model.diagnostics_['application_acceptance'] = dict(status='accepted', candidate_index=0)
    original = deepcopy(model.candidates_)
    physical, report = select_physical(model, **kwargs)
    assert report['status'] == 'accepted' and report['candidate_index'] == 1
    assert [r['source_structure']['shared_graph_size'] for r in report['rows']] == [1, 3]
    assert [r['physical_structure']['shared_graph_size'] for r in report['rows']] == [5, 3]
    assert model.candidates_ == original and model.best_.system.roots == (0,)
    assert physical.source['search_coordinate_acceptance'] is None
    assert model.diagnostics_['application_acceptance']['candidate_index'] == 0
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


def test_relative_guard_blocks_a_smaller_accurate_candidate():
    model, kwargs = fixture()
    model.candidates_[1]['constants'] = [1.0001]
    X, Y = kwargs['training_data']
    candidate = model.candidates_[1]
    prediction = System.from_dict(candidate['system']).evaluate(kwargs['coordinates'].inputs(X), candidate['constants'])
    error = np.sqrt(np.mean(((prediction-Y)/model.scales_)**2, axis=0)).tolist()
    candidate['errors'] = candidate['validation_errors'] = error
    model.diagnostics_['frozen_candidate_sha256'] = digest([{k:c[k] for k in FIELDS} for c in model.candidates_])
    kwargs['physical_limits']['nrmse'] = [1.]
    _, report = select_physical(model, **kwargs)
    assert report['candidate_index'] == 0
    assert report['rows'][1]['accuracy_pass'] and not report['rows'][1]['relative_constraints_pass']


def test_all_invalid_physical_candidates_return_complete_failure_report():
    model, kwargs = fixture()
    graph = System((Vertex('x0'), Vertex('div', (0, 0))), (1,))
    X, Y = kwargs['training_data']
    error = np.sqrt(np.mean(((graph.evaluate(kwargs['coordinates'].inputs(X), [])-Y)/model.scales_)**2, axis=0)).tolist()
    candidate = Candidate(graph, np.array([]), np.asarray(error), 3., 0., 0.)
    model.candidates_ = [dict(candidate.to_dict(), validation_errors=error, score=1., violation=0.)]
    model.diagnostics_['frozen_candidate_sha256'] = digest([{k:c[k] for k in FIELDS} for c in model.candidates_])
    physical, report = select_physical(model, **kwargs)
    assert physical is None and report['status'] == 'not_met' and report['candidate_index'] is None
    assert len(report['rows']) == report['candidate_count'] == 1 and report['valid_count'] == 0


def test_new_fit_hashes_survive_reload_and_reject_changed_arrays(tmp_path):
    cfg = SearchConfig(population_size=6, epochs=1, gp_generations=1, selection_generations=2,
        max_evaluations=60, constant_max_nfev=5, constant_restarts=1, min_candidates=2,
        polish_candidates=0, structured_seeds=False, operators=('add', 'mul'), max_nodes=7, max_depth=4)
    X = np.random.default_rng(31).uniform(.2, .8, (20, 2))
    Y = (X[:, 0]+X[:, 1])[:, None]
    m = DimenSRMIMORegressor(cfg).fit(X, Y, X_val=X, Y_val=Y, input_bounds=[[0.,1.]]*2)
    m.save(tmp_path/'model.json');m = DimenSRMIMORegressor.load(tmp_path/'model.json')
    limits = dict(nrmse=[2.], shared_graph_size=30, independent_parameters=10, expanded_total_nodes=40, maximum_output_depth=8)
    physical, report = select_physical(m, physical_limits=limits, training_data=(X, Y), validation_data=(X, Y))
    assert physical is not None and report['source_data_check']['byte_identity_verified']
    changed = X.copy();changed[0, 0] += 1e-12
    with pytest.raises(ValueError, match='search data hashes'):
        select_physical(m, physical_limits=limits, training_data=(X, Y), validation_data=(changed, Y))


def test_dimension_domain_and_unit_constraints_checked_before_selection():
    model, kwargs = fixture()
    kwargs['coordinates'] = AffineCoordinates.identity(1, 1)
    with pytest.raises(ValueError, match='dimensions'):
        select_physical(model, **kwargs)
    model, kwargs = fixture();kwargs['input_bounds'] = [[9.,11.],[-1.,1.]]
    with pytest.raises(ValueError, match='outside'):
        select_physical(model, **kwargs)
    model, kwargs = fixture();model.config_.use_units = False
    with pytest.raises(ValueError, match='requires units'):
        select_physical(model, **kwargs)


def test_physical_selection_preserves_shared_parameter_units():
    from dimensr_mimo._single.units import unit
    length = unit([1, 0, 0, 0, 0, 0, 0])
    model, kwargs = fixture()
    graph = System((Vertex('x0', units=length), Vertex('x1', units=length), Vertex('c0', parameter=0),
        Vertex('mul', (2, 0), length), Vertex('add', (0, 1), length), Vertex('mul', (2, 4), length)), (3, 5))
    candidate = Candidate(graph, np.array([2.]), np.zeros(2), float(graph.size), 0., 0.)
    model.n_outputs_ = 2
    model.feature_units_ = model.target_units_ = (length, length)
    model.best_ = candidate;model.validation_nrmse_ = np.zeros(2)
    model.reference_train_ = model.reference_validation_ = np.ones(2)
    model.candidates_ = [dict(candidate.to_dict(), validation_errors=[0., 0.], score=0., violation=0.)]
    model.diagnostics_['frozen_candidate_sha256'] = digest([{k:c[k] for k in FIELDS} for c in model.candidates_])
    coordinates = AffineCoordinates((10., 20.), (2., 3.), (4., 8.), (5., 7.))
    X = np.random.default_rng(13).uniform([8., 17.], [12., 23.], (32, 2))
    search_y = graph.evaluate(coordinates.inputs(X), candidate.constants)
    model.scales_ = search_y.std(axis=0)
    Y = coordinates.restore_outputs(search_y)
    limits = dict(nrmse=[1e-4]*2, shared_graph_size=50, independent_parameters=12, expanded_total_nodes=70, maximum_output_depth=10)
    physical, report = select_physical(model, physical_limits=limits, training_data=(X,Y), validation_data=(X,Y),
        coordinates=coordinates, input_bounds=[[8.,12.],[17.,23.]])
    assert report['status'] == 'accepted'
    record = physical.to_dict()
    assert record['fitted_parameters'] == 1 and record['fixed_coordinate_parameters'] == 8
    shared = next(p for p in record['parameters'] if p['role'] == 'fitted')
    assert shared['outputs'] == [0,1] and shared['units'] == ['0']*7
    np.testing.assert_allclose(physical.predict(X), Y, rtol=1e-14, atol=1e-14)
