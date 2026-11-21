import numpy as np
import pytest

from dimensr_mimo import DimenSRMIMORegressor, SearchConfig, System, Vertex
from dimensr_mimo._single.units import ZERO
from dimensr_mimo.compression import compression_candidate
from dimensr_mimo.evaluation import Candidate, Evaluator


def test_nonfinite_intermediates_cannot_disappear_in_a_reciprocal():
    g = System((Vertex('x0'), Vertex('c0', parameter=0), Vertex('square', (0,)),
                Vertex('div', (1, 2))), (0, 3))
    x = np.array([1e300, -2., np.inf, -0., np.nan])[:, None]
    pred = g.evaluate(x, [1.])
    np.testing.assert_array_equal(pred[:, 0], x[:, 0])
    np.testing.assert_array_equal(pred[:, 1], [np.nan, .25, np.nan, np.nan, np.nan])
    assert np.signbit(pred[3, 0])


def test_sparse_training_candidate_preserves_joint_target():
    x = np.linspace(.1, 1., 64)[:, None]
    y = np.column_stack((3*x[:, 0]+2, -2*x[:, 0]+1))
    cfg = SearchConfig.accuracy_pareto(max_evaluations=32, compression_r2_drop=.25,
                                       interval_checks=True, max_nodes=60, max_depth=30)
    e = Evaluator(x, y, cfg, input_bounds=[(.1, 1.)])
    nodes = [Vertex('x0'), Vertex('c0', parameter=0), Vertex('mul', (0, 1)),
             Vertex('c0', parameter=1), Vertex('add', (2, 3)), Vertex('c0', parameter=2),
             Vertex('mul', (0, 5)), Vertex('c0', parameter=3), Vertex('add', (6, 7))]
    roots = [4, 8]
    constants = [3., 2., -2., 1.]
    for _ in range(4):
        for j in range(2):
            nodes.append(Vertex('c0', parameter=len(constants)))
            constants.append(0.)
            nodes.append(Vertex('add', (roots[j], len(nodes)-1)))
            roots[j] = len(nodes)-1
    graph = System(tuple(nodes), tuple(roots))
    parent = Candidate(graph, np.asarray(constants), np.zeros(2), float(graph.size), 0., 0.)
    compact, report = compression_candidate([parent], e, [ZERO], [ZERO]*2, [ZERO], cfg, np.ones(2))
    assert report['accepted'] and compact.complexity < parent.complexity
    np.testing.assert_array_equal(report['floors'], [.90, .90])
    np.testing.assert_allclose(compact.system.evaluate(x, compact.constants), y, atol=1e-12)


def test_fit_compression_freezes_before_validation_and_roundtrips(tmp_path):
    x = np.linspace(.1, 1., 32)[:, None]
    y = np.column_stack((x[:, 0], 2*x[:, 0]))
    cfg = SearchConfig.accuracy_pareto(max_evaluations=24, population_size=4, max_nodes=12,
        max_depth=5, gp_generations=1, min_candidates=4, interval_checks=True,
        structured_seeds=False, constant_restarts=1, constant_max_nfev=3, random_state=42,
        polish_candidates=0, compression_r2_drop=[0., .005])
    def fit(labels):
        return DimenSRMIMORegressor(cfg).fit(x, y, X_val=x[:8], Y_val=labels, input_bounds=[(.1, 1.)])
    a, b = fit(y[:8]), fit(-y[:8])
    assert a.diagnostics_['training_compression'] == b.diagnostics_['training_compression']
    assert a.search_fingerprint_ == b.search_fingerprint_
    path = tmp_path/'model.json'
    a.save(path)
    loaded = DimenSRMIMORegressor.load(path)
    np.testing.assert_array_equal(loaded.predict(x), a.predict(x))
    assert loaded.diagnostics_['training_compression'] == a.diagnostics_['training_compression']


@pytest.mark.parametrize('tol', [-1, True, [float('nan')], []])
def test_invalid_compression_tolerance(tol):
    with pytest.raises(ValueError, match='compression_r2_drop'):
        SearchConfig.accuracy_pareto(compression_r2_drop=tol, interval_checks=True)
