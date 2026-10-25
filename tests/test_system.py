from dataclasses import replace
import json
import numpy as np
import pytest
from dimensr_mimo import SearchConfig, DimenSRMIMORegressor
from dimensr_mimo.graph import System, Vertex as V, Builder
from dimensr_mimo._single.expression import Node
from dimensr_mimo._single.units import ZERO, unit
from dimensr_mimo.evaluation import Evaluator, Candidate
from dimensr_mimo.complexity import local_rademacher
from dimensr_mimo.generation import SystemGrammar
from dimensr_mimo.genetics import offspring
from dimensr_mimo.pareto import ordered, violations


def shared():
    return System((V('c0', parameter=0), V('x0'), V('mul', (0, 1)),
                   V('sin', (2,)), V('cos', (2,))), (3, 4))


def cfg(**kw):
    base = SearchConfig(population_size=12, epochs=2, gp_generations=2, selection_generations=3,
                        max_evaluations=400, max_nodes=11, max_depth=5, min_candidates=8,
                        constant_restarts=1, domain_restarts=1, constant_max_nfev=12,
                        polish_candidates=1, polish_max_nfev=20, structured_seeds=False)
    return replace(base, **kw)


def data():
    X = np.random.default_rng(44).uniform(-1, 1, (70, 2))
    return X, np.column_stack((X[:, 0] ** 2 + X[:, 1], X[:, 0] ** 2 - X[:, 1]))


def test_explicit_graph_sharing_and_raw_expansion():
    s = shared()
    X = np.linspace(-1, 1, 31)[:, None]
    expected = np.column_stack((np.sin(1.7 * X[:, 0]), np.cos(1.7 * X[:, 0])))
    np.testing.assert_allclose(s.evaluate(X, [1.7]), expected)
    for j, r in enumerate(s.roots):
        t, c = s.expanded(r, [1.7])
        np.testing.assert_array_equal(t.evaluate(X, c), expected[:, j])
    assert s.n_parameters == 1 and s.sharing['shared_nodes'] == 3
    assert s.size == 5
    assert all(a['status'] == 'enclosed' for a in s.audit([1.7], [[-1, 1]]))


def test_equal_values_are_not_parameter_binding():
    b = Builder()
    b.roots = [b.tree(Node('c0'), [2]), b.tree(Node('c0'), [2])]
    s, c = b.finish()
    assert s.n_parameters == 2 and s.size == 2
    c[0] = 3
    np.testing.assert_array_equal(s.evaluate(np.zeros((2, 1)), c), [[3, 2], [3, 2]])


def test_joint_nonlinear_parameter_and_scales():
    s = shared()
    X = np.linspace(-.8, .8, 90)[:, None]
    Y = s.evaluate(X, [1.7])
    e = Evaluator(X, Y, cfg(constant_max_nfev=60), input_bounds=[[-.8, .8]])
    c = e(s, np.ones(1))
    assert c is not None
    np.testing.assert_allclose(c.constants, [1.7], atol=1e-6)
    assert e.stats['constant_nfev'] > 0 and np.max(c.errors) < 1e-6
    signs = np.random.default_rng(7).choice([-1., 1.], (32, Y.size))
    r, scaled = local_rademacher(s, c.constants, X, np.std(Y, axis=0), signs, .1)
    assert 0 < r <= .1 and scaled > 0


@pytest.mark.parametrize('token', ['div', 'log', 'sqrt'])
def test_boundary_illegality_is_rejected(token):
    x = Node('x0')
    t = Node(token, (x, x) if token == 'div' else (x,))
    b = Builder(); b.roots = [b.tree(t), b.tree(Node('x0'))]
    s, c = b.finish()
    X = np.array([[.2], [.4], [.7]])
    Y = s.evaluate(X, c)
    assert Evaluator(X, Y, cfg(), input_bounds=[[-1, 1]])(s, c) is None
    assert s.audit(c, [[-1, 1]])[0]['status'] == 'invalid'


def test_cancellation_cannot_hide_invalid_inner_node():
    s = System((V('x0'), V('div', (0, 0)), V('c0', parameter=0), V('mul', (1, 2))), (3,))
    assert np.isnan(s.evaluate(np.array([[0.]]), [0])).all()


def test_units_bind_and_graph_validation():
    L = unit([1, 0, 0, 0, 0, 0, 0])
    with pytest.raises(ValueError):
        System((V('c0', units=ZERO, parameter=0), V('c1', units=L, parameter=0)), (0, 1))
    with pytest.raises(ValueError):
        System((V('add', (1, 1)), V('x0')), (0,))
    with pytest.raises(ValueError):
        System((V('x0'), V('x1')), (0,))


def test_every_shared_operator_can_change_a_system():
    s = shared()
    # Add a private coefficient of the same units for parameter binding.
    b = Builder(s, [1.7])
    p = b.tree(Node('c0'), [.5]); b.nodes.append(V('add', (s.roots[1], p)))
    b.roots[1] = len(b.nodes) - 1
    s, constants = b.finish()
    c = Candidate(s, constants, np.ones(2), 10., 0., 0.)
    g = SystemGrammar((ZERO,), (ZERO, ZERO), (ZERO,), cfg(max_nodes=25, max_depth=8))
    for op in ['share', 'unshare', 'bind', 'unbind', 'shared_mutation', 'mutation', 'crossover']:
        rng = np.random.default_rng(10)
        results = [offspring(c, c, g, rng, op)[0] for _ in range(50)]
        assert any(r != s for r in results), op
    rng = np.random.default_rng(12)
    assert any(offspring(c, c, g, rng, 'bind')[0].n_parameters < s.n_parameters for _ in range(80))


def test_constrained_two_objective_order():
    s = shared()
    weak = Candidate(s, np.ones(1), np.array([0., .8]), 1., 0., 0.)
    fair = Candidate(s, np.ones(1), np.array([.4, .4]), 20., 0., 0.)
    ref = np.array([.5, .5])
    assert weak.error == fair.error  # arithmetic mean, not RMS across outputs
    assert ordered([weak, fair], cfg(), ref)[0] is fair
    assert violations(weak.errors, ref, cfg()) > 0
    assert ordered([weak, fair], cfg(use_constraints=False), ref)[0] is weak


@pytest.fixture(scope='module')
def fitted():
    X, Y = data()
    return DimenSRMIMORegressor(cfg()).fit(X[:50], Y[:50], X_val=X[50:], Y_val=Y[50:], input_bounds=[[-1, 1], [-1, 1]])


def test_full_search_mechanisms(fitted):
    h, d = fitted.history_, fitted.diagnostics_
    assert sum(x.get('neural_valid', 0) for x in h) > 0
    assert sum(x.get('parameter_delta', 0) for x in h) > 0
    assert any(x.get('imitation_systems', 0) > 0 for x in h)
    assert sum(x['new_evaluations'] for x in h if x['phase'] == 'gp') > 0
    nsga = [x for x in h if x['phase'] == 'nsga2']
    assert len(nsga) >= 2 and sum(x['new_evaluations'] for x in nsga) > 0
    assert sum(d['changed'].get(k, 0) for k in ['share', 'bind', 'shared_mutation']) > 0


def test_save_load_and_tamper(tmp_path, fitted):
    path = tmp_path / 'model.json'; fitted.save(path)
    m = DimenSRMIMORegressor.load(path)
    X, _ = data()
    np.testing.assert_array_equal(fitted.predict(X), m.predict(X))
    assert fitted.audit_domain([[-1, 1], [-1, 1]]) == m.audit_domain([[-1, 1], [-1, 1]])
    d = json.loads(path.read_text()); d['best']['constants'] += [1.]
    path.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        DimenSRMIMORegressor.load(path)


def test_validation_firewall(fitted):
    X, Y = data()
    m = DimenSRMIMORegressor(cfg()).fit(X[:50], Y[:50], X_val=X[50:], Y_val=Y[50:] * 9 + 7, input_bounds=[[-1, 1], [-1, 1]])
    assert fitted.search_fingerprint_ == m.search_fingerprint_
    assert fitted.diagnostics_['frozen_candidate_sha256'] == m.diagnostics_['frozen_candidate_sha256']


@pytest.mark.parametrize('flag', ['use_rnn', 'use_gp', 'use_sharing', 'use_rademacher', 'use_units', 'use_size_diversity', 'interval_checks'])
def test_switches_are_runnable(flag):
    X, Y = data()
    c = cfg(population_size=6, epochs=1, gp_generations=1, selection_generations=2, max_evaluations=70,
            min_candidates=1, polish_candidates=0, **{flag: False})
    m = DimenSRMIMORegressor(c).fit(X[:50], Y[:50], X_val=X[50:], Y_val=Y[50:])
    assert m.predict(X).shape == Y.shape
    if flag == 'use_rnn':
        assert m.policy_ is None
    if flag == 'use_gp':
        assert not m.diagnostics_['operations']
    if flag == 'use_sharing':
        assert all(not any(n['token'].startswith('r') for n in candidate['system']['nodes']) for candidate in m.candidates_)
        assert not m.diagnostics_['operations'].get('bind', 0)
    if flag == 'use_rademacher':
        assert m.best_.rademacher == 0
    if flag == 'interval_checks':
        assert m.diagnostics_.get('interval_enclosed', 0) == 0
