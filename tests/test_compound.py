from dataclasses import replace
import sys

import numpy as np

from dimensr_mimo import SearchConfig, DimenSRMIMORegressor
from dimensr_mimo import structure
from dimensr_mimo.generation import SystemGrammar
from dimensr_mimo._single.units import ZERO, unit, scale


def inventory(grammar, X, y, compound):
    captured = {}
    prior = sys.getprofile()
    def capture(frame, event, arg):
        if event == 'return' and frame.f_code is structure.structured_seeds.__code__:
            captured.update(features=frame.f_locals['features'], proposals=frame.f_locals['proposals'])
    sys.setprofile(capture)
    try:
        seeds, counts = structure.structured_seeds(grammar, X, y, 12, compound=compound)
    finally:
        sys.setprofile(prior)
    return seeds, counts, captured


def test_compound_features_reach_retained_seeds_with_fixed_limits():
    length = unit([1, 0, 0, 0, 0, 0, 0])
    time = unit([0, 0, 1, 0, 0, 0, 0])
    cfg = SearchConfig(max_nodes=15, max_depth=6,
        operators=('add', 'mul', 'div', 'square', 'sin', 'cos', 'exp', 'log', 'sqrt'))
    grammar = SystemGrammar((time, length), (length,), (ZERO, scale(time, -1)), cfg).plain[0]
    X = np.random.default_rng(41).uniform([.1, .4], [2., 1.6], (32, 2))
    y = X[:, 1] * np.exp(-X[:, 0]) * np.sin(X[:, 0])
    old, old_counts, baseline = inventory(grammar, X, y, False)
    seeds, counts, captured = inventory(grammar, X, y, True)
    assert not any(n.token == 'exp' for t, _, _ in baseline['features'] for _, n in t.walk())
    assert any({'exp', 'sin'} <= {n.token for _, n in t.walk()} for t, c in seeds)
    assert len(seeds) <= 12 and counts['retained_features'] <= 512
    assert 0 < counts['compound_products_screened'] <= 2080 + 64 * 16
    for tree, c, values in captured['features']:
        grammar.trace(tree)
        assert tree.size <= 11 and tree.depth <= cfg.max_depth
        assert np.isfinite(values).all()
        np.testing.assert_array_equal(tree.evaluate(X, c), values)
    for tree, c in seeds:
        grammar.trace(tree)
        assert tree.units == length and tree.size <= cfg.max_nodes and tree.depth <= cfg.max_depth
        assert np.isfinite(tree.evaluate(X, c)).all()
    repeated, repeated_counts, _ = inventory(grammar, X, y, False)
    assert old_counts == repeated_counts and [t for t, _ in old] == [t for t, _ in repeated]
    for (_, a), (_, b) in zip(old, repeated):
        np.testing.assert_array_equal(a, b)


def test_compound_respects_operator_catalog_and_empty_budget():
    cfg = SearchConfig(max_nodes=7, max_depth=4, operators=('add', 'mul'))
    grammar = SystemGrammar((ZERO,), (ZERO,), (ZERO,), cfg).plain[0]
    X = np.linspace(-1., 1., 24)[:, None]
    seeds, counts = structure.structured_seeds(grammar, X, X[:, 0]**3, 4, compound=True)
    assert len(seeds) <= 4 and counts['retained_features'] <= 512
    assert all(n.token in ('add', 'mul', 'x0', 'c0') for tree, _ in seeds for _, n in tree.walk())
    assert structure.structured_seeds(grammar, X, X[:, 0], 0, compound=True)[0] == []


def test_dimensionless_inputs_keep_fittable_function_scales():
    cfg = SearchConfig(max_nodes=9, max_depth=5, operators=('add', 'mul', 'exp'))
    grammar = SystemGrammar((ZERO,), (ZERO,), (ZERO,), cfg).plain[0]
    X = np.linspace(.1, 1., 24)[:, None]
    _, _, captured = inventory(grammar, X, np.exp(-X[:, 0]), True)
    assert any(tree.token == 'exp' and tree.n_constants > 0 for tree, _, _ in captured['features'])


def test_compound_config_real_fit_save_reload(tmp_path):
    cfg = SearchConfig(initialization='compound', max_nodes=9, max_depth=5,
        operators=('add', 'mul', 'sin', 'exp'), population_size=6, epochs=1,
        gp_generations=1, selection_generations=2, max_evaluations=80,
        constant_max_nfev=5, constant_restarts=1, polish_candidates=0, min_candidates=2)
    cfg.validate()
    assert replace(cfg, initialization='legacy').initialization == SearchConfig().initialization
    X = np.random.default_rng(4).uniform(.1, 1., (24, 1))
    Y = np.column_stack((X[:, 0], 2 * X[:, 0]))
    model = DimenSRMIMORegressor(cfg).fit(X, Y, X_val=X, Y_val=Y)
    assert all('compound_products_screened' in row for row in model.diagnostics_['initialization_diagnostics'])
    model.save(tmp_path / 'model.json')
    loaded = DimenSRMIMORegressor.load(tmp_path / 'model.json')
    np.testing.assert_array_equal(model.predict(X), loaded.predict(X))
    assert model.search_fingerprint_ == loaded.search_fingerprint_
