import numpy as np
import pytest

from dimensr_mimo import SearchConfig, audit
from dimensr_mimo._single.units import unit
from dimensr_mimo.generation import SystemGrammar
from dimensr_mimo.graph import in_bounds
from experiments.compositional_completion_20261031.harness import configuration
from experiments.completion_release_validation_20261104 import cases


@pytest.mark.parametrize('family', cases.FAMILIES)
def test_truth_is_representable_unit_legal_and_safe(family):
    from types import SimpleNamespace
    name = family+'_n0'
    spec = cases.design(name)
    graph, constants = cases.truth_graph(name)
    cfg = SearchConfig(**dict(configuration(SimpleNamespace(ROOT=cases.HERE.parents[1])),
                              final_selection='precision', acceptance=spec['acceptance']))
    audit.units(graph, spec['features'], spec['targets'], spec['constants'])
    assert in_bounds(graph, cfg)
    shape = audit.complexity(graph)
    assert all(shape[k] <= spec['acceptance'][k] for k in (
        'shared_graph_size', 'independent_parameters', 'expanded_total_nodes', 'maximum_output_depth'))
    grammar = SystemGrammar(*[tuple(unit(u) for u in spec[k]) for k in ('features', 'targets', 'constants')], cfg)
    for index, root in enumerate(graph.roots):
        tree, _ = graph.expanded(root, constants)
        grammar.plain[index].trace(tree)
    X = np.array([[.2, .2], [.2, 1.4], [1.4, .2], [1.4, 1.4], [.7, .8]])
    np.testing.assert_allclose(graph.evaluate(X, constants), cases.latent(name, X), rtol=1e-14, atol=1e-14)
    assert all(row['status'] == 'enclosed' for row in graph.audit(constants, spec['bounds'], 128, None))


def test_test_labels_require_seal_and_training_rejects_test(monkeypatch, tmp_path):
    monkeypatch.setattr(cases, 'HERE', tmp_path)
    assert len(cases.CASES) == 8 and len(cases.SEEDS) == 2
    for name in cases.CASES:
        for seed in cases.SEEDS:
            with pytest.raises(ValueError, match='Training'):
                cases.training_data(name, 'test', seed)
            with pytest.raises(RuntimeError, match='seal'):
                cases.generate(name, 'test', seed)
    with pytest.raises(ValueError, match='seeds'):
        cases.inputs(cases.CASES[0], 'train', 9131)


def test_separate_noise_layers_seeds_splits_and_repeatability():
    name = cases.CASES[1]
    seed = cases.SEEDS[0]
    X, Y = cases.training_data(name, 'train', seed)
    V, Z = cases.training_data(name, 'validation', seed)
    repeated = cases.training_data(name, 'train', seed)
    np.testing.assert_array_equal(X, repeated[0])
    np.testing.assert_array_equal(Y, repeated[1])
    assert X.shape == Y.shape == (96, 2) and V.shape == Z.shape == (48, 2)
    assert np.isfinite(Y).all() and np.all(Y.std(axis=0) > 0)
    assert not np.array_equal(X[:48], V)
    assert not np.array_equal(X, cases.inputs(name, 'train', cases.SEEDS[1]))
    assert not np.array_equal(X, cases.inputs(cases.CASES[0], 'train', seed))
    assert not np.array_equal(Y, cases.latent(name, X))
