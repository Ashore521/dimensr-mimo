import importlib.util
from pathlib import Path
import numpy as np
import pytest

from dimensr_mimo import SearchConfig
from dimensr_mimo._single.units import ZERO, unit
from dimensr_mimo.evaluation import Candidate, Evaluator
from dimensr_mimo.genetics import offspring
from dimensr_mimo.generation import SystemGrammar
from dimensr_mimo.graph import Builder, System, Vertex
from dimensr_mimo.sharing import matched_proposals, signature
from dimensr_mimo.structure import operator
from dimensr_mimo.syntax import Node


def pair(first=1., second=2., variable='x0'):
    b = Builder()
    b.roots = [b.tree(operator('mul', Node('c0'), Node('x0')), [first]),
               b.tree(operator('mul', Node('c0'), Node(variable)), [second])]
    return b.finish()


def test_matching_is_a_smaller_hypothesis_not_functional_equivalence():
    system, constants = pair()
    original = system.to_dict()
    proposals = matched_proposals(system, constants)
    assert len(proposals) == 1  # Opposite directions have the same canonical graph.
    X = np.linspace(.1, 1., 40)[:, None]
    Y = system.evaluate(X, constants)
    for graph, warm in proposals:
        assert graph.size < system.size and graph.n_parameters == 1
        assert graph.sharing['shared_nodes'] == 3
        assert not np.array_equal(graph.evaluate(X, warm), Y)
        fitted = Evaluator(X, Y, SearchConfig())(graph, warm)
        assert fitted is not None and np.all(fitted.errors > .1)
    assert system.to_dict() == original
    np.testing.assert_array_equal(constants, [1., 2.])


def test_joint_fit_can_recover_a_shared_parameter_after_proposal():
    b = Builder()
    for op in ('sin', 'cos'):
        b.roots.append(b.tree(operator(op, operator('mul', Node('c0'), Node('x0'))), [1.]))
    system, constants = b.finish()
    X = np.linspace(.1, .8, 40)[:, None]
    Y = np.column_stack((np.sin(1.7*X[:, 0]), np.cos(1.7*X[:, 0])))
    for graph, warm in matched_proposals(system, constants):
        candidate = Evaluator(X, Y, SearchConfig())(graph, warm)
        assert candidate is not None and np.all(candidate.errors < 1e-8)
        np.testing.assert_allclose(candidate.constants, [1.7], atol=1e-7)


def test_parameter_alias_pattern_and_variable_order_are_preserved():
    s = System((Vertex('c0', parameter=0), Vertex('c0', parameter=0), Vertex('mul', (0, 1)),
                Vertex('c0', parameter=1), Vertex('c0', parameter=2), Vertex('mul', (3, 4)),
                Vertex('c0', parameter=3), Vertex('c0', parameter=3), Vertex('mul', (6, 7))), (2, 5, 8))
    assert signature(s, 2) == signature(s, 8)
    assert signature(s, 2) != signature(s, 5)
    variables, values = pair(variable='x1')
    assert matched_proposals(variables, values) == []
    length = unit([1, 0, 0, 0, 0, 0, 0])
    time = unit([0, 0, 1, 0, 0, 0, 0])
    dimensional = System((Vertex('c0', units=length, parameter=0), Vertex('x0'), Vertex('mul', (0, 1), length),
                          Vertex('c1', units=time, parameter=1), Vertex('x0'), Vertex('mul', (3, 4), time)), (2, 5))
    assert signature(dimensional, 2) != signature(dimensional, 5)
    assert matched_proposals(dimensional, [1., 1.]) == []


def test_proposals_are_bounded_deterministic_and_evaluator_keeps_structure_gate():
    system, constants = pair()
    first, second = matched_proposals(system, constants), matched_proposals(system, constants)
    assert 0 < len(first) <= 16
    for (a, av), (b, bv) in zip(first, second):
        assert a == b
        np.testing.assert_array_equal(av, bv)
    X = np.linspace(.1, 1., 20)[:, None]
    evaluator = Evaluator(X, system.evaluate(X, constants), SearchConfig(max_nodes=2))
    assert evaluator(*first[0]) is None
    assert evaluator.stats['structure_bound_rejections'] == 1


def test_domain_failure_is_not_hidden_by_sharing():
    b = Builder()
    tree = operator('add', Node('c0'), operator('div', Node('x0'), operator('sub', Node('x0'), Node('x0'))))
    b.roots = [b.tree(tree), b.tree(tree)]
    system, constants = b.finish()
    proposals = matched_proposals(system, constants)
    assert proposals
    X = np.linspace(.1, 1., 20)[:, None]
    for graph, warm in proposals:
        assert Evaluator(X, np.ones((20, 2)), SearchConfig())(graph, warm) is None


def test_operator_dispatch_and_default_compatibility():
    system, constants = pair()
    parent = Candidate(system, constants, np.zeros(2), float(system.size), 0., 0.)
    cfg = SearchConfig()
    assert cfg.sharing_strategy == 'random'
    grammar = SystemGrammar((ZERO,), (ZERO, ZERO), (ZERO,), cfg)
    child, _, op = offspring(parent, parent, grammar, np.random.default_rng(3), 'matched_share')
    assert op == 'matched_share' and child.size < system.size
    # Compare to the published 1.5.0 implementation, including RNG consumption.
    path = Path(__file__).resolve().parents[1] / 'experiments/physical_pool_20261027/source/src/dimensr_mimo/genetics.py'
    spec = importlib.util.spec_from_file_location('dimensr_mimo._frozen_genetics', path)
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    for seed in range(40):
        a = np.random.default_rng(seed); b = np.random.default_rng(seed)
        default = offspring(parent, parent, grammar, a)
        previous = old.offspring(parent, parent, grammar, b)
        assert default[0] == previous[0] and default[2] == previous[2]
        np.testing.assert_array_equal(default[1], previous[1])
        assert a.random() == b.random()
    cfg.sharing_strategy = 'matched'
    operators = {offspring(parent, parent, grammar, np.random.default_rng(seed))[2] for seed in range(200)}
    assert {'share', 'matched_share', 'bind', 'unbind', 'unshare', 'shared_mutation'} <= operators
    with pytest.raises(ValueError, match='sharing_strategy'):
        SearchConfig(sharing_strategy='invalid').validate()
