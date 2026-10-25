from dataclasses import replace
import itertools
import numpy as np
from dimensr_mimo import SearchConfig
from dimensr_mimo.graph import Builder, System, Vertex
from dimensr_mimo._single.expression import Node
from dimensr_mimo._single.units import ZERO
from dimensr_mimo.generation import SystemGrammar
from dimensr_mimo.evaluation import Evaluator
from dimensr_mimo.genetics import offspring
from dimensr_mimo.complexity import local_rademacher


def test_binding_and_unbinding_change_fitted_hypothesis_class():
    cfg = SearchConfig(max_nodes=9, max_depth=5)
    g = SystemGrammar((ZERO,), (ZERO, ZERO), (ZERO,), cfg)
    b = Builder()
    tree = Node('mul', (Node('c0'), Node('x0')))
    b.roots = [b.tree(tree), b.tree(tree)]
    system, warm = b.finish()
    X = np.linspace(-1, 1, 40)[:, None]
    Y = np.column_stack((X[:, 0], 2 * X[:, 0]))
    independent = Evaluator(X, Y, cfg)(system, warm)
    assert independent.error < 1e-12
    rng = np.random.default_rng(3)
    bound_system, bound_warm, _ = offspring(independent, independent, g, rng, 'bind')
    bound = Evaluator(X, Y, cfg)(bound_system, bound_warm)
    assert bound_system.n_parameters == 1 and bound.error > .1
    unbound_system, unbound_warm, _ = offspring(bound, bound, g, rng, 'unbind')
    unbound = Evaluator(X, Y, cfg)(unbound_system, unbound_warm)
    assert unbound_system.n_parameters == 2 and unbound.error < 1e-12


def test_joint_jacobian_distinguishes_bound_from_equal_parameters():
    private = System((Vertex('c0', parameter=0), Vertex('c0', parameter=1)), (0, 1))
    bound = System((Vertex('c0', parameter=0),), (0, 0))
    X = np.zeros((2, 1))
    signs = np.array(list(itertools.product([-1., 1.], repeat=4)))
    a, _ = local_rademacher(private, np.ones(2), X, np.ones(2), signs, .1)
    b, _ = local_rademacher(bound, np.ones(1), X, np.ones(2), signs, .1)
    assert 0 < b < a <= .1


def test_shared_parameter_across_distinct_nonlinear_functions():
    s = System((Vertex('c0', parameter=0), Vertex('x0'), Vertex('mul', (0, 1)),
                Vertex('sin', (2,)), Vertex('x1'), Vertex('mul', (0, 4)), Vertex('cos', (5,))), (3, 6))
    X = np.random.default_rng(1).uniform(-.8, .8, (60, 2))
    Y = s.evaluate(X, [1.7])
    c = Evaluator(X, Y, SearchConfig())(s, np.ones(1))
    np.testing.assert_allclose(c.constants, [1.7], atol=1e-7)
    assert s.sharing['shared_parameters'] == 1 and s.n_parameters == 1
