import numpy as np
import pytest

from dimensr_mimo import SearchConfig
from dimensr_mimo._single.units import ZERO, unit, add, sub, scale
from dimensr_mimo.evaluation import Candidate
from dimensr_mimo.generation import SystemGrammar
from dimensr_mimo.genetics import offspring
from dimensr_mimo.graph import Builder, paths, in_bounds
from dimensr_mimo.structure import operator
from dimensr_mimo.syntax import Node


def setup():
    mass = unit([0, 1, 0, 0, 0, 0, 0])
    velocity = unit([1, 0, -1, 0, 0, 0, 0])
    cfg = SearchConfig(max_nodes=15, max_depth=6, operators=('mul', 'div', 'square', 'sqrt'))
    grammar = SystemGrammar((mass, velocity), (add(mass, velocity),), (ZERO, scale(velocity, -2)), cfg)
    a, b = Node('x0', units=mass), Node('x1', units=velocity)
    ratio = operator('sqrt', operator('div', b, a))
    tree = operator('mul', operator('square', a), operator('square', ratio))
    builder = Builder();builder.roots.append(builder.tree(tree))
    graph, constants = builder.finish()
    parent = Candidate(graph, constants, np.zeros(1), float(graph.size), 0., 0.)
    location = next(i for i, (_, node) in enumerate(paths(graph, graph.roots[0])) if graph.nodes[node].units == ratio.units)
    assert ratio.units not in grammar.plain[0].units
    return grammar, parent, location, ratio.units


class TargetedRNG:
    def __init__(self, location, seed):
        self.forced = iter((0, location));self.rng = np.random.default_rng(seed)
    def integers(self, *args, **kwargs):
        return next(self.forced, None) if not getattr(self, 'finished', False) else self.rng.integers(*args, **kwargs)
    def choice(self, *args, **kwargs):
        self.finished = True
        return self.rng.choice(*args, **kwargs)


@pytest.mark.parametrize('op', ['mutation', 'shared_mutation'])
def test_fractional_intermediate_unit_mutation_uses_a_capable_grammar(op):
    grammar, parent, location, required = setup()
    changed = 0
    for seed in range(5):
        child, constants, operation = offspring(parent, parent, grammar, TargetedRNG(location, seed), operator=op)
        assert operation == op and in_bounds(child, grammar.cfg)
        assert child.nodes[child.roots[0]].units == grammar.targets[0]
        changed += child != parent.system
    assert changed > 0
    assert grammar.mutation_grammar(0, required) is grammar.mutation_grammar(0, required)


def test_unavailable_unit_is_an_infeasible_production_not_key_error():
    grammar, _, _, required = setup()
    assert grammar.plain[0].options(required, 15, 6) == {}
    with pytest.raises(ValueError, match='No feasible production'):
        grammar.plain[0].generate(np.random.default_rng(1), units=required)
