"""Complete the existing typed seed vocabulary with parameterized sin/cos.

No labels, target formula, target parameter values or output pairing are used.
The original seeds are retained; these generic univariate atoms are appended.
"""
from dimensr_mimo._single.expression import Node
from dimensr_mimo._single.units import ZERO, sub
from dimensr_mimo._single.seeding import typed_seeds as original_seeds


def nonlinear_seeds(grammar):
    if 'mul' not in grammar.tokens:
        return
    def coefficient(units):
        return next((Node(t,units=u) for t,u in grammar.terminals.items()
                     if t.startswith('c') and u==units),None)
    features=[Node(t,units=u) for t,u in grammar.terminals.items() if t.startswith('x')]
    amplitude=coefficient(grammar.target)
    seen=set()
    for x in features:
        scale=coefficient(sub(ZERO,x.units))
        if scale is None:
            continue
        for op in ('sin','cos'):
            if op not in grammar.tokens:
                continue
            atom=Node(op,(Node('mul',(scale,x),ZERO),),ZERO)
            proposals=([atom] if grammar.target==ZERO else [])
            if amplitude is not None:
                proposals.append(Node('mul',(amplitude,atom),grammar.target))
            for tree in proposals:
                tree,_=tree.canonical()
                if tree in seen or tree.size>grammar.max_nodes or tree.depth>grammar.max_depth:
                    continue
                try:
                    grammar.trace(tree)
                except (ValueError,KeyError):
                    continue
                seen.add(tree)
                yield tree


def typed_seeds(grammar,rng):
    original=list(original_seeds(grammar,rng))
    yield from original
    present=set(original)
    for tree in nonlinear_seeds(grammar):
        if tree not in present:
            yield tree
