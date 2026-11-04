"""Complete the existing typed seed vocabulary with parameterized sin/cos.

No labels, target formula, target parameter values or output pairing are used.
The original seeds are retained; these generic univariate atoms are appended.
"""
from ._single.expression import Node
from ._single.units import ZERO, sub
from ._single.seeding import typed_seeds as original_seeds


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

from collections import defaultdict,deque
appended_seeds = typed_seeds

UNARY=frozenset(('square','sqrt','log','exp','sin','cos'))


def category(tree):
    nodes=[n for _,n in tree.walk()]
    support=tuple(sorted({int(n.token[1:]) for n in nodes if n.token.startswith('x')}))
    unary={n.token for n in nodes if n.token in UNARY}
    family=('constant' if not support else next(iter(unary)) if len(unary)==1 else
            'composition' if unary else 'ratio' if any(n.token=='div' for n in nodes) else 'affine')
    return family,support


def balanced_prefix(trees,limit):
    if type(limit) is not int or limit<0:raise ValueError('Seed limit must be a nonnegative integer.')
    buckets=defaultdict(lambda:defaultdict(deque))
    for tree in dict.fromkeys(trees):
        family,support=category(tree);buckets[family][support].append(tree)
    families=deque((family,deque(buckets[family][support] for support in sorted(buckets[family]))) for family in sorted(buckets))
    selected=[]
    while families and len(selected)<limit:
        family,supports=families.popleft();queue=supports.popleft()
        selected.append(queue.popleft())
        if queue:supports.append(queue)
        if supports:families.append((family,supports))
    return selected


def typed_limit(config,n_outputs):
    limit=max(1,int(config.max_evaluations*config.initialization_fraction))
    reserve=min(limit-1,max(1,config.population_size//n_outputs)) if config.structured_seeds else 0
    return limit-reserve


def generator(config,n_outputs):
    limit=typed_limit(config,n_outputs)
    def scheduled(grammar,rng):
        return iter(balanced_prefix(list(appended_seeds(grammar,rng)),limit))
    return scheduled
