"""Allocate a fixed seed budget across syntax families and feature supports.

Only syntax and configured budgets are used. All proposals still pass the
unchanged joint evaluator, constant fitter and numerical-domain checks.
"""
from collections import defaultdict,deque
from experiments.shared_discovery_20261026.seeds import typed_seeds as appended_seeds

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
