"""Coordinate syntax families across roots without targets or extra evaluations."""
from collections import defaultdict,deque
import numpy as np
from dimensr_mimo.generation import SystemGrammar
from dimensr_mimo._single.units import unit
from experiments.typed_coverage_20261026.schedule import category,balanced_prefix,typed_limit
from experiments.shared_discovery_20261026.seeds import typed_seeds


def align(lists,limit):
    if any(not rows for rows in lists):
        # No typed anchor for this unit: retain the existing structured fallback.
        return [balanced_prefix(rows,limit) for rows in lists]
    groups=[];anchors=[]
    for rows in lists:
        families=defaultdict(list)
        for tree in dict.fromkeys(rows):families[category(tree)[0]].append(tree)
        groups.append({k:deque(balanced_prefix(v,len(v))) for k,v in families.items()})
        anchors.append(next((t for t in rows if all(n.token not in ('div','sqrt','log','exp') for _,n in t.walk())),rows[0]))
    order=deque(sorted({k for g in groups for k in g}));result=[[] for _ in lists]
    while order and len(result[0])<limit:
        family=order.popleft()
        for j,g in enumerate(groups):result[j].append(g[family].popleft() if g.get(family) else anchors[j])
        if any(g.get(family) for g in groups):order.append(family)
    return result


def generator(config,spec):
    q=len(spec['targets']);pending={};next_output=0
    def scheduled(grammar,rng):
        nonlocal next_output
        j=int(np.argmax(grammar.output_context[:q]))
        if j!=next_output:raise RuntimeError('Expected one ordered seed call per output.')
        if j==0:
            # Structured seeding between these calls is deterministic and uses no RNG.
            catalog=SystemGrammar(tuple(map(unit,spec['features'])),tuple(map(unit,spec['targets'])),tuple(map(unit,spec['constants'])),config)
            lists=[list(typed_seeds(g,rng)) for g in catalog.plain]
            pending.update(enumerate(align(lists,typed_limit(config,q))))
        next_output+=1
        return iter(pending.pop(j))
    return scheduled
