"""Bounded structural sharing proposals, never an equivalence simplifier."""
from heapq import nsmallest
from itertools import islice

from .graph import Builder, paths


def signature(system, root):
    """Ordered structure and local parameter aliases, excluding fitted values."""
    parameters = {}

    def visit(i):
        v = system.nodes[i]
        if v.parameter is not None:
            if v.parameter not in parameters:
                parameters[v.parameter] = len(parameters)
            return ('parameter', v.units, parameters[v.parameter])
        return (v.token, v.units, tuple(visit(c) for c in v.children))

    return visit(root)


def matched_proposals(system, constants):
    """Return at most 16 smaller systems from the first 64 ordered matches.

    Subgraphs with more than 256 expanded nodes are outside this bounded
    operator. Every returned graph still requires joint fitting and validation.
    """
    sizes, parameterized = [], []
    for v in system.nodes:
        sizes.append(min(257, 1 + sum(sizes[c] for c in v.children)))
        parameterized.append(v.parameter is not None or any(parameterized[c] for c in v.children))
    groups = {}
    keys = {}
    for i, v in enumerate(system.nodes):
        if v.children and parameterized[i] and sizes[i] <= 256:
            keys[i] = signature(system, i)
            groups.setdefault(keys[i], []).append(i)
    reach = [system.reachable(r) for r in system.roots]

    def matches():
        for j, root in enumerate(system.roots):
            other = set().union(*(s for k, s in enumerate(reach) if k != j))
            for path, old in islice(paths(system, root), 256):
                if old in keys:
                    for donor in groups[keys[old]]:
                        if donor != old and donor in other:
                            yield (-sizes[old], j, path, donor, old)

    result, seen = [], set()
    for _, j, path, donor, old in nsmallest(64, matches()):
        builder = Builder(system, constants)
        builder.roots[j] = builder.replace(system.roots[j], path, donor)
        child, warm = builder.finish()
        if child.size < system.size and child not in seen:
            result.append((child, warm))
            seen.add(child)
            if len(result) == 16:
                break
    return result
