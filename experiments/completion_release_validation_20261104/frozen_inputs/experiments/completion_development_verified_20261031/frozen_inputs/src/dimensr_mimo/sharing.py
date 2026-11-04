"""Bounded structural sharing proposals, never an equivalence simplifier."""
from heapq import nsmallest
from itertools import islice
import numpy as np

from .graph import Builder, paths, in_bounds
from .selection import digest


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


def protected_closure(pool, evaluator, grammar=None):
    """Add bounded, jointly fitted sharing proposals without changing base states."""
    base = list(pool)
    extended = list(base)
    report = dict(base_candidate_count=len(base), base_candidate_sha256=digest([c.to_dict() for c in base]),
        base_search_fingerprint=digest([c.to_dict() for c in evaluator.cache.values() if c is not None]),
        base_evaluations=evaluator.calls,
        rounds=0, proposals=0, cache_skips=0, structure_rejections=0, prediction_rejections=0,
        new_evaluations=0, valid=0, protected=0, protection_rejections=0, events=[])
    if evaluator.cfg.precision_repair:
        from .repair import precision_repair
        retained, report['precision_repair'] = precision_repair(base, evaluator, grammar)
        extended.extend(retained)
    elif evaluator.cfg.compositional_completion:
        from .completion import compositional_completion
        retained, report['compositional_completion'] = compositional_completion(base, evaluator, grammar)
        extended.extend(retained)
    identities = {}

    def identity(system):
        if system not in identities:
            identities[system] = digest(system.to_dict())
        return identities[system]

    def expired():
        return evaluator.expired() or evaluator.calls >= evaluator.cfg.max_evaluations or report['new_evaluations'] >= 48

    for step in range(2):
        if expired():
            break
        parents = sorted(extended, key=lambda c:(c.error, c.complexity, identity(c.system)))[:16]
        proposals = {}
        for parent in parents:
            if expired():
                break
            prediction = parent.system.evaluate(evaluator.X, parent.constants)
            for graph, warm in matched_proposals(parent.system, parent.constants):
                if expired():
                    break
                report['proposals'] += 1
                if graph in evaluator.cache:
                    report['cache_skips'] += 1
                    continue
                if not in_bounds(graph, evaluator.cfg):
                    report['structure_rejections'] += 1
                    continue
                with np.errstate(all='ignore'):
                    distance = np.sqrt(np.mean(((graph.evaluate(evaluator.X, warm)-prediction)/evaluator.scales)**2, axis=0))
                if not np.isfinite(distance).all():
                    report['prediction_rejections'] += 1
                    continue
                rank = (float(distance.max()), graph.size, parent.error, identity(graph), identity(parent.system))
                if graph not in proposals or rank < proposals[graph][0]:
                    proposals[graph] = rank, parent, warm
        report['rounds'] += 1
        retained = []
        round_start = report['new_evaluations']
        for graph, (_, parent, warm) in sorted(proposals.items(), key=lambda item:item[1][0]):
            if expired() or report['new_evaluations']-round_start >= 24:
                break
            before = evaluator.calls
            child = evaluator(graph, warm)
            report['new_evaluations'] += evaluator.calls-before
            protected = child is not None and bool(np.all(child.errors <= parent.errors*1.1+1e-8))
            report['valid'] += int(child is not None)
            report['protected'] += int(protected)
            report['protection_rejections'] += int(child is not None and not protected)
            report['events'].append(dict(round=step, parent=parent.to_dict(), graph=graph.to_dict(),
                warm=warm.tolist(), child=None if child is None else child.to_dict(), protected=protected,
                new_evaluation=evaluator.calls>before))
            if protected:
                retained.append(child)
        extended.extend(retained)
        if not retained:
            break
    assert digest([c.to_dict() for c in base]) == report['base_candidate_sha256']
    return extended, report
