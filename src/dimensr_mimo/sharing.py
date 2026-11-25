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


def protected_closure(pool, evaluator, grammar=None, *, extensions=None, max_rounds=2, max_attempts=None):
    """Add bounded, jointly fitted sharing proposals without changing base states."""
    base = list(pool)
    extended = list(base) + ([] if extensions is None else list(extensions))
    report = dict(base_candidate_count=len(base), base_candidate_sha256=digest([c.to_dict() for c in base]),
        base_search_fingerprint=digest([c.to_dict() for c in evaluator.cache.values() if c is not None]),
        base_evaluations=evaluator.calls,
        rounds=0, proposals=0, cache_skips=0, structure_rejections=0, prediction_rejections=0,
        new_evaluations=0, valid=0, protected=0, protection_rejections=0, events=[])
    if evaluator.cfg.accuracy_search and extensions is None:
        import time
        deadline = evaluator.deadline
        evaluator.deadline = time.monotonic() + max(0., deadline-time.monotonic())*.25
    try:
        if evaluator.cfg.precision_repair and extensions is None:
            from .repair import precision_repair
            retained, report['precision_repair'] = precision_repair(base, evaluator, grammar)
            extended.extend(retained)
        elif evaluator.cfg.compositional_completion and extensions is None:
            from .completion import compositional_completion
            retained, report['compositional_completion'] = compositional_completion(base, evaluator, grammar)
            extended.extend(retained)
    finally:
        if evaluator.cfg.accuracy_search and extensions is None:
            evaluator.deadline = deadline
    if evaluator.cfg.accuracy_search and extensions is None:
        from .residual import residual_search
        evaluator.deadline = deadline-evaluator.cfg.wall_seconds*.05
        try:
            retained, report['accuracy_search'] = residual_search(extended, evaluator, grammar)
        finally:
            evaluator.deadline = deadline
        extended.extend(retained)
    identities = {}
    pareto_mode = evaluator.cfg.final_selection == 'pareto'
    per_round = evaluator.cfg.sharing_evaluations_per_round if pareto_mode else 24
    evaluation_limit = max_rounds*per_round if pareto_mode else 48

    def identity(system):
        if system not in identities:
            identities[system] = digest(system.to_dict())
        return identities[system]

    def expired():
        return evaluator.expired() or evaluator.calls >= evaluator.cfg.max_evaluations or report['new_evaluations'] >= evaluation_limit

    for step in range(max_rounds):
        if expired():
            break
        display = getattr(evaluator, 'progress_display', None)
        if display is not None and evaluator.cfg.search_schedule != 'stagnation':
            display.show(dict(phase='sharing_round', step=step+1,
                              action='检查共享结构', evaluations=evaluator.calls))
        from .evaluation import search_score
        parents = sorted(extended, key=lambda c:(search_score(c.errors, evaluator.cfg), c.complexity, identity(c.system)))[:16]
        proposals = {}
        for parent in parents:
            if expired():
                break
            prediction = parent.system.evaluate(evaluator.X, parent.constants)
            for graph, warm in matched_proposals(parent.system, parent.constants):
                if expired() or (max_attempts is not None and report['proposals'] >= max_attempts):
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
                rank = (float(distance.max()), graph.size, search_score(parent.errors, evaluator.cfg), identity(graph), identity(parent.system))
                if graph not in proposals or rank < proposals[graph][0]:
                    proposals[graph] = rank, parent, warm
        report['rounds'] += 1
        retained = []
        round_start = report['new_evaluations']
        ranked = sorted(proposals.items(), key=lambda item:item[1][0])
        if pareto_mode:
            cursor = 0
            while cursor < len(ranked) and not expired() and report['new_evaluations']-round_start < per_round:
                remaining = per_round-(report['new_evaluations']-round_start)
                batch = ranked[cursor:cursor+remaining]
                cursor += len(batch)
                before, cached = evaluator.calls, set(evaluator.cache)
                values = evaluator.evaluate_many([(graph, details[2]) for graph, details in batch])
                report['new_evaluations'] += evaluator.calls-before
                for (graph, (_, parent, warm)), child in zip(batch, values):
                    protected = child is not None and bool(np.all(child.errors <=
                        parent.errors*(1+evaluator.cfg.degradation_relative)+evaluator.cfg.degradation_absolute))
                    report['valid'] += int(child is not None)
                    report['protected'] += int(protected)
                    report['protection_rejections'] += int(child is not None and not protected)
                    report['events'].append(dict(round=step, parent_sha256=identity(parent.system),
                        graph_sha256=identity(graph), protected=protected,
                        training_nrmse=None if child is None else child.errors.tolist(),
                        new_evaluation=graph not in cached and graph in evaluator.cache))
                    if protected:
                        retained.append(child)
            extended.extend(retained)
            if not retained:
                break
            continue
        for graph, (_, parent, warm) in ranked:
            if expired() or report['new_evaluations']-round_start >= per_round:
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
    if extensions is not None:
        report['stop_reason'] = ('stage_evaluation_budget' if evaluator.expired() else
            'evaluation_limit' if report['new_evaluations'] >= evaluation_limit else
            'attempt_limit' if max_attempts is not None and report['proposals'] >= max_attempts else
            'round_limit' if report['rounds'] >= max_rounds else 'no_retained_proposal')
    assert digest([c.to_dict() for c in base]) == report['base_candidate_sha256']
    return extended, report
