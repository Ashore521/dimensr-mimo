"""Training-triggered implicit starts after the unchanged full base search."""
import numpy as np

from .genetics import clone_subgraph
from .graph import Builder, in_bounds
from .selection import digest
from .structure import structured_seeds


def precision_repair(base, evaluator, grammar):
    cfg = evaluator.cfg
    q = evaluator.Y.shape[1]
    eligible = sum(bool(np.all(c.errors <= np.asarray(cfg.acceptance['nrmse']))) for c in base)
    report = dict(triggered=eligible==0, training_eligible=eligible,
        seed_limit=min(12, cfg.population_size//q), new_evaluations=0, valid=0, protected=0,
        protection_rejections=0, cache_skips=0, nonimplicit_skips=0, structure_rejections=0,
        initialization_diagnostics=[], events=[], stop_reason='training_precision_met' if eligible else 'completed')
    retained = []
    if eligible:
        return retained, report
    parent = min(base, key=lambda c:(c.error, c.complexity, digest(c.system.to_dict())))
    report['parent'] = parent.to_dict()

    def stopped():
        if evaluator.expired():
            report['stop_reason'] = 'deadline'
        elif evaluator.calls >= cfg.max_evaluations:
            report['stop_reason'] = 'global_budget'
        elif report['new_evaluations'] >= 12:
            report['stop_reason'] = 'repair_budget'
        else:
            return False
        return True

    seed_lists, implicit_slots = [], []
    for j, g in enumerate(grammar.plain):
        if stopped():
            return retained, report
        try:
            seeds, stats = structured_seeds(g, evaluator.X, evaluator.Y[:, j], report['seed_limit'],
                compound=True, implicit=True, deadline=evaluator.deadline)
        except TimeoutError as error:
            report['initialization_diagnostics'].append(error.initialization_diagnostics)
            report['stop_reason'] = 'deadline'
            return retained, report
        report['initialization_diagnostics'].append(stats)
        seed_lists.append(seeds)
        implicit_slots.append(set(stats.get('implicit_retained_indices', ())))
    for k in range(max(map(len, seed_lists))):
        if stopped():
            break
        implicit_outputs = [j for j, seeds in enumerate(seed_lists) if k < len(seeds) and k in implicit_slots[j]]
        if not implicit_outputs:
            report['nonimplicit_skips'] += 1
            continue
        builder = Builder()
        for j, seeds in enumerate(seed_lists):
            if k < len(seeds):
                tree, constants = seeds[k]
                builder.roots.append(builder.tree(tree, constants))
            else:
                builder.roots.append(clone_subgraph(builder, parent.system, parent.constants, parent.system.roots[j]))
        graph, warm = builder.finish()
        if graph in evaluator.cache:
            report['cache_skips'] += 1
            continue
        if not in_bounds(graph, cfg):
            report['structure_rejections'] += 1
            continue
        before = evaluator.calls
        child = evaluator(graph, warm)
        new = evaluator.calls > before
        protected = new and child is not None and bool(np.all(child.errors <= parent.errors*1.1+1e-8))
        report['new_evaluations'] += evaluator.calls-before
        report['valid'] += int(new and child is not None)
        report['protected'] += int(protected)
        report['protection_rejections'] += int(new and child is not None and not protected)
        report['events'].append(dict(seed_index=k, implicit_outputs=implicit_outputs, parent=parent.to_dict(),
            graph=graph.to_dict(), warm=warm.tolist(), child=None if child is None else child.to_dict(),
            protected=protected, new_evaluation=new))
        if protected:
            retained.append(child)
    return retained, report
