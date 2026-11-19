"""Opt-in compact seeds and whole-system, weak-output residual proposals."""
from collections import Counter
import time

import numpy as np

from .evaluation import search_score
from .graph import Builder, Vertex, in_bounds, paths
from .pareto import violations
from .residual import feature_library, _sum, Node, operator
from .residual_session import ResidualSession


def compact_seeds(grammar, X, y, probes, terms, check_stop):
    start, cpu = time.perf_counter(), time.process_time()
    counts = Counter()
    features = feature_library(grammar, X, probes, None, counts, work_callback=check_stop)
    if not features:
        counts.update(wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu)
        return [], dict(counts)
    matrix = np.column_stack([f[2] for f in features])
    amplitude = next(Node(t, units=u) for t, u in grammar.terminals.items()
                     if t.startswith('c') and u == grammar.target)
    chosen, seeds = [], []
    design = np.ones((len(X), 1))
    for _ in range(terms):
        check_stop()
        q, singular, _ = np.linalg.svd(design, full_matrices=False)
        q = q[:, singular > singular[0] * 1e-12]
        residual = y - q @ (q.T @ y)
        projected = matrix - q @ (q.T @ matrix)
        norms = np.linalg.norm(projected, axis=0)
        score = np.abs(projected.T @ residual) / np.maximum(norms, 1e-12)
        score[norms < 1e-10] = -np.inf
        score[chosen] = -np.inf
        index = int(np.argmax(score))
        if not np.isfinite(score[index]) or score[index] < 1e-10:
            break
        chosen.append(index)
        design = np.column_stack((design, matrix[:, index]))
        norms = np.maximum(np.linalg.norm(design, axis=0), 1e-12)
        weights = np.linalg.lstsq(design / norms, y, rcond=None)[0] / norms
        counts['compact_seed_linear_subproblems'] += 1
        parts = [(amplitude, np.array([weights[0]]))]
        for k, weight in zip(chosen, weights[1:]):
            tree, values, _ = features[k]
            parts.append((operator('mul', amplitude, tree), np.r_[weight, values]))
        tree, warm = _sum(parts)
        try:
            grammar.trace(tree)
        except (ValueError, KeyError):
            continue
        seeds.append((tree, warm))
    counts['compact_seeds'] = len(seeds)
    counts.update(wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu)
    return seeds, dict(counts)


class GuidedResidualSession(ResidualSession):
    def __init__(self, base, evaluator, grammar, check_stop):
        self.evaluator, self.grammar = evaluator, grammar
        self.rounds, self.pending, self.checkpoint_callback = 0, None, None
        self.retained, self.blocked = [], set()
        reference = getattr(getattr(evaluator, 'monitor', None), 'reference', None)
        self.parent = min(base, key=lambda c: (violations(c.errors, reference, evaluator.cfg),
            search_score(c.errors, evaluator.cfg), c.error, c.complexity))
        self.working = self.parent
        self.libraries = []
        self.report = dict(mode='whole_system_guided_residual', events=[], outputs=[],
                           new_evaluations=0, protected=0, stop_reason=None)
        if not {'add', 'mul'}.issubset(evaluator.cfg.operators):
            self.report['stop_reason'] = 'operators'
            return
        for g in grammar.plain:
            counts = Counter()
            self.libraries.append(feature_library(g, evaluator.X, evaluator.probes, None, counts,
                                                  work_callback=check_stop))
            self.report['outputs'].append(dict(counts))

    def _prediction(self, graph, warm):
        self.evaluator.stats['residual_screen_predictions'] += 1
        return graph.evaluate(self.evaluator.X, warm, stats=self.evaluator.stats)

    def _propose(self):
        if self.report['stop_reason'] is not None:
            return None
        e, cfg, parent = self.evaluator, self.evaluator.cfg, self.working
        # Keep a deterministic minority of proposals for other outputs.
        j = (self.rounds // 4) % len(parent.errors) if self.rounds % 4 == 3 else int(np.argmax(parent.errors))
        pred = self._prediction(parent.system, parent.constants)
        residual = e.Y[:, j] - pred[:, j]
        features = self.libraries[j]
        choices = []
        if features:
            matrix = np.column_stack([f[2] for f in features])
            norms2 = np.sum(matrix * matrix, axis=0)
            scores = np.abs(matrix.T @ residual) / np.sqrt(np.maximum(norms2, 1e-24))
            amplitude = next(Node(t, units=u) for t, u in self.grammar.plain[j].terminals.items()
                             if t.startswith('c') and u == self.grammar.targets[j])
            for k in np.argsort(-scores, kind='stable')[:32]:
                if scores[k] < 1e-10:
                    break
                feature, values, _ = features[k]
                coefficient = float(matrix[:, k] @ residual / max(norms2[k], 1e-24))
                e.stats['residual_linear_subproblems'] += 1
                b = Builder(parent.system, parent.constants)
                term = b.tree(operator('mul', amplitude, feature), np.r_[coefficient, values])
                b.nodes.append(Vertex('add', (b.roots[j], term), self.grammar.targets[j]))
                b.roots[j] = len(b.nodes) - 1
                graph, warm = b.finish()
                if graph not in self.blocked and in_bounds(graph, cfg):
                    return graph, warm, dict(kind='residual_add', output=j, feature=int(k))
        # Removal is an evaluated alternative, never asserted algebraically
        # equivalent from training rank or from a small coefficient alone.
        for path, index in paths(parent.system, parent.system.roots[j]):
            node = parent.system.nodes[index]
            if node.token != 'add':
                continue
            for child in node.children:
                b = Builder(parent.system, parent.constants)
                b.roots[j] = b.replace(b.roots[j], path, child)
                graph, warm = b.finish()
                if graph in self.blocked or not in_bounds(graph, cfg):
                    continue
                estimate = self._prediction(graph, warm)
                errors = np.sqrt(np.mean(((estimate - e.Y) / e.scales) ** 2, axis=0))
                if np.isfinite(errors).all():
                    choices.append((search_score(errors, cfg), graph, warm))
            if len(choices) >= 32:
                break
        if choices:
            _, graph, warm = min(choices, key=lambda row: (row[0], row[1].n_parameters, row[1].size))
            return graph, warm, dict(kind='compact_proposal', output=j)
        self.report['stop_reason'] = 'proposal_space_exhausted'
        return None

    def step(self):
        e, cfg = self.evaluator, self.evaluator.cfg
        if self.pending is None:
            start, cpu = time.perf_counter(), time.process_time()
            self.pending = self._propose()
            e.stats['residual_screen_wall_seconds'] += time.perf_counter() - start
            e.stats['residual_screen_cpu_seconds'] += time.process_time() - cpu
            if self.pending is None:
                return False
            if self.checkpoint_callback:
                self.checkpoint_callback()
        graph, warm, detail = self.pending
        before = e.calls
        child = e(graph, warm)
        reference = getattr(getattr(e, 'monitor', None), 'reference', None)
        protected = child is not None and violations(child.errors, reference, cfg) == 0 and bool(np.all(
            child.errors <= self.parent.errors * (1 + cfg.degradation_relative) + cfg.degradation_absolute))
        self.blocked.add(graph)
        if protected:
            self.retained.append(child)
            if (search_score(child.errors, cfg), child.error) < (search_score(self.parent.errors, cfg), self.parent.error):
                self.parent = child
                self.working = child
            elif (child.system.n_parameters, child.system.size) < (self.working.system.n_parameters, self.working.system.size):
                self.working = child
        self.report['new_evaluations'] += e.calls - before
        self.report['protected'] += int(protected)
        self.report['events'].append(dict(round=self.rounds, **detail, protected=protected,
            new_evaluation=e.calls > before, training_nrmse=None if child is None else child.errors.tolist(),
            champion_nrmse=self.parent.errors.tolist(), parameters=graph.n_parameters))
        self.rounds += 1
        self.pending = None
        return True
