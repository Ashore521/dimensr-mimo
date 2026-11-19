"""Checkpointable rounds of the existing training-residual proposal operator."""
from collections import Counter

import numpy as np

from .audit import complexity
from .evaluation import search_score
from .generation import SystemGrammar
from .graph import Builder
from .residual import feature_library, _sum, Node, ZERO, operator
from .selection import digest


class ResidualSession:
    def __init__(self, base, evaluator, grammar, check_stop):
        self.evaluator, self.grammar = evaluator, grammar
        self.rounds = 0
        self.pending = None
        self.checkpoint_callback = None
        self.retained = []
        self.parent = min(base, key=lambda c: (search_score(c.errors, evaluator.cfg), c.complexity,
                                               digest(c.system.to_dict())))
        self.libraries, self.selected, self.blocked, self.initial, self.designs = [], [], [], [], []
        self.report = dict(mode='checkpointed_training_residual', events=[], outputs=[], initial_sources=[],
                           new_evaluations=0, protected=0, stop_reason=None)
        if 'add' not in evaluator.cfg.operators or 'mul' not in evaluator.cfg.operators:
            self.report['stop_reason'] = 'proposal_space_exhausted'
            return
        shapes = {c.system: complexity(c.system)['outputs'] for c in base}
        for j, g in enumerate(grammar.plain):
            check_stop()
            counts = Counter()
            features = feature_library(g, evaluator.X, evaluator.probes, None, counts, work_callback=check_stop)
            self.report['outputs'].append(dict(counts))
            if not features:
                self.report['stop_reason'] = 'proposal_space_exhausted'
                return
            self.libraries.append(features)
            self.selected.append([])
            self.blocked.append(set())
            minimum = min(c.errors[j] for c in base)
            bound = minimum * 1.1 + 1e-8
            eligible = [c for c in base if c.errors[j] <= bound]
            best = min(eligible, key=lambda c: (shapes[c.system][j]['expanded_nodes'],
                                               c.errors[j], c.complexity))
            self.report['initial_sources'].append(dict(output=j, graph_sha256=digest(best.system.to_dict()),
                training_error=float(best.errors[j]), minimum_training_error=float(minimum),
                training_error_bound=float(bound)))
            tree, values = best.system.expanded(best.system.roots[j], best.constants)
            self.initial.append((tree, values))
            self.designs.append(np.column_stack((np.ones(len(evaluator.X)), tree.evaluate(evaluator.X, values))))

    def __getstate__(self):
        state = dict(vars(self))
        state.pop('evaluator')
        state.pop('grammar')
        state.pop('checkpoint_callback', None)
        return state

    def attach(self, evaluator, grammar, checkpoint_callback=None):
        self.evaluator, self.grammar = evaluator, grammar
        self.checkpoint_callback = checkpoint_callback

    def _propose(self):
        if self.report['stop_reason'] is not None:
            return None
        evaluator, grammar = self.evaluator, self.grammar
        cfg = evaluator.cfg
        builder, selections, roots, warms = Builder(), [], [], []
        for j, (features, design, g) in enumerate(zip(self.libraries, self.designs, grammar.plain)):
            matrix = np.column_stack([feature[2] for feature in features])
            u, singular, _ = np.linalg.svd(design, full_matrices=False)
            evaluator.stats['residual_projection_svd'] += 1
            q = u[:, singular > max(singular[0], 1e-12) * 1e-12]
            residual = evaluator.Y[:, j] - q @ (q.T @ evaluator.Y[:, j])
            remaining = matrix - q @ (q.T @ matrix)
            norm = np.linalg.norm(remaining, axis=0)
            score = np.abs(remaining.T @ residual) / np.maximum(norm, 1e-12)
            score[norm < 1e-10] = -np.inf
            if self.blocked[j]:
                score[list(self.blocked[j])] = -np.inf
            index = int(np.argmax(score))
            if not np.isfinite(score[index]) or score[index] < 1e-10:
                index = None
            selections.append(index)
            trial = design if index is None else np.column_stack((design, matrix[:, index]))
            norms = np.linalg.norm(trial, axis=0)
            norms[norms == 0] = 1.
            weights = np.linalg.lstsq(trial / norms, evaluator.Y[:, j], rcond=None)[0] / norms
            evaluator.stats['residual_linear_subproblems'] += 1
            constants = {u: Node(t, units=u) for t, u in g.terminals.items() if t.startswith('c')}
            terms = [(constants[g.target], np.array([weights[0]])),
                     (operator('mul', constants[ZERO], self.initial[j][0]), np.r_[weights[1], self.initial[j][1]])]
            indices = self.selected[j] if index is None else self.selected[j] + [index]
            for k, weight in zip(indices, weights[2:]):
                tree, values, _ = features[k]
                terms.append((operator('mul', constants[g.target], tree), np.r_[weight, values]))
            tree, warm = _sum(terms)
            roots.append(tree)
            warms.append(warm)
        if all(index is None for index in selections):
            self.report['stop_reason'] = 'proposal_space_exhausted'
            return None
        if cfg.structure_limits is None:
            cfg.max_nodes = max(cfg.max_nodes + 16, max(tree.size for tree in roots))
            cfg.max_depth = max(cfg.max_depth, max(tree.depth for tree in roots))
            expanded = SystemGrammar(grammar.features, grammar.targets, grammar.constants, cfg, grammar.extra)
        for j, (tree, warm) in enumerate(zip(roots, warms)):
            if cfg.structure_limits is None:
                expanded.plain[j].trace(tree)
            builder.roots.append(builder.tree(tree, warm))
        graph, warm = builder.finish()
        return graph, warm, selections

    def step(self):
        if self.pending is None:
            self.pending = self._propose()
            if self.pending is None:
                return False
            if self.checkpoint_callback is not None:
                self.checkpoint_callback()
        graph, warm, selections = self.pending
        evaluator, cfg = self.evaluator, self.evaluator.cfg
        before = evaluator.calls
        child = evaluator(graph, warm)
        protected = child is not None and bool(np.all(child.errors <=
            self.parent.errors * (1 + cfg.degradation_relative) + cfg.degradation_absolute))
        self.report['new_evaluations'] += evaluator.calls - before
        self.report['protected'] += int(protected)
        self.report['events'].append(dict(round=self.rounds, protected=protected,
            new_evaluation=evaluator.calls > before, feature_indices=selections,
            graph_sha256=digest(graph.to_dict()),
            training_nrmse=None if child is None else child.errors.tolist()))
        for j, index in enumerate(selections):
            if index is not None:
                self.blocked[j].add(index)
        if protected:
            self.retained.append(child)
            if search_score(child.errors, cfg) < search_score(self.parent.errors, cfg):
                self.parent = child
            for j, index in enumerate(selections):
                if index is not None:
                    self.selected[j].append(index)
                    self.designs[j] = np.column_stack((self.designs[j], self.libraries[j][index][2]))
        self.rounds += 1
        self.pending = None
        return True
