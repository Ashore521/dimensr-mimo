"""Training-residual expansion with growing expression capacity and raw domains."""
from itertools import combinations_with_replacement
import time

import numpy as np

from ._single.expression import Node
from ._single.units import ZERO, sub
from .audit import complexity
from .generation import SystemGrammar
from .graph import Builder
from .selection import digest
from .structure import operator
from .evaluation import search_score


def feature_library(grammar, X, probes, deadline, counts, work_callback=None):
    """Cover every input before interactions, within a numerical memory budget."""
    constants = {u: Node(t, units=u) for t, u in grammar.terminals.items() if t.startswith('c')}
    amplitude = constants.get(grammar.target)
    one = constants.get(ZERO)
    if amplitude is None or one is None or 'mul' not in grammar.tokens:
        return []
    features, seen, signatures = [], set(), set()
    memory_limit = 256 * 1024**2

    def admit(tree):
        if work_callback is not None:
            work_callback()
        if (deadline is not None and time.monotonic() >= deadline) or len(features)*len(X)*8 >= memory_limit:
            return False
        tree, values = tree.canonical()
        if tree in seen:
            return True
        seen.add(tree)
        counts['feature_attempts'] += 1
        try:
            grammar.trace(tree)
        except (ValueError, KeyError):
            counts['feature_structure_rejections'] += 1
            return True
        prediction = tree.evaluate(X, values)
        if not np.isfinite(prediction).all() or not np.isfinite(tree.evaluate(probes, values)).all():
            counts['feature_domain_rejections'] += 1
            return True
        centered = prediction-prediction.mean()
        norm = np.linalg.norm(centered)
        if not np.isfinite(norm) or norm < 1e-12:
            return True
        signature = np.round(centered/norm, 12).tobytes()
        if signature in signatures:
            return True
        signatures.add(signature)
        features.append((tree, values, prediction))
        return True

    atoms = []
    for token, units in grammar.terminals.items():
        if not token.startswith('x'):
            continue
        tree = Node(token, units=units)
        if units != ZERO:
            if 'div' not in grammar.tokens:
                continue
            if units in constants:
                tree = operator('div', tree, constants[units])
            elif sub(grammar.target, units) in constants:
                tree = operator('div', operator('mul', constants[sub(grammar.target, units)], tree), amplitude)
            else:
                continue
        atoms.append(tree)
        if not admit(tree):
            break
    # Every variable is considered before any unary or interaction family.
    products = list(atoms)
    for tree in atoms:
        for token in ('square', 'sin', 'cos', 'log', 'sqrt', 'exp'):
            if token in grammar.tokens and not admit(operator(token, tree)):
                return features
        if 'div' in grammar.tokens:
            inverse = operator('div', one, tree)
            if not admit(inverse):
                return features
            products.append(inverse)
            if 'square' in grammar.tokens:
                if not admit(operator('square', inverse)):
                    return features
    for left, right in combinations_with_replacement(products, 2):
        if not admit(operator('mul', left, right)):
            break
    counts['feature_count'] = len(features)
    counts['matrix_bytes'] = len(features)*len(X)*8
    counts['covered_variables'] = sorted({n.token for t, _, _ in features for _, n in t.walk() if n.token.startswith('x')})
    return features


def _sum(terms):
    if len(terms) == 1:
        return terms[0]
    middle = len(terms)//2
    left, lc = _sum(terms[:middle])
    right, rc = _sum(terms[middle:])
    return operator('add', left, right), np.r_[lc, rc]


def residual_search(base, evaluator, grammar, *, max_rounds=None, work_callback=None):
    """Fit each growing complete system once, retaining original output guards."""
    from collections import Counter
    cfg = evaluator.cfg
    report = dict(mode='training_residual_v1', events=[], outputs=[], initial_sources=[], new_evaluations=0,
        protected=0, stop_reason='deadline', initial_max_nodes=cfg.max_nodes, initial_max_depth=cfg.max_depth)
    retained = []
    if 'add' not in cfg.operators or 'mul' not in cfg.operators:
        report['stop_reason'] = 'operators'
        return retained, report
    parent = min(base, key=lambda c: (search_score(c.errors, cfg), c.complexity, digest(c.system.to_dict())))
    shapes = {c.system: complexity(c.system)['outputs'] for c in base}
    report['seed_policy'] = ('fewest_parameters_within_training_protection_envelope' if cfg.parameter_diversity else
                             'smallest_root_within_training_protection_envelope')
    libraries, selected, blocked, initial, designs = [], [], [], [], []
    for j, g in enumerate(grammar.plain):
        if evaluator.expired():
            return retained, report
        counts = Counter()
        features = feature_library(g, evaluator.X, evaluator.probes, evaluator.deadline, counts,
                                   **({'work_callback': work_callback} if work_callback is not None else {}))
        report['outputs'].append(dict(counts))
        if not features:
            report['stop_reason'] = 'no_features'
            return retained, report
        libraries.append(features)
        selected.append([])
        blocked.append(set())
        minimum = min(c.errors[j] for c in base)
        bound = minimum*1.1+1e-8
        eligible = [c for c in base if c.errors[j] <= bound]
        best = min(eligible, key=lambda c: ((shapes[c.system][j]['independent_parameters'],) if cfg.parameter_diversity else ()) +
                   (shapes[c.system][j]['expanded_nodes'], c.errors[j], c.complexity))
        report['initial_sources'].append(dict(output=j, graph_sha256=digest(best.system.to_dict()),
            constants=best.constants.tolist(), root=best.system.roots[j], training_error=float(best.errors[j]),
            minimum_training_error=float(minimum), training_error_bound=float(bound)))
        tree, values = best.system.expanded(best.system.roots[j], best.constants)
        initial.append((tree, values))
        designs.append(np.column_stack((np.ones(len(evaluator.X)), tree.evaluate(evaluator.X, values))))
    round_index = 0
    while (not evaluator.expired() and evaluator.calls < cfg.max_evaluations
           and (max_rounds is None or round_index < max_rounds)):
        if work_callback is not None:
            work_callback()
        builder = Builder()
        selections, roots, warms = [], [], []
        for j, (features, design, g) in enumerate(zip(libraries, designs, grammar.plain)):
            if evaluator.expired():
                return retained, report
            matrix = np.column_stack([f[2] for f in features])
            u, singular, _ = np.linalg.svd(design, full_matrices=False)
            q = u[:, singular > max(singular[0], 1e-12)*1e-12]
            residual = evaluator.Y[:, j]-q@(q.T@evaluator.Y[:, j])
            remaining = matrix-q@(q.T@matrix)
            norm = np.linalg.norm(remaining, axis=0)
            score = np.abs(remaining.T@residual)/np.maximum(norm, 1e-12)
            score[norm < 1e-10] = -np.inf
            if blocked[j]:
                score[list(blocked[j])] = -np.inf
            index = int(np.argmax(score))
            if not np.isfinite(score[index]) or score[index] < 1e-10:
                index = None
            selections.append(index)
            trial = design if index is None else np.column_stack((design, matrix[:, index]))
            norms = np.linalg.norm(trial, axis=0)
            norms[norms == 0] = 1.
            weights = np.linalg.lstsq(trial/norms, evaluator.Y[:, j], rcond=None)[0]/norms
            constants = {u: Node(t, units=u) for t, u in g.terminals.items() if t.startswith('c')}
            terms = [(constants[g.target], np.array([weights[0]])),
                (operator('mul', constants[ZERO], initial[j][0]), np.r_[weights[1], initial[j][1]])]
            indices = selected[j] if index is None else selected[j]+[index]
            for k, weight in zip(indices, weights[2:]):
                tree, values, _ = features[k]
                terms.append((operator('mul', constants[g.target], tree), np.r_[weight, values]))
            tree, warm = _sum(terms)
            roots.append(tree)
            warms.append(warm)
        if all(index is None for index in selections):
            report['stop_reason'] = 'no_usable_feature' if max_rounds is not None else 'residual_converged'
            break
        if cfg.structure_limits is None:
            cfg.max_nodes = max(cfg.max_nodes+16, max(t.size for t in roots))
            cfg.max_depth = max(cfg.max_depth, max(t.depth for t in roots))
            expanded = SystemGrammar(grammar.features, grammar.targets, grammar.constants, cfg, grammar.extra)
        for j, (tree, warm) in enumerate(zip(roots, warms)):
            if cfg.structure_limits is None:
                expanded.plain[j].trace(tree)
            builder.roots.append(builder.tree(tree, warm))
        graph, warm = builder.finish()
        before = evaluator.calls
        child = evaluator(graph, warm)
        pareto_mode = cfg.final_selection == 'pareto'
        relative = cfg.degradation_relative if pareto_mode else .1
        absolute = cfg.degradation_absolute if pareto_mode else 1e-8
        protected = child is not None and bool(np.all(child.errors <= parent.errors*(1+relative)+absolute))
        report['new_evaluations'] += evaluator.calls-before
        report['protected'] += int(protected)
        event = dict(round=round_index, protected=protected, new_evaluation=evaluator.calls>before,
                     feature_indices=selections, max_nodes=cfg.max_nodes, max_depth=cfg.max_depth)
        if pareto_mode:
            event.update(parent_sha256=digest(parent.system.to_dict()), graph_sha256=digest(graph.to_dict()),
                         training_nrmse=None if child is None else child.errors.tolist())
        else:
            event.update(parent=parent.to_dict(), graph=graph.to_dict(), warm=warm.tolist(),
                child=None if child is None else child.to_dict(),
                root_features=[None if k is None else libraries[j][k][0].to_dict() for j, k in enumerate(selections)])
        report['events'].append(event)
        for j, index in enumerate(selections):
            if index is not None:
                blocked[j].add(index)
        if protected:
            retained.append(child)
            if search_score(child.errors, cfg) < search_score(parent.errors, cfg):
                parent = child
            for j, index in enumerate(selections):
                if index is not None:
                    selected[j].append(index)
                    designs[j] = np.column_stack((designs[j], libraries[j][index][2]))
        round_index += 1
    if evaluator.calls >= cfg.max_evaluations:
        report['stop_reason'] = 'global_budget'
    elif max_rounds is not None:
        if evaluator.expired():
            report['stop_reason'] = 'stage_evaluation_budget'
        elif round_index >= max_rounds:
            report['stop_reason'] = 'round_limit'
        report['completed_rounds'] = round_index
    report.update(final_max_nodes=cfg.max_nodes, final_max_depth=cfg.max_depth)
    return retained, report
