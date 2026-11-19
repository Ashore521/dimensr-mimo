"""Training-only root archives, typed features and capacity-aware local proposals."""
from dataclasses import dataclass

import numpy as np

from ._single.units import ZERO, add, sub, scale
from .capacity import parameter_partition, shape
from .genetics import offspring
from .graph import Builder, Vertex, paths
from .selection import digest
from .syntax import Node


@dataclass
class Proposal:
    system: object
    warm: np.ndarray
    producer: str
    outputs: tuple = ()
    operation: str = ''
    level: str = 'screen'
    refine: bool = False
    parents: tuple = ()
    trace: object = None
    generation_cost: float = 0.
    screening_rejections: tuple = ()


def identity(candidate):
    return digest(dict(system=candidate.system.to_dict(), constants=candidate.constants.tolist()))


def capacity_rejection(system, cfg):
    """Report every breached hard limit, including concrete measured values."""
    metrics = shape(system, cfg)
    # This metric is intrinsic to the graph, independent of the chosen solver.
    metrics['nonlinear_parameters'] = len(parameter_partition(system)[1])
    limits = dict(nodes=cfg.max_nodes * len(system.roots), depth=cfg.max_depth)
    limits.update(cfg.structure_limits or {})
    reasons = []
    for key, limit in limits.items():
        observed = max(metrics[key]) if key == 'depth' else metrics[key]
        if observed > limit:
            reasons.append(dict(limit=key, observed=observed, maximum=limit))
    return reasons, metrics


def feature_library(features, target, constants, operators, limit):
    """Enumerate bounded, label-free low-order expressions by dimensional rules.

    Multiplication by an allowed coefficient supplies the missing unit. A target
    unit feature therefore uses an existing dimensionless coefficient directly.
    """
    atoms = [Node('x' + str(j), units=units) for j, units in enumerate(features)]
    terms = list(atoms)
    if 'square' in operators:
        terms.extend(Node('square', (node,), scale(node.units, 2)) for node in atoms)
    result, seen = [], set()

    def accept(node):
        for k, units in enumerate(constants):
            if add(units, node.units) != target or 'mul' not in operators:
                continue
            term = Node('mul', (Node('c' + str(k), units=units), node), target)
            if term not in seen and len(result) < limit:
                seen.add(term)
                result.append(term)
        if node.units == target and node not in seen and len(result) < limit:
            seen.add(node)
            result.append(node)

    for node in terms:
        accept(node)
    # Input denominators and their squares cover rational dimensions without
    # adding task-specific formula templates or inventing constant unit types.
    examined = 0
    for left in atoms:
        for right in terms:
            for op, units in (('mul', add(left.units, right.units)),
                              ('div', sub(left.units, right.units))):
                if op in operators:
                    accept(Node(op, (left, right), units))
            examined += 1
            if len(result) >= limit or examined >= 32 * limit:
                return result
    for node in atoms:
        if node.units == ZERO:
            for op in ('sin', 'cos', 'exp', 'log'):
                if op in operators:
                    accept(Node(op, (node,), ZERO))
    for k, units in enumerate(constants):
        if units == target:
            node = Node('c' + str(k), units=units)
            if node not in seen and len(result) < limit:
                result.append(node)
    return result


def assemble_roots(selections):
    """Namespace independent candidates, retaining explicit sharing per source."""
    builder, nodes, parameters = Builder(), {}, {}

    def visit(candidate, source, index):
        key = source, index
        if key not in nodes:
            node = candidate.system.nodes[index]
            children = tuple(visit(candidate, source, child) for child in node.children)
            parameter = None
            if node.parameter is not None:
                parameter_key = source, node.parameter
                if parameter_key not in parameters:
                    parameters[parameter_key] = len(builder.constants)
                    builder.constants.append(float(candidate.constants[node.parameter]))
                parameter = parameters[parameter_key]
            builder.nodes.append(Vertex(node.token, children, node.units, parameter))
            nodes[key] = len(builder.nodes) - 1
        return nodes[key]

    for candidate, output in selections:
        builder.roots.append(visit(candidate, identity(candidate), candidate.system.roots[output]))
    return builder.finish()


def replace_term(candidate, output, tree, path=(), *, additive=False):
    """Preserve every retained binding, assign a new namespace to inserted terms."""
    builder = Builder(candidate.system, candidate.constants)
    replacement = builder.tree(tree)
    root = candidate.system.roots[output]
    if additive:
        builder.nodes.append(Vertex('add', (root, replacement), tree.units))
        builder.roots[output] = len(builder.nodes) - 1
    else:
        builder.roots[output] = builder.replace(root, path, replacement)
    return builder.finish()


class RootArchive:
    def __init__(self, outputs, capacity):
        self.roots = [[] for _ in range(outputs)]
        self.capacity, self.version = capacity, 0

    def add(self, candidate):
        changed = False
        for output, entries in enumerate(self.roots):
            key = identity(candidate)
            if any(identity(item) == key for item in entries):
                continue
            previous = [identity(item) for item in entries]
            values = entries + [candidate]
            # Keep the accuracy champion plus complexity-diverse Pareto roots.
            values.sort(key=lambda item: (float(item.errors[output]), item.complexity, identity(item)))
            frontier = []
            best_complexity = float('inf')
            for item in values:
                if item.complexity < best_complexity:
                    frontier.append(item)
                    best_complexity = item.complexity
            remaining = [item for item in values if all(identity(item) != identity(c) for c in frontier)]
            self.roots[output] = (frontier + remaining)[:self.capacity]
            changed |= previous != [identity(item) for item in self.roots[output]]
        self.version += int(changed)
        return changed

    def best_errors(self):
        return np.asarray([min((float(c.errors[j]) for c in entries), default=float('inf'))
                           for j, entries in enumerate(self.roots)])


class Producers:
    def __init__(self, grammar, controls):
        self.grammar, self.controls = grammar, controls
        self.features = [feature_library(grammar.features, target, grammar.constants,
                         grammar.cfg.operators, controls['feature_limit']) for target in grammar.targets]

    def initial(self, index, rng):
        builder = Builder()
        for output, features in enumerate(self.features):
            if features:
                tree = features[index % len(features)]
            else:
                tree, _ = self.grammar.plain[output].generate(rng)
            builder.roots.append(builder.tree(tree))
        return Proposal(*builder.finish(), 'features', tuple(range(len(self.features))), 'typed_seed')

    def propose(self, name, context):
        rng, archive = context['rng'], context['roots']
        pool, cfg = context['pool'], self.grammar.cfg
        if name == 'grammar':
            system, warm, trace = self.grammar.generate(rng, context['policy'])
            return Proposal(system, warm, name, operation='grammar_sample', trace=trace)
        if not pool:
            if name == 'features':
                return self.initial(context['index'], rng)
            return None
        if name == 'gp':
            if not cfg.use_gp:
                return None
            parent, donor = [pool[int(rng.integers(len(pool)))] for _ in range(2)]
            system, warm, operation = offspring(parent, donor, self.grammar, rng)
            return Proposal(system, warm, name, operation=operation,
                            parents=(identity(parent), identity(donor)))
        if name == 'roots':
            if any(not entries for entries in archive.roots):
                return None
            selections = [(entries[int(rng.integers(len(entries)))], j)
                          for j, entries in enumerate(archive.roots)]
            return Proposal(*assemble_roots(selections), name,
                            tuple(range(len(selections))), 'root_assembly',
                            parents=tuple(identity(c) for c, _ in selections))
        return self.residual(context)

    def residual(self, context):
        """Rank typed terms on training residuals, retain round-robin exploration."""
        archive, index = context['roots'], context['index']
        errors = archive.best_errors()
        exploration_index = index // 3
        output = int(np.argmax(errors)) if index % 3 else exploration_index % len(errors)
        features = self.features[output]
        if not features or not archive.roots[output]:
            return None
        parent = archive.roots[output][0]
        # The complete parent retains cross-output sharing during local edits.
        if index % 3:
            prediction = parent.system.evaluate(context['X'], parent.constants, stats=context['stats'])
            residual = context['Y'][:, output] - prediction[:, output]
            ranking = []
            for k, tree in enumerate(features):
                with np.errstate(all='ignore'):
                    values = tree.evaluate(context['X'], np.ones(tree.n_constants))
                context['stats']['portfolio_feature_predictions'] += 1
                context['stats']['portfolio_feature_sample_node_values'] += len(values) * tree.size
                if np.isfinite(values).all():
                    with np.errstate(all='ignore'):
                        norm = float(np.linalg.norm(values))
                        score = abs(float(np.dot(values, residual))) / max(norm, 1e-15)
                    if np.isfinite(score):
                        ranking.append((-score, k))
            if not ranking:
                return None
            ranking.sort()
            tree = features[ranking[index % min(4, len(ranking))][1]]
        else:
            tree = features[(exploration_index // len(errors)) % len(features)]
        operation = 'residual_add'
        rejected = []
        if 'add' in self.grammar.cfg.operators:
            system, warm = replace_term(parent, output, tree, additive=True)
            reasons, metrics = capacity_rejection(system, self.grammar.cfg)
            if not reasons:
                return Proposal(system, warm, 'features', (output,), operation,
                                parents=(identity(parent),))
            rejected.append(dict(operation=operation, reasons=reasons, shape=metrics))
        # Capacity fallback replaces an equal-unit term or the entire root.
        locations = [(path, node) for path, node in paths(parent.system, parent.system.roots[output])
                     if parent.system.nodes[node].units == tree.units]
        locations.sort(key=lambda item: (len(item[0]), item[0]))
        for path, _ in locations:
            system, warm = replace_term(parent, output, tree, path)
            reasons, metrics = capacity_rejection(system, self.grammar.cfg)
            if not reasons:
                return Proposal(system, warm, 'features', (output,), 'residual_replace',
                                parents=(identity(parent),), screening_rejections=tuple(rejected))
            rejected.append(dict(operation='residual_replace', path=list(path),
                                 reasons=reasons, shape=metrics))
        # Return a concrete rejected replacement so its hard limit is recorded.
        return Proposal(*replace_term(parent, output, tree), 'features', (output,),
                        'residual_replace', parents=(identity(parent),),
                        screening_rejections=tuple(rejected))
