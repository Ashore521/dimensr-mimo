"""Typed sparse supports and compact multi-output recombination for the portfolio."""
from fractions import Fraction
from functools import lru_cache

import numpy as np

from ._single.units import ZERO, add, scale, sub
from .capacity import compact
from .graph import Builder
from .portfolio_proposals import Proposal, RootArchive, assemble_roots, capacity_rejection, identity
from .syntax import Node


def _units(op, a, b=None):
    if op in ('add', 'sub'):
        return a if a == b else None
    if op == 'mul':
        return add(a, b)
    if op == 'div':
        return sub(a, b)
    if op == 'square':
        return scale(a, 2)
    if op == 'sqrt':
        return scale(a, Fraction(1, 2))
    return ZERO if a == ZERO else None


def library(grammar, indices, limit):
    """Round-robin operator families; no fitted coefficients in basis screening."""
    atoms = [Node(f'x{i}', units=grammar.features[i]) for i in indices]
    ops = set(grammar.cfg.operators)
    unary = [op for op in ('square', 'sin', 'cos', 'log', 'exp', 'sqrt') if op in ops]
    binary = [op for op in ('mul', 'div', 'add', 'sub') if op in ops]
    def transform(op, *terms):
        units = _units(op, *(t.units for t in terms))
        return None if units is None else Node(op, terms, units)
    simple = [t for x in atoms for op in unary if (t := transform(op, x)) is not None]
    pairs = [t for i, x in enumerate(atoms) for j, y in enumerate(atoms) for op in binary
             if i != j and (op not in ('mul', 'add') or i < j)
             and (t := transform(op, x, y)) is not None]
    families = [iter(atoms), iter(simple), iter(pairs)]
    # Use only bounded frontiers even for many-input callers.
    mixed = (transform(op, x, y) for y in simple for x in atoms for op in binary if op in ('mul', 'div'))
    reverse = (transform('div', y, x) for y in simple for x in atoms if 'div' in ops)
    nested = (transform(op, p) for p in pairs for op in unary)
    cubic = (transform('mul', x, p) for p in pairs if p.token == 'mul'
             for x in atoms if 'mul' in ops)
    # A learnable inner translation or scale can expose nonlinear shape. Its
    # starting value is fixed at one; the native fitter estimates all constants.
    warped = []
    for x in atoms:
        for j, u in enumerate(grammar.constants):
            c = Node(f'c{j}', units=u)
            for op in ('add', 'mul'):
                if op not in ops:
                    continue
                v = transform(op, x, c)
                if v is not None:
                    warped.extend(transform(f, v) for f in unary if f not in ('square', 'sqrt'))
                    if op == 'add' and 'div' in ops:
                        warped.extend(transform('div', y, v) for y in atoms)
    families.extend(map(iter, (mixed, reverse, nested, cubic, warped)))
    seen = {}
    while families and len(seen) < limit:
        active = []
        for family in families:
            for tree in family:
                if tree is not None and tree not in seen:
                    seen[tree] = None
                    active.append(family)
                    break
            if len(seen) == limit:
                break
        families = active
    return list(seen)


@lru_cache(maxsize=1024)
def root_shape(graph, output):
    # Alpha-rename local parameters while preserving repeated parameter uses.
    local, parameters, nodes = {}, {}, []
    def visit(i):
        if i not in local:
            v = graph.nodes[i]
            children = tuple(visit(k) for k in v.children)
            p = None if v.parameter is None else parameters.setdefault(v.parameter, len(parameters))
            nodes.append((v.token, v.units, children, p))
            local[i] = len(nodes) - 1
        return local[i]
    visit(graph.roots[output])
    return tuple(nodes)


class StructuralRootArchive(RootArchive):
    def add(self, candidate):
        changed = False
        for j, entries in enumerate(self.roots):
            shape = root_shape(candidate.system, j)
            values = list(entries)
            for k, old in enumerate(values):
                if root_shape(old.system, j) == shape:
                    if old.errors[j] <= candidate.errors[j]:
                        break
                    values[k] = candidate
                    break
            else:
                values.append(candidate)
            values.sort(key=lambda c: (float(c.errors[j]), len(root_shape(c.system, j)), c.complexity))
            frontier, rest = [], []
            smallest = float('inf')
            for item in values:
                size = len(root_shape(item.system, j))
                if size < smallest:
                    frontier.append(item)
                    smallest = size
                else:
                    rest.append(item)
            keep = (frontier + rest)[:self.capacity]
            changed |= tuple(map(id, keep)) != tuple(map(id, entries))
            self.roots[j] = keep
        self.version += int(changed)
        return changed


def recombine(archive, cfg, cache):
    if not all(archive.roots):
        return None
    beam = [((), 0., 0., 0)]
    for j, entries in enumerate(archive.roots):
        choices = [(c, float(c.errors[j]), len(root_shape(c.system, j))) for c in entries]
        expanded = [(parents + (c,), max(worst, err), total + err, size + n)
                    for parents, worst, total, size in beam for c, err, n in choices]
        # Reserve room for both accurate assemblies and compact partial programs.
        by_error = sorted(expanded, key=lambda v: v[1:])
        by_size = sorted(expanded, key=lambda v: (v[3], v[1], v[2]))
        beam = by_error[:12] + by_size[:6]
    tried = set()
    for parents, *_ in sorted(beam, key=lambda v: v[1:]):
        key = tuple(map(id, parents))
        if key in tried or len(set(key)) == 1:
            continue
        tried.add(key)
        graph, warm = compact(*assemble_roots([(c, j) for j, c in enumerate(parents)]))
        if graph not in cache and not capacity_rejection(graph, cfg)[0]:
            return Proposal(graph, warm, 'roots', operation='compact_root_assembly',
                            parents=tuple(map(identity, parents)))
    return None


class SupportSearch:
    """Rank structures by training correlations. All parameter fits go through fit_request."""
    def __init__(self, grammar, X, Y, stats):
        self.grammar = grammar
        sample = np.linspace(0, len(X) - 1, min(256, len(X)), dtype=int)
        x, self.y = X[sample], Y[sample]
        y0 = self.y - self.y.mean(axis=0)
        y0 = y0 / np.maximum(np.linalg.norm(y0, axis=0), 1e-15)
        x0 = x - x.mean(axis=0)
        association = np.max(np.abs((x0 / np.maximum(np.linalg.norm(x0, axis=0), 1e-15)).T @ y0), axis=1)
        indices = np.argsort(-association, kind='stable')[:12].tolist()
        self.trees, columns = [], []
        for tree in library(grammar, indices, min(384, max(48, 1_048_576 // max(1, len(x) * 8)))):
            v = tree.evaluate(x, np.ones(tree.n_constants))
            stats['portfolio_feature_predictions'] += 1
            stats['portfolio_feature_sample_node_values'] += len(x) * tree.size
            if np.isfinite(v).all() and np.max(np.abs(v)) < 1e100 and np.ptp(v) > 1e-14:
                self.trees.append(tree)
                columns.append(v)
        self.columns = np.column_stack(columns) if columns else np.empty((len(x), 0))
        self.coefs = {u: Node(f'c{i}', units=u) for i, u in enumerate(grammar.constants)}
        self.supports = {}

    def ranked_terms(self, output, offset, alternate, stats):
        key = (output, offset, alternate)
        if key in self.supports:
            return self.supports[key]
        goal = self.grammar.targets[output]
        ops = self.grammar.cfg.operators
        eligible, weighted = [], []
        for i, term in enumerate(self.trees):
            coef = self.coefs.get(sub(goal, term.units))
            if coef is not None and 'mul' in ops:
                weighted.append(Node('mul', (coef, term), goal))
            elif term.units == goal:
                weighted.append(term)
            else:
                continue
            eligible.append(i)
        y = self.y[:, output]
        A = self.columns[:, eligible].copy()
        if offset:
            A -= A.mean(axis=0)
            y = y - y.mean()
        norms = np.linalg.norm(A, axis=0)
        A /= np.maximum(norms, 1e-15)
        chosen = []
        sizes = np.asarray([t.size for t in weighted])
        for step in range(min(4, len(eligible))):
            lengths = np.linalg.norm(A, axis=0)
            usable = lengths > 1e-7
            if not usable.any():
                break
            scores = np.abs(y @ A) / np.maximum(lengths, 1e-15)
            # Mild size preference suppresses near-collinear elaborate terms.
            scores /= 1. + .012 * sizes
            scores[~usable] = -np.inf
            stats['structural_direction_scores'] += int(usable.sum())
            order = np.argsort(-scores, kind='stable')
            idx = order[min(alternate if step == 0 else 0, int(usable.sum()) - 1)]
            chosen.append(weighted[idx])
            direction = A[:, idx] / lengths[idx]
            A -= np.outer(direction, direction @ A)
        self.supports[key] = chosen
        return chosen

    def propose(self, index, cache, stats):
        # Alternate intercept/no-intercept and progressively richer supports.
        # Rotating the first atom explores alternative paths through correlated features.
        count = (index // 2) % 4 + 1
        alternate = index // 8
        g = self.grammar
        builder = Builder()
        for j, target in enumerate(g.targets):
            offset = index % 2 == 1 and target in self.coefs and 'add' in g.cfg.operators
            terms = self.ranked_terms(j, offset, alternate, stats)[:count]
            if 'add' not in g.cfg.operators:
                terms = terms[:1]
            if offset:
                terms = terms + [self.coefs[target]]
            if not terms:
                return None
            while len(terms) > 1:
                terms = [Node('add', (terms[k], terms[k+1]), target) if k+1 < len(terms) else terms[k]
                         for k in range(0, len(terms), 2)]
            builder.roots.append(builder.tree(terms[0]))
        graph, warm = compact(*builder.finish())
        if graph in cache or capacity_rejection(graph, g.cfg)[0]:
            return None
        return Proposal(graph, warm, 'features', operation='sparse_support', level='standard')
