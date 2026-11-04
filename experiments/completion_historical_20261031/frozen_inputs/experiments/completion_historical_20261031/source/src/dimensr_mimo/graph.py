"""Explicit multi-root DAG; raw operator semantics adapted from DimenSR expression.py."""
from dataclasses import dataclass
import numpy as np
from ._single.expression import Node, ARITY
from ._single.units import ZERO, unit, encode
from ._single.intervals import audit_box


@dataclass(frozen=True)
class Vertex:
    token: str
    children: tuple = ()
    units: tuple = ZERO
    parameter: int | None = None


@dataclass(frozen=True)
class System:
    nodes: tuple
    roots: tuple

    def __post_init__(self):
        if not self.roots or any(type(r) is not int or not 0 <= r < len(self.nodes) for r in self.roots):
            raise ValueError("System requires valid output roots.")
        parameter_units = {}
        for i, v in enumerate(self.nodes):
            if any(type(c) is not int or not 0 <= c < i for c in v.children):
                raise ValueError("References must be topological (no cycles or forward references).")
            Node(v.token, tuple(Node('c0', units=self.nodes[c].units) for c in v.children), v.units)
            if v.token.startswith('c') and not v.children:
                if type(v.parameter) is not int or v.parameter < 0:
                    raise ValueError("Constants require explicit nonnegative parameter IDs.")
                if v.parameter in parameter_units and parameter_units[v.parameter] != v.units:
                    raise ValueError("Bound parameters must have identical units.")
                parameter_units[v.parameter] = v.units
            elif v.parameter is not None:
                raise ValueError("Only constants can have parameter IDs.")
        if set(parameter_units) != set(range(len(parameter_units))):
            raise ValueError("Parameter IDs must be contiguous.")
        if len(self.reachable()) != len(self.nodes):
            raise ValueError("Unused nodes are not allowed.")

    def reachable(self, root=None):
        seen = set()
        def visit(i):
            if i not in seen:
                seen.add(i)
                for c in self.nodes[i].children:
                    visit(c)
        for r in self.roots if root is None else (root,):
            visit(r)
        return seen

    @property
    def n_parameters(self):
        return len({v.parameter for v in self.nodes if v.parameter is not None})

    @property
    def size(self):
        return sum(v.parameter is None for v in self.nodes) + self.n_parameters

    @property
    def n_operator_types(self):
        return len({v.token for v in self.nodes if v.children})

    @property
    def sharing(self):
        reach = [self.reachable(r) for r in self.roots]
        node_uses = {i: sum(i in s for s in reach) for i in range(len(self.nodes))}
        params = [{self.nodes[i].parameter for i in s if self.nodes[i].parameter is not None} for s in reach]
        return {'shared_nodes': sum(k > 1 for k in node_uses.values()),
                'shared_parameters': sum(sum(p in s for s in params) > 1 for p in range(self.n_parameters)),
                'unique_parameters': self.n_parameters, 'dag_size': self.size}

    def evaluate(self, X, constants):
        X, constants = np.asarray(X, float), np.asarray(constants, float)
        if X.ndim != 2 or constants.shape != (self.n_parameters,):
            raise ValueError("Invalid input or parameter shape.")
        values = []
        with np.errstate(all='ignore'):
            for v in self.nodes:
                if v.parameter is not None:
                    value = np.full(len(X), constants[v.parameter])
                elif v.token.startswith('x'):
                    value = X[:, int(v.token[1:])]
                else:
                    args = [values[c] for c in v.children]
                    f = {'add': np.add, 'sub': np.subtract, 'mul': np.multiply,
                         'div': np.divide, 'neg': np.negative, 'square': np.square,
                         'sqrt': np.sqrt, 'sin': np.sin, 'cos': np.cos,
                         'exp': np.exp, 'log': np.log}[v.token]
                    value = f(*args)
                    # An outer finite value must not hide an invalid intermediate.
                    valid = np.logical_and.reduce([np.isfinite(a) for a in args])
                    value = np.where(valid & np.isfinite(value), value, np.nan)
                values.append(value)
        return np.column_stack([values[r] for r in self.roots])

    def expanded(self, root, constants):
        """Audit/display expansion, never simplify or infer parameter equality."""
        def visit(i):
            v = self.nodes[i]
            parts = [visit(c) for c in v.children]
            tree = Node(v.token, tuple(t for t, _ in parts), v.units)
            c = ([constants[v.parameter]] if v.parameter is not None else
                 [x for _, values in parts for x in values])
            return tree, c
        tree, c = visit(root)
        return tree, np.asarray(c, float)

    def audit(self, constants, bounds, max_boxes=32, output_bounds=None):
        return [audit_box(*self.expanded(r, constants), bounds, max_boxes=max_boxes,
                          output_bounds=output_bounds) for r in self.roots]

    def formulas(self, constants, names=None):
        return [str(t.sympy(c, names)) for t, c in (self.expanded(r, constants) for r in self.roots)]

    def to_dict(self):
        return {'nodes': [dict(token=v.token, children=list(v.children), units=encode(v.units),
                               parameter=v.parameter) for v in self.nodes], 'roots': list(self.roots)}

    @classmethod
    def from_dict(cls, d):
        return cls(tuple(Vertex(v['token'], tuple(v['children']), unit(v['units']), v['parameter'])
                         for v in d['nodes']), tuple(d['roots']))


class Builder:
    def __init__(self, system=None, constants=None):
        self.nodes = list(system.nodes) if system else []
        self.roots = list(system.roots) if system else []
        self.constants = list(constants) if constants is not None else []

    def tree(self, tree, constants=None, references=()):
        values = iter(np.ones(tree.n_constants) if constants is None else constants)
        def visit(t):
            if t.token.startswith('r'):
                return references[int(t.token[1:])]
            children = tuple(visit(c) for c in t.children)
            p = None
            if t.token.startswith('c') and not t.children:
                p = len(self.constants)
                self.constants.append(float(next(values)))
            self.nodes.append(Vertex(t.token, children, t.units, p))
            return len(self.nodes) - 1
        return visit(tree)

    def replace(self, root, path, replacement):
        if not path:
            if self.nodes[root].units != self.nodes[replacement].units:
                raise ValueError('Replacement must preserve units.')
            return replacement
        v = self.nodes[root]
        children = list(v.children)
        children[path[0]] = self.replace(children[path[0]], path[1:], replacement)
        self.nodes.append(Vertex(v.token, tuple(children), v.units, v.parameter))
        return len(self.nodes) - 1

    def finish(self):
        """Prune, topologically renumber and alpha-rename parameters; no merging."""
        nodes, ids, params, values = [], {}, {}, []
        def visit(i):
            if i not in ids:
                v = self.nodes[i]
                children = tuple(visit(c) for c in v.children)
                p = None
                if v.parameter is not None:
                    if v.parameter not in params:
                        params[v.parameter] = len(values)
                        values.append(self.constants[v.parameter])
                    p = params[v.parameter]
                nodes.append(Vertex(v.token, children, v.units, p))
                ids[i] = len(nodes) - 1
            return ids[i]
        roots = tuple(visit(r) for r in self.roots)
        return System(tuple(nodes), roots), np.asarray(values, float)


def paths(system, root, prefix=()):
    yield prefix, root
    for j, child in enumerate(system.nodes[root].children):
        yield from paths(system, child, prefix + (j,))


def in_bounds(system, cfg):
    sizes, depths = [], []
    for v in system.nodes:
        sizes.append(min(cfg.max_nodes + 1, 1 + sum(sizes[c] for c in v.children)))
        depths.append(1 + max((depths[c] for c in v.children), default=0))
    return all(sizes[r] <= cfg.max_nodes and depths[r] <= cfg.max_depth for r in system.roots)
