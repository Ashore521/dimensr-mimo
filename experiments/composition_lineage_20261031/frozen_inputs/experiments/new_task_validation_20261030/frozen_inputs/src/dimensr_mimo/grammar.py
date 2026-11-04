"""Finite typed grammar with exact completion costs and explicit search bounds."""
from dataclasses import dataclass, field
from fractions import Fraction
from math import lcm
import numpy as np
from .syntax import ARITY, Node
from ._single.units import ZERO, add, sub, scale

@dataclass(frozen=True)
class Production:
    token: str
    children: tuple = ()


@dataclass
class Trace:
    tokens: list
    parents: list
    contexts: list
    masks: list
    production_features: list = field(default_factory=list)
    production_indices: list = field(default_factory=list)


class Grammar:
    def __init__(self, feature_units, target_units, constant_units, operators,
                 max_nodes=15, max_depth=6, extra_units=(), *, reference_units=(),
                 max_references=12, output_context=(), reference_shapes=()):
        self.target = target_units
        self.max_nodes = max_nodes
        self.max_depth = max_depth
        self.output_context = list(output_context)
        self.terminals = {f"x{i}": u for i, u in enumerate(feature_units)}
        self.terminals.update({f"c{i}": u for i, u in enumerate(constant_units)})
        # Reference slots keep vocabulary stable; unavailable slots are masked.
        self.tokens = tuple(self.terminals) + tuple(f'r{i}' for i in range(max_references)) + tuple(operators)
        self.terminals.update({f'r{i}': u for i, u in enumerate(reference_units)})
        if reference_shapes and (len(reference_shapes)!=len(reference_units) or
                any(len(s)!=2 or any(type(v) is not int or v<1 for v in s) for s in reference_shapes)):
            raise ValueError('Reference shapes must provide positive integer size/depth pairs.')
        self.terminal_shapes={t:(1,1) for t in self.terminals}
        self.terminal_shapes.update({f'r{i}':tuple(s) for i,s in enumerate(reference_shapes)})
        self.index = {t: i for i, t in enumerate(self.tokens)}
        self.bos = len(self.tokens)
        base = set(self.terminals.values()) | {ZERO, target_units} | set(extra_units)
        pool = set(base)
        for a in base:
            pool.update(scale(a, k) for k in (-2, -1, 2, Fraction(1, 2)))
            for b in base:
                pool.update((add(a, b), sub(a, b)))
        if len(pool) > 256:
            raise ValueError("More than 256 unit types; reduce features or unit types for this bounded grammar.")
        self.units = tuple(sorted(pool))
        # A common integer lattice preserves exact Fraction equality while the
        # quadratic binary-production loop avoids repeated Fraction arithmetic.
        denominator = lcm(*(v.denominator for u in self.units for v in u))
        lattice = [tuple(int(v * denominator) for v in u) for u in self.units]
        by_lattice = dict(zip(lattice, self.units))
        self.productions = {u: [] for u in self.units}
        for t, u in self.terminals.items():
            self.productions[u].append(Production(t))
        for t in operators:
            for a, ia in zip(self.units, lattice):
                if ARITY[t] == 2:
                    if t in ("add", "sub"):
                        self.productions[a].append(Production(t, (a, a)))
                        continue
                    for b, ib in zip(self.units, lattice):
                        key = tuple(x + y for x, y in zip(ia, ib)) if t == "mul" else tuple(x - y for x, y in zip(ia, ib))
                        out = by_lattice.get(key)
                        if out is not None:
                            self.productions[out].append(Production(t, (a, b)))
                else:
                    if t in ("sin", "cos", "log", "exp"):
                        if a != ZERO:
                            continue
                        out = ZERO
                    else:
                        out = scale(a, {"square": 2, "sqrt": Fraction(1, 2), "neg": 1}[t])
                    if out in pool:
                        self.productions[out].append(Production(t, (a,)))
        # cost[d][u] = minimum nodes for a tree of units u and depth <= d.
        impossible = max_nodes + 1
        self.cost = [{u: impossible for u in self.units}]
        for depth in range(1, max_depth + 1):
            self.cost.append({u: min([impossible] + [self.production_cost(p,depth) for p in ps])
                              for u, ps in self.productions.items()})
        if self.cost[-1][target_units] > max_nodes:
            raise ValueError("Target units cannot be generated within the grammar bounds. "
                             "Supply suitable constant_units/extra_units or increase max_nodes/max_depth.")
        self._options = {}
        self._production_features = {}

    def production_cost(self,production,depth):
        if not production.children:
            size,height=self.terminal_shapes[production.token]
            return size if height<=depth else self.max_nodes+1
        return 1+sum(self.cost[depth-1][u] for u in production.children)

    def options(self, units, budget, depth):
        key = (units, budget, depth)
        if key not in self._options:
            by_token = {}
            if depth > 0:
                for p in self.productions.get(units, ()):
                    if self.production_cost(p,depth) <= budget:
                        by_token.setdefault(self.index[p.token], []).append(p)
            self._options[key] = by_token
        return self._options[key]

    def generate(self, rng, policy=None, units=None, budget=None, depth=None):
        units = self.target if units is None else units
        budget = self.max_nodes if budget is None else budget
        depth = self.max_depth if depth is None else depth
        trace = Trace([], [], [], [])
        hidden = None

        def visit(required, remaining, height, parent):
            nonlocal hidden
            choices = self.options(required, remaining, height)
            if not choices:
                raise ValueError("No feasible production for subtree.")
            mask = np.zeros(len(self.tokens), dtype=bool)
            mask[list(choices)] = True
            context = [float(x) for x in required] + [remaining / self.max_nodes, height / self.max_depth] + self.output_context
            previous = trace.tokens[-1] if trace.tokens else self.bos
            if policy is None:
                token = int(rng.choice(list(choices)))
            else:
                token, hidden = policy.sample_step(previous, parent, context, mask, hidden)
            features = self.production_features(choices[token], height)
            choice = (policy.sample_production(hidden, features, token) if policy is not None else
                      int(rng.integers(len(choices[token]))))
            p = choices[token][choice]
            trace.tokens.append(token)
            trace.parents.append(parent)
            trace.contexts.append(context)
            trace.masks.append(mask)
            trace.production_features.append(features)
            trace.production_indices.append(choice)
            if not p.children:
                return Node(p.token,units=required),self.terminal_shapes[p.token][0]
            children = []
            remaining -= 1
            used = 1
            for i, child_units in enumerate(p.children):
                reserved = sum(self.cost[height - 1][u] for u in p.children[i + 1:])
                child,size = visit(child_units, remaining - reserved, height - 1, token)
                remaining -= size
                used += size
                children.append(child)
            return Node(p.token, tuple(children), required),used

        tree,_=visit(units, budget, depth, self.bos)
        return tree,trace

    def production_features(self, productions, depth):
        key = tuple(productions), depth
        if key not in self._production_features:
            features = np.zeros((len(productions), 16), dtype=np.float32)
            for i, production in enumerate(productions):
                for j, dims in enumerate(production.children):
                    features[i, j * 7:(j + 1) * 7] = [float(u) for u in dims]
                    features[i, 14 + j] = self.cost[depth - 1][dims] / self.max_nodes
            self._production_features[key] = features
        # Traces remain writable without letting a caller corrupt cached values.
        return self._production_features[key].copy()

    def trace(self, tree):
        result = Trace([], [], [], [])

        def visit(node, budget, depth, parent):
            choices = self.options(node.units, budget, depth)
            token = self.index[node.token]
            p = Production(node.token, tuple(c.units for c in node.children))
            if p not in choices.get(token, ()):
                raise ValueError("Tree cannot be replayed under this grammar.")
            mask = np.zeros(len(self.tokens), dtype=bool)
            mask[list(choices)] = True
            result.tokens.append(token)
            result.parents.append(parent)
            result.contexts.append([float(u) for u in node.units] +
                                   [budget / self.max_nodes, depth / self.max_depth] + self.output_context)
            result.masks.append(mask)
            result.production_features.append(self.production_features(choices[token], depth))
            result.production_indices.append(choices[token].index(p))
            if not node.children:
                return self.terminal_shapes[node.token][0]
            remaining = budget - 1
            used = 1
            for i, child in enumerate(node.children):
                reserved = sum(self.cost[depth - 1][c.units] for c in node.children[i + 1:])
                size=visit(child, remaining - reserved, depth - 1, token)
                remaining -= size
                used += size
            return used
        visit(tree, self.max_nodes, self.max_depth, self.bos)
        return result
