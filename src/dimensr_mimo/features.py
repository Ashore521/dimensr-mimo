"""Typed basis functions, with value-aware sharing and strict raw-tree domains."""
from dataclasses import dataclass
from fractions import Fraction
import numpy as np
from ._vendor.expression import Node
from ._vendor.units import ZERO, add, sub, scale


@dataclass(frozen=True)
class Feature:
    tree: Node
    constants: tuple = ()

    def to_dict(self):
        return {"tree": self.tree.to_dict(), "constants": list(self.constants)}


def compose(token, *features):
    children = tuple(f.tree for f in features)
    dims = [n.units for n in children]
    out = (add(*dims) if token == "mul" else sub(*dims) if token == "div" else
           scale(dims[0], 2) if token == "square" else scale(dims[0], Fraction(1, 2))
           if token == "sqrt" else ZERO if token in ("sin", "cos", "exp", "log") else dims[0])
    tree = Node(token, children, out)
    constants = tuple(v for f in features for v in f.constants)
    tree, constants = tree.canonical(constants)
    return Feature(tree, tuple(constants))


def graph_keys(feature):
    """Equal constants share a node; equal token labels alone do not."""
    values = iter(feature.constants)

    def visit(node):
        children = tuple(visit(c) for c in node.children)
        return (node.token, node.units, next(values) if not children and node.token.startswith("c") else None, children)
    return visit(feature.tree)


def graph_size(features):
    seen = set()

    def visit(key):
        if key in seen:
            return
        seen.add(key)
        for child in key[3]:
            visit(child)
    for feature in features:
        visit(graph_keys(feature))
    return len(seen)


def evaluate_features(features, X):
    """Evaluate every shared node once, propagating illegal intermediates."""
    cache = {}
    unary = {"neg": np.negative, "square": np.square, "sqrt": np.sqrt, "sin": np.sin,
             "cos": np.cos, "exp": np.exp, "log": np.log}
    binary = {"add": np.add, "sub": np.subtract, "mul": np.multiply, "div": np.divide}

    def visit(key):
        if key not in cache:
            token, _, value, children = key
            if not children:
                result = X[:, int(token[1:])] if token.startswith("x") else np.full(len(X), value)
            else:
                args = [visit(c) for c in children]
                result = (binary[token] if len(args) == 2 else unary[token])(*args)
                result = np.where(np.logical_and.reduce([np.isfinite(a) for a in args]), result, np.nan)
            cache[key] = np.where(np.isfinite(result), result, np.nan)
        return cache[key]
    with np.errstate(all="ignore"):
        columns = [visit(graph_keys(f)) for f in features]
    return np.column_stack(columns) if columns else np.empty((len(X), 0))
