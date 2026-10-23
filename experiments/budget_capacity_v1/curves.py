"""Exact validation-only selection under shared-DAG complexity caps."""
from itertools import product

import numpy as np

from dimensr_mimo._vendor.expression import Node
from dimensr_mimo.features import Feature, graph_keys
from dimensr_mimo.search import predict_candidate


def decode(saved):
    return dict(saved,
        features=tuple(Feature(Node.from_dict(f["tree"]), tuple(f["constants"])) for f in saved["features"]),
        coefficients=np.asarray(saved["coefficients"], dtype=float),
        intercepts=np.asarray(saved["intercepts"], dtype=float))


def node_keys(features):
    keys = set()
    def visit(key):
        if key not in keys:
            keys.add(key)
            for child in key[3]:
                visit(child)
    for feature in features:
        visit(graph_keys(feature))
    return keys


def validation_frontier(archives):
    """Return every change in the validation-optimal model as a K cap increases.

    No outer labels or scores are accepted by this function. Enumerating output
    combinations preserves cross-output exact-node sharing in the K constraint.
    """
    if not archives or any(not candidates for candidates in archives):
        raise ValueError("Each output system needs a nonempty valid archive.")
    nodes, encoded = {}, []
    for candidates in archives:
        options = []
        for saved in candidates:
            candidate = decode(saved)
            mask = 0
            for key in node_keys(candidate["features"]):
                index = nodes.setdefault(key, len(nodes))
                mask |= 1 << index
            edges = int(np.count_nonzero(candidate["coefficients"]) + np.count_nonzero(candidate["intercepts"]))
            loss = float(saved["validation_loss"])
            if not np.isfinite(loss):
                raise ValueError("Nonfinite inner validation score.")
            options.append((mask, edges, loss))
        encoded.append(options)
    by_size, count = {}, 0
    for indices in product(*(range(len(a)) for a in encoded)):
        mask, edges, loss = 0, 0, 0.
        for output, index in enumerate(indices):
            bits, n_edges, score = encoded[output][index]
            mask |= bits
            edges += n_edges
            loss += score / len(encoded)
        size = mask.bit_count() + edges
        key = (loss, indices)
        if size not in by_size or key < by_size[size]:
            by_size[size] = key
        count += 1
    frontier, best = [], None
    for size, (loss, indices) in sorted(by_size.items()):
        if best is None or loss < best:
            frontier.append(dict(complexity=size, validation_loss=loss, indices=list(indices)))
            best = loss
    return frontier, count


def select_cap(frontier, cap):
    eligible = [point for point in frontier if point["complexity"] <= cap]
    return eligible[-1] if eligible else None


def predict_tuple(archives, indices, X):
    return np.column_stack([predict_candidate(decode(archives[j][index]), X)
                            for j, index in enumerate(indices)])
