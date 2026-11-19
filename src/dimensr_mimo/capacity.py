"""Opt-in proposal capacities and structure-only affine parameter analysis."""
from collections import Counter

def parameter_partition(system):
    """Return a deterministic jointly affine subset and its nonlinear complement.

    A parameter under a nonlinear operator or denominator cannot be eliminated.
    Products add conflicts, including self-conflicts for repeated bindings.
    """
    dependencies, forced, conflicts = [], set(), [set() for _ in range(system.n_parameters)]
    for node in system.nodes:
        parts = [dependencies[c] for c in node.children]
        used = {node.parameter} if node.parameter is not None else set().union(*parts)
        if node.token == 'mul':
            for left in parts[0]:
                conflicts[left].update(parts[1])
            for right in parts[1]:
                conflicts[right].update(parts[0])
        elif node.token == 'div':
            forced.update(parts[1])
        elif node.children and node.token not in ('add', 'sub', 'neg'):
            forced.update(used)
        dependencies.append(used)
    forced.update(p for p, neighbours in enumerate(conflicts) if p in neighbours)
    available = set(range(system.n_parameters)) - forced
    linear = []
    for parameter in sorted(available, key=lambda p: (len(conflicts[p] & available), p)):
        if not conflicts[parameter].intersection(linear):
            linear.append(parameter)
    linear = tuple(sorted(linear))
    return linear, tuple(p for p in range(system.n_parameters) if p not in linear)


def shape(system, cfg=None):
    depths = []
    for node in system.nodes:
        depths.append(1 + max((depths[c] for c in node.children), default=0))
    separable_dimension = len(parameter_partition(system)[1])
    mode = getattr(cfg, 'constant_fit_mode', 'joint')
    dimension = system.n_parameters if separable_dimension else 0
    if mode == 'separable':
        dimension = separable_dimension
    elif mode == 'local':
        dimension = min(system.n_parameters, cfg.constant_local_parameters)
    if getattr(cfg, 'search_schedule', None) == 'portfolio':
        dimension = separable_dimension
    return dict(parameters=system.n_parameters, nonlinear_parameters=dimension,
                separable_nonlinear_parameters=separable_dimension,
                nodes=len(system.nodes), depth=[depths[r] for r in system.roots])


def rejection(system, limits, cfg=None):
    if limits is None:
        return None
    metrics = shape(system, cfg)
    for key, limit in limits.items():
        values = metrics[key] if key == 'depth' else [metrics[key]]
        if any(value > limit for value in values):
            return key
    return None


def compact(system, constants):
    """Hash-cons identical vertices without merging independent parameters."""
    from .graph import Builder, Vertex
    builder, mapped, seen = Builder(), {}, {}
    builder.constants = list(constants)
    for index, node in enumerate(system.nodes):
        vertex = Vertex(node.token, tuple(mapped[c] for c in node.children), node.units, node.parameter)
        if vertex not in seen:
            seen[vertex] = len(builder.nodes)
            builder.nodes.append(vertex)
        mapped[index] = seen[vertex]
    builder.roots = [mapped[r] for r in system.roots]
    return builder.finish()


def output_sparsity(system, samples):
    import numpy as np
    from scipy.sparse import csr_matrix
    dependencies = []
    for node in system.nodes:
        dependencies.append({node.parameter} if node.parameter is not None else
                            set().union(*(dependencies[c] for c in node.children)))
    mask = np.zeros((len(system.roots), system.n_parameters), dtype=bool)
    for j, root in enumerate(system.roots):
        mask[j, list(dependencies[root])] = True
    return csr_matrix(np.tile(mask, (samples, 1)))


def local_parameters(system, limit):
    """Round-robin outputs, preferring parameters nearest each output root."""
    from collections import deque
    ordered = []
    for root in system.roots:
        pending, seen, distances = deque([(root, 0)]), set(), {}
        while pending:
            index, distance = pending.popleft()
            if index in seen:
                continue
            seen.add(index)
            node = system.nodes[index]
            if node.parameter is not None:
                distances.setdefault(node.parameter, distance)
            pending.extend((child, distance + 1) for child in node.children)
        ordered.append(sorted(distances, key=lambda p: (distances[p], p)))
    chosen = []
    for rank in range(max(map(len, ordered), default=0)):
        for parameters in ordered:
            if rank < len(parameters) and parameters[rank] not in chosen:
                chosen.append(parameters[rank])
                if len(chosen) == limit:
                    return tuple(sorted(chosen))
    return tuple(sorted(chosen))


def imitation(grammar, candidates, count):
    """Scan real complete systems and return only exact grammar replays."""
    traces, reasons = [], Counter()
    for candidate in candidates:
        if len(traces) >= count:
            break
        try:
            traces.append(grammar.trace(candidate.system))
        except (ValueError, KeyError, MemoryError, RecursionError) as error:
            reasons[getattr(error, 'reason', type(error).__name__)] += 1
    return traces, dict(reasons)
