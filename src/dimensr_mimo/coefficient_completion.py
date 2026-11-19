"""Identity-initialized amplitude proposals on the additive output frontier."""
from collections import Counter

from ._single.units import ZERO
from .graph import Builder, Vertex
from .portfolio_proposals import Proposal, capacity_rejection, identity


def complete_coefficients(candidate, constant_units, cfg):
    """Return bounded single-term edits and explicit reasons for skipped edits.

    Only dimensionless multiplication preserves an existing term's units and
    domain. Parameters remain bound through Builder, including cross-output
    bindings. Counts saturate at two, so shared DAGs never expand for this scan.
    """
    system = candidate.system
    skipped, proposals = [], []
    if 'mul' not in cfg.operators:
        return [], [dict(reason='multiplication_unavailable')]
    if ZERO not in constant_units:
        return [], [dict(reason='dimensionless_constant_unavailable')]
    if cfg.constant_bound < 1.:
        return [], [dict(reason='identity_outside_constant_bound', identity=1., maximum=float(cfg.constant_bound))]
    token = 'c' + str(constant_units.index(ZERO))
    occurrences, amplitudes = [], []
    for node in system.nodes:
        counts = Counter()
        for child in node.children:
            counts.update(occurrences[child])
        if node.parameter is not None:
            counts[node.parameter] += 1
        occurrences.append(Counter({p: min(2, n) for p, n in counts.items()}))
        amplitude = {node.parameter} if node.parameter is not None else set()
        if node.token in ('mul', 'div', 'neg'):
            amplitude.update(p for child in node.children for p in amplitudes[child])
        amplitudes.append(amplitude)
    total = Counter()
    for root in system.roots:
        total.update(occurrences[root])
    # Frontier traversal is bounded independently of expanded DAG size. Shared
    # additive subgraphs may otherwise contain exponentially many occurrences.
    limit = cfg.max_nodes * len(system.roots)
    examined = 0
    for output, root in enumerate(system.roots):
        pending = [((), root)]
        while pending and examined < limit:
            path, index = pending.pop()
            examined += 1
            node = system.nodes[index]
            if node.token in ('add', 'sub'):
                pending.extend((path + (j,), child) for j, child in reversed(list(enumerate(node.children))))
                continue
            if any(total[p] == 1 for p in amplitudes[index]):
                continue
            builder = Builder(system, candidate.constants)
            parameter = len(builder.constants)
            builder.constants.append(1.)
            builder.nodes.append(Vertex(token, units=ZERO, parameter=parameter))
            coefficient = len(builder.nodes) - 1
            builder.nodes.append(Vertex('mul', (coefficient, index), node.units))
            builder.roots[output] = builder.replace(root, path, len(builder.nodes) - 1)
            graph, warm = builder.finish()
            reasons, metrics = capacity_rejection(graph, cfg)
            if reasons:
                skipped.append(dict(reason='capacity', output=output, path=list(path),
                                    reasons=reasons, shape=metrics))
                continue
            proposals.append(Proposal(graph, warm, 'features', (output,),
                'coefficient_completion', parents=(identity(candidate),)))
        if pending:
            skipped.append(dict(reason='frontier_scan_limit', output=output, maximum=limit))
            break
    return proposals, skipped
