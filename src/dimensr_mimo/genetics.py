"""Typed system variation, extending DimenSR's subtree mutation/crossover."""
from dataclasses import replace
import numpy as np
from .graph import Builder, Vertex, paths


def clone_subgraph(builder, system, constants, root):
    mapped, params = {}, {}
    def visit(i):
        if i not in mapped:
            v = system.nodes[i]
            children = tuple(visit(c) for c in v.children)
            p = None
            if v.parameter is not None:
                if v.parameter not in params:
                    params[v.parameter] = len(builder.constants)
                    builder.constants.append(constants[v.parameter])
                p = params[v.parameter]
            builder.nodes.append(Vertex(v.token, children, v.units, p))
            mapped[i] = len(builder.nodes) - 1
        return mapped[i]
    return visit(root)


def offspring(parent, donor, grammar, rng, operator=None):
    system, cfg = parent.system, grammar.cfg
    b = Builder(system, parent.constants)
    if operator is not None:
        op = operator
    elif cfg.use_sharing and rng.random() < .4:
        op = str(rng.choice(['share', 'unshare', 'bind', 'unbind', 'shared_mutation']))
    elif rng.random() < .15:
        op = 'add_term'
    else:
        op = 'crossover' if rng.random() < cfg.crossover_rate else 'mutation'
    if operator is None and op == 'share' and cfg.sharing_strategy == 'matched' and rng.random() < .5:
        op = 'matched_share'
    if op == 'matched_share':
        from .sharing import matched_proposals
        proposals = matched_proposals(system, parent.constants)
        if not proposals:
            return system, parent.constants.copy(), op
        return (*proposals[int(rng.integers(len(proposals)))], op)
    j = int(rng.integers(len(system.roots)))
    locations = list(paths(system, system.roots[j]))
    path, old = locations[int(rng.integers(len(locations)))]
    v = system.nodes[old]
    replacement = old
    if op in ('mutation', 'shared_mutation'):
        full, _ = system.expanded(system.roots[j], parent.constants)
        subtree, _ = system.expanded(old, parent.constants)
        try:
            tree, _ = grammar.mutation_grammar(j, v.units).generate(rng, units=v.units,
                        budget=cfg.max_nodes - full.size + subtree.size,
                        depth=cfg.max_depth - len(path))
        except ValueError:
            # No legal replacement in this bounded grammar; consume the attempt.
            return system, parent.constants.copy(), op
        replacement = b.tree(tree)
    elif op == 'crossover':
        options = [i for i, n in enumerate(donor.system.nodes) if n.units == v.units]
        if options:
            replacement = clone_subgraph(b, donor.system, donor.constants, int(rng.choice(options)))
    elif op == 'add_term' and 'add' in cfg.operators:
        root = system.roots[j]
        tree, _ = grammar.plain[j].generate(rng)
        term = b.tree(tree)
        b.nodes.append(Vertex('add', (root, term), system.nodes[root].units))
        b.roots[j] = len(b.nodes) - 1
        return (*b.finish(), op)
    elif op == 'share':
        other = set().union(*(system.reachable(r) for k, r in enumerate(system.roots) if k != j))
        options = [i for i in sorted(other) if system.nodes[i].units == v.units and i != old]
        if options:
            replacement = int(rng.choice(options))
    elif op == 'unshare':
        replacement = clone_subgraph(b, system, parent.constants, old)
    elif op in ('bind', 'unbind'):
        leaves = [(p, i) for p, i in locations if system.nodes[i].parameter is not None]
        if not leaves:
            return system, parent.constants.copy(), op
        path, old = leaves[int(rng.integers(len(leaves)))]
        v = system.nodes[old]
        p = v.parameter
        if op == 'bind':
            options = sorted({n.parameter for n in system.nodes
                              if n.parameter is not None and n.parameter != p and n.units == v.units})
            if not options:
                return system, parent.constants.copy(), op
            p = int(rng.choice(options))
        else:
            p = len(b.constants)
            b.constants.append(parent.constants[v.parameter])
        b.nodes.append(replace(v, parameter=p))
        replacement = len(b.nodes) - 1
    if op == 'shared_mutation':
        # Rewrite every use of this graph node, not just the selected root path.
        mapped = {old: replacement}
        def rewrite(i):
            if i not in mapped:
                n = system.nodes[i]
                children = tuple(rewrite(c) for c in n.children)
                if children == n.children:
                    mapped[i] = i
                else:
                    b.nodes.append(replace(n, children=children))
                    mapped[i] = len(b.nodes) - 1
            return mapped[i]
        b.roots = [rewrite(r) for r in system.roots]
    else:
        b.roots[j] = b.replace(system.roots[j], path, replacement)
    return (*b.finish(), op)
