"""Sequential root generation with typed references to previously generated nodes."""
from dataclasses import dataclass
from collections import Counter, defaultdict
import numpy as np
from .grammar import Grammar
from .graph import Builder, Vertex
from .capacity import compact
from .syntax import Node


@dataclass
class SystemTrace:
    traces: list

    @property
    def tokens(self):
        return [x for t in self.traces for x in t.tokens]


class ReplayError(ValueError):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


class SystemGrammar:
    def __init__(self, feature_units, target_units, constant_units, cfg, extra_units=()):
        self.features, self.targets, self.constants = feature_units, target_units, constant_units
        self.cfg, self.extra = cfg, extra_units
        self.parameter_slots = []
        if cfg.parameter_bank_counts is not None:
            if len(cfg.parameter_bank_counts) != len(constant_units):
                raise ValueError('One parameter-bank count is required per constant unit.')
            for token, (units, count) in enumerate(zip(constant_units, cfg.parameter_bank_counts)):
                self.parameter_slots.extend([('c' + str(token), units)] * count)
        self._cache = {}
        self._mutation_grammars = {}
        self.plain = [self.grammar(j) for j in range(len(target_units))]
        self.n_tokens = len(self.plain[0].tokens)
        self.context_size = 9 + len(target_units) + 2

    def bank(self, nodes):
        if not self.cfg.use_sharing:
            return []
        reserved = len(self.parameter_slots)
        return list(range(reserved)) + list(range(max(reserved, len(nodes) - (self.cfg.max_references-reserved)), len(nodes)))

    def builder(self):
        b = Builder()
        for token, units in self.parameter_slots:
            b.nodes.append(Vertex(token, units=units, parameter=len(b.constants)))
            b.constants.append(1.)
        return b

    def mutation_grammar(self, j, units, *, budget=None, depth=None):
        plain = self.plain[j]
        nodes = max(self.cfg.max_nodes, budget or self.cfg.max_nodes)
        height = max(self.cfg.max_depth, depth or self.cfg.max_depth)
        in_plain = units in plain.units
        if in_plain and nodes <= plain.max_nodes and height <= plain.max_depth:
            return plain
        # Archived Pareto parents can exceed the current generation capacity.
        key = (j, units)
        cached = self._mutation_grammars.get(key)
        if cached is None or cached.max_nodes < nodes or cached.max_depth < height:
            if cached is not None:
                nodes, height = max(nodes, cached.max_nodes), max(height, cached.max_depth)
            target = self.targets[j] if in_plain else units
            self._mutation_grammars[key] = Grammar(self.features, target, self.constants,
                self.cfg.operators, nodes, height, self.extra,
                max_references=self.cfg.max_references)
        return self._mutation_grammars[key]

    def grammar(self, j, nodes=(), references=()):
        context = [float(i == j) for i in range(len(self.targets))]
        context += [len(nodes) / (len(self.targets) * self.cfg.max_nodes),
                    len({v.parameter for v in nodes if v.parameter is not None}) /
                    (len(self.targets) * self.cfg.max_nodes)]
        ref_units = tuple(nodes[i].units for i in references)
        ref_shapes = ()
        if self.cfg.reference_budget=='expanded':
            sizes,depths=[],[]
            for node in nodes:
                sizes.append(min(self.cfg.max_nodes+1,1+sum(sizes[c] for c in node.children)))
                depths.append(1+max((depths[c] for c in node.children),default=0))
            ref_shapes=tuple((sizes[i],depths[i]) for i in references)
        key = (j, ref_units, ref_shapes)
        if key in self._cache:
            self._cache[key].output_context = context
            return self._cache[key]
        g = Grammar(self.features, self.targets[j], self.constants, self.cfg.operators,
                       self.cfg.max_nodes, self.cfg.max_depth, self.extra,
                       reference_units=ref_units,
                       max_references=self.cfg.max_references, output_context=context,
                       **({'reference_shapes':ref_shapes} if self.cfg.reference_budget=='expanded' else {}))
        self._cache[key] = g
        return g

    def generate(self, rng, policy=None):
        b, traces = self.builder(), []
        for j in range(len(self.targets)):
            bank = self.bank(b.nodes)
            g = self.grammar(j, b.nodes, bank)
            tree, trace = g.generate(rng, policy)
            b.roots.append(b.tree(tree, references=bank))
            traces.append(trace)
        graph, constants = b.finish()
        if self.parameter_slots:
            graph, constants = compact(graph, constants)
        return graph, constants, SystemTrace(traces)

    def trace(self, system, *, return_system=False):
        """Replay surviving DAGs using references where representable in the bank.

        An elite with an out-of-bank binding is skipped, never given a false
        likelihood. GP can reach a larger graph family than neural generation.
        """
        b, mapped, traces = self.builder(), {}, []
        parameter_nodes = {}
        parameter_refs = {}
        if self.parameter_slots:
            uses = [0] * len(system.nodes)
            for root in system.roots:
                uses[root] += 1
            counts, units = Counter(), {}
            for i in reversed(range(len(system.nodes))):
                node = system.nodes[i]
                if node.parameter is not None:
                    counts[node.parameter] += uses[i]
                    units[node.parameter] = node.units
                for child in node.children:
                    uses[child] = min(2, uses[child] + uses[i])
            available = defaultdict(list)
            for index, (_, unit) in enumerate(self.parameter_slots):
                available[unit].append(index)
            for parameter in sorted(counts):
                if counts[parameter] > 1:
                    if not available[units[parameter]]:
                        raise ReplayError('parameter_bank_capacity')
                    parameter_refs[parameter] = available[units[parameter]].pop(0)
        for j, root in enumerate(system.roots):
            bank = self.bank(b.nodes)
            by_old = {old: bank.index(new) for old, new in mapped.items() if new in bank}
            def syntax(i):
                v = system.nodes[i]
                if i in by_old:
                    return Node('r' + str(by_old[i]), units=v.units)
                if v.parameter in parameter_refs:
                    return Node('r' + str(bank.index(parameter_refs[v.parameter])), units=v.units)
                if i in mapped and not self.parameter_slots:
                    raise ReplayError('reference_bank')
                if v.parameter is not None and v.parameter in parameter_nodes:
                    old = parameter_nodes[v.parameter]
                    if old in by_old:
                        return Node('r' + str(by_old[old]), units=v.units)
                    raise ReplayError('parameter_binding')
                if v.parameter is not None:
                    parameter_nodes[v.parameter] = i
                return Node(v.token, tuple(syntax(c) for c in v.children), v.units)
            t = syntax(root)
            g = self.grammar(j, b.nodes, bank)
            try:
                traces.append(g.trace(t))
            except (ValueError, KeyError) as error:
                reason = ('length_and_depth' if t.size > g.max_nodes and t.depth > g.max_depth else
                          'length' if t.size > g.max_nodes else 'depth' if t.depth > g.max_depth else
                          'grammar_production')
                raise ReplayError(reason) from error
            new_root = b.tree(t, references=bank)
            def map_nodes(old, new):
                mapped[old] = new
                for a, c in zip(system.nodes[old].children, b.nodes[new].children):
                    map_nodes(a, c)
            map_nodes(root, new_root)
            b.roots.append(new_root)
        replayed, warm = b.finish()
        if self.parameter_slots:
            replayed, _ = compact(replayed, warm)
            expected, _ = compact(system, np.ones(system.n_parameters))
            if replayed != expected:
                raise ReplayError('graph_semantics')
        trace = SystemTrace(traces)
        return (trace, replayed) if return_system else trace
