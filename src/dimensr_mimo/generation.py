"""Sequential root generation with typed references to previously generated nodes."""
from dataclasses import dataclass
from .grammar import Grammar
from .graph import Builder
from .syntax import Node


@dataclass
class SystemTrace:
    traces: list

    @property
    def tokens(self):
        return [x for t in self.traces for x in t.tokens]


class SystemGrammar:
    def __init__(self, feature_units, target_units, constant_units, cfg, extra_units=()):
        self.features, self.targets, self.constants = feature_units, target_units, constant_units
        self.cfg, self.extra = cfg, extra_units
        self._cache = {}
        self.plain = [self.grammar(j) for j in range(len(target_units))]
        self.n_tokens = len(self.plain[0].tokens)
        self.context_size = 9 + len(target_units) + 2

    def bank(self, nodes):
        return list(range(max(0, len(nodes) - self.cfg.max_references), len(nodes))) if self.cfg.use_sharing else []

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
        b, traces = Builder(), []
        for j in range(len(self.targets)):
            bank = self.bank(b.nodes)
            g = self.grammar(j, b.nodes, bank)
            tree, trace = g.generate(rng, policy)
            b.roots.append(b.tree(tree, references=bank))
            traces.append(trace)
        graph, constants = b.finish()
        return graph, constants, SystemTrace(traces)

    def trace(self, system):
        """Replay surviving DAGs using references where representable in the bank.

        An elite with an out-of-bank binding is skipped, never given a false
        likelihood. GP can reach a larger graph family than neural generation.
        """
        b, mapped, traces = Builder(), {}, []
        parameter_nodes = {}
        for j, root in enumerate(system.roots):
            bank = self.bank(b.nodes)
            by_old = {old: bank.index(new) for old, new in mapped.items() if new in bank}
            def syntax(i):
                v = system.nodes[i]
                if i in by_old:
                    return Node('r' + str(by_old[i]), units=v.units)
                if i in mapped:
                    raise ValueError('Shared node outside neural reference bank.')
                if v.parameter is not None and v.parameter in parameter_nodes:
                    old = parameter_nodes[v.parameter]
                    if old in by_old:
                        return Node('r' + str(by_old[old]), units=v.units)
                    raise ValueError('Within-root parameter binding not replayable.')
                if v.parameter is not None:
                    parameter_nodes[v.parameter] = i
                return Node(v.token, tuple(syntax(c) for c in v.children), v.units)
            t = syntax(root)
            g = self.grammar(j, b.nodes, bank)
            traces.append(g.trace(t))
            new_root = b.tree(t, references=bank)
            def map_nodes(old, new):
                mapped[old] = new
                for a, c in zip(system.nodes[old].children, b.nodes[new].children):
                    map_nodes(a, c)
            map_nodes(root, new_root)
            b.roots.append(new_root)
        return SystemTrace(traces)
