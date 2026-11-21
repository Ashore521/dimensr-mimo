"""Optional training-only final compression in the engineering flow."""
import numpy as np
import sympy as sp
from . import audit
from .graph import System, Vertex
from ._single.units import unit, ZERO, add, sub
from .complexity import local_rademacher, mixed_complexity
from .evaluation import Candidate
from .portfolio_proposals import capacity_rejection


class Graph:
    def __init__(self, meta):
        self.inputs = list(map(unit, meta["feature_units"]))
        self.catalog = list(map(unit, meta["constant_units"]))
        self.goal = unit(meta["target_units"][0])
        assert all(unit(u) == self.goal for u in meta["target_units"])
        self.nodes, self.ids, self.values = [], {}, []

    def vertex(self, token, children, units, parameter=None):
        v = Vertex(token, tuple(children), units, parameter)
        if v not in self.ids:
            self.ids[v] = len(self.nodes)
            self.nodes.append(v)
        return self.ids[v]

    def op(self, token, *children):
        args = [self.nodes[c].units for c in children]
        if token in ("mul", "div"):
            units = (add if token == "mul" else sub)(*args)
        elif token == "square":
            units = add(args[0], args[0])
        elif token in ("cos", "sin"):
            assert args == [ZERO]
            units = ZERO
        else:
            assert token == "add" and args[0] == args[1]
            units = args[0]
        return self.vertex(token, children, units)

    def product(self, factors):
        if not factors:
            return None
        node = factors[0]
        for other in factors[1:]:
            node = self.op("mul", node, other)
        return node

    def basis(self, term):
        positive, negative = [], []
        for atom, power in sorted(term.as_powers_dict().items(), key=lambda kv: str(kv[0])):
            if atom == 1:
                continue
            assert power.is_Integer
            trig = atom.func in (sp.sin, sp.cos)
            symbol = atom.args[0] if trig else atom
            assert symbol.is_Symbol and symbol.name.startswith("x")
            i = int(symbol.name[1:])
            node = self.vertex(symbol.name, (), self.inputs[i])
            if trig:
                node = self.op(atom.func.__name__, node)
            pieces = negative if power < 0 else positive
            n = abs(int(power))
            if n >= 2:
                pieces.extend([self.op("square", node)] * (n // 2))
            if n % 2:
                pieces.append(node)
        return self.product(positive), self.product(negative)

    def weighted(self, term, value):
        num, den = self.basis(term)
        units = [self.nodes[v].units if v is not None else ZERO for v in (num, den)]
        cu = sub(self.goal, sub(*units))
        if cu not in self.catalog:
            return None
        coef = self.vertex("c"+str(self.catalog.index(cu)), (), cu, len(self.values))
        self.values.append(float(value))
        if num is None:
            return coef if den is None else self.op("div", coef, den)
        factor = num if den is None else self.op("div", num, den)
        return self.op("mul", coef, factor)

    def total(self, columns):
        # A balanced sum avoids charging linear accumulation as polynomial depth.
        while len(columns) > 1:
            columns = [self.op("add", *columns[i:i+2]) if i+1 < len(columns) else columns[i]
                       for i in range(0, len(columns), 2)]
        return columns[0]


def expansion_bound(expr, limit):
    if expr.is_Symbol or expr.is_Number:
        return 1
    if expr.func in (sp.sin, sp.cos):
        return 1 if expr.args[0].is_Symbol else limit+1
    children = [expansion_bound(arg, limit) for arg in expr.args]
    if max(children, default=0) > limit:
        return limit+1
    if expr.is_Add:
        return sum(children)
    if expr.is_Mul:
        n = 1
        for count in children:
            n *= count
            if n > limit:
                break
        return n
    if expr.is_Pow and expr.exp.is_Integer and abs(expr.exp) <= 16:
        if expr.exp < 0:
            return 1 if children[0] == 1 else limit+1
        return children[0] ** int(expr.exp)
    return limit+1


def dictionary(teacher, constants, meta, X, bounds, max_columns):
    terms = set()
    for formula in teacher.formulas(constants):
        f = sp.sympify(formula, rational=True)
        if expansion_bound(f, 2*max_columns) > 2*max_columns:
            continue
        terms.update(t.as_coeff_Mul()[1] for t in sp.Add.make_args(sp.expand(f)))
    terms.discard(sp.Integer(1))
    if len(terms) > max_columns:
        return [], None, {"library_limit": len(terms)}
    keep, columns, rejects = [], [], {}
    for term in sorted(terms, key=str):
        if any(not power.is_Integer or not (atom.is_Symbol or
               (atom.func in (sp.sin, sp.cos) and atom.args[0].is_Symbol))
               for atom, power in term.as_powers_dict().items()):
            rejects["unsupported_basis"] = rejects.get("unsupported_basis", 0)+1
            continue
        if any(trig.args[0].is_Symbol and unit(meta["feature_units"][int(trig.args[0].name[1:])]) != ZERO
               for trig in term.atoms(sp.sin, sp.cos)):
            rejects["trig_units"] = rejects.get("trig_units", 0)+1
            continue
        b = Graph(meta)
        root = b.weighted(term, 1.)
        if root is None:
            rejects["coefficient_units"] = rejects.get("coefficient_units", 0)+1
            continue
        g = System(tuple(b.nodes), (root,))
        if g.audit(b.values, bounds)[0]["status"] != "enclosed":
            rejects["domain"] = rejects.get("domain", 0)+1
            continue
        col = g.evaluate(X, b.values)[:, 0]
        if np.isfinite(col).all() and col.std() > 1e-12:
            keep.append(term)
            columns.append(col)
    return keep, np.column_stack(columns) if columns else None, rejects


def fit_support(columns, y, floor):
    centered = columns-columns.mean(axis=0)
    norms = np.linalg.norm(centered, axis=0)
    standardized = centered / norms
    directions = standardized.copy()
    v = y-y.mean()
    variance_sum = v@v
    chosen = []
    for _ in range(columns.shape[1]):
        resnorm = np.linalg.norm(directions, axis=0)
        gain = np.abs(v @ directions) / np.maximum(resnorm, 1e-15)
        gain[resnorm < 1e-7] = -np.inf
        gain[chosen] = -np.inf
        i = int(gain.argmax())
        if not np.isfinite(gain[i]):
            break
        chosen.append(i)
        q = directions[:, i]/resnorm[i]
        directions -= np.outer(q, q@directions)
        coef = np.linalg.lstsq(standardized[:, chosen], y-y.mean(), rcond=1e-12)[0] / norms[chosen]
        intercept = y.mean()-columns[:, chosen].mean(axis=0)@coef
        v = y-(intercept+columns[:, chosen]@coef)
        r2 = 1-(v@v)/variance_sum
        if r2 >= floor:
            return chosen, coef, intercept
    return None


def tolerances(settings, outputs):
    if set(settings) != {"enabled", "r2_drop"} or type(settings["enabled"]) is not bool:
        raise ValueError("Expected enabled (boolean) and r2_drop.")
    a = np.asarray(settings["r2_drop"])
    if a.dtype.kind not in "fiu" or a.shape not in [(), (outputs,)]:
        raise ValueError("r2_drop requires one number or one per output.")
    if not np.isfinite(a).all() or (a < 0).any():
        raise ValueError("r2_drop must be finite and nonnegative.")
    return np.broadcast_to(a.astype(float), (outputs,))


def compress(parent, X, Y, scales, signs, task, cfg, settings, target, *, working_memory=None):
    drop = tolerances(settings, Y.shape[1])
    variance = Y.var(axis=0)
    report = dict(enabled=settings["enabled"], r2_drop=drop.tolist())

    def keep(reason):
        return parent, dict(report, accepted=False, reason=reason)

    if not settings["enabled"]:
        return keep("disabled")
    if (variance == 0).any() or len(set(map(unit, task["targets"]))) != 1:
        return keep("unsupported_constant_or_mixed_dimension_outputs")
    ref = 1 - (np.asarray(parent.errors)*scales)**2/variance
    floor = ref-drop
    if (ref >= target).all():
        floor = np.maximum(floor, target)
    report.update(reference_r2=ref.tolist(), floors=floor.tolist(), reference_complexity=parent.complexity)
    if audit.complexity(parent.system)["expanded_total_nodes"] > 10000:
        return keep("expansion_limit")
    meta = dict(feature_units=task["features"], constant_units=task["constants"], target_units=task["targets"])
    if unit(task["targets"][0]) not in set(map(unit, task["constants"])):
        return keep("no_typed_intercept")
    max_columns = min(512, int(cfg.max_working_memory_mb*2**20 // (8*len(X)*10)))
    terms, A, skipped = dictionary(parent.system, parent.constants, meta, X, task["bounds"], max_columns)
    report.update(terms=len(terms), skipped=skipped)
    if A is None:
        return keep("empty_or_oversize_library")
    builder = Graph(meta)
    roots = []
    for j in range(Y.shape[1]):
        try:
            support = fit_support(A, Y[:, j], floor[j])
        except np.linalg.LinAlgError:
            return keep("linear_solve_failed")
        if support is None:
            return keep("accuracy_floor")
        indices, weights, intercept = support
        parts = [builder.weighted(sp.Integer(1), intercept)]
        parts.extend(builder.weighted(terms[i], w) for i, w in zip(indices, weights))
        roots.append(builder.total(parts))
    g, c = System(tuple(builder.nodes), tuple(roots)), np.asarray(builder.values)
    if any(n.children and n.token not in cfg.operators for n in g.nodes):
        return keep("operator_catalog")
    if not np.isfinite(c).all() or np.max(np.abs(c)) > cfg.constant_bound:
        return keep("coefficient_bound")
    if capacity_rejection(g, cfg)[0]:
        return keep("structure_limit")
    if working_memory is not None and not working_memory(g)["allowed"]:
        return keep("working_memory_limit")
    audit.units(g, task["features"], task["targets"], task["constants"])
    if any(a["status"] != "enclosed" for a in g.audit(c, task["bounds"], cfg.interval_max_boxes, cfg.output_bounds)):
        return keep("domain_check")
    errors = np.sqrt(np.mean(((g.evaluate(X, c)-Y)/scales)**2, axis=0))
    r2 = 1-(errors*scales)**2/variance
    if not np.isfinite(r2).all() or (r2 < floor).any():
        return keep("replayed_accuracy_floor")
    raw, scaled = local_rademacher(g, c, X, scales, signs, cfg.perturbation_radius)
    complexity = mixed_complexity(g, scaled, cfg.complexity_weights)
    if not np.isfinite(complexity) or complexity >= parent.complexity:
        return keep("complexity")
    child = Candidate(g, c, errors, complexity, raw, scaled)
    return child, dict(report, accepted=True, r2=r2.tolist(), complexity=complexity)


def compression_candidate(pool, evaluator, features, targets, constants, cfg, reference):
    """Generate one training-only alternative before the validation pool freezes."""
    from .pareto import violations
    from .selection import digest
    from .stopping import select_training

    scale = evaluator.scales
    variance = evaluator.Y.var(axis=0)
    metric = cfg.stop_metric or 'r2'
    cutoff = .90 if cfg.stop_target is None else cfg.stop_target
    parent, selection = select_training(pool, scale, variance, metric=metric, target=cutoff)
    # Convert the trained parent's joint NRMSE target to the same componentwise
    # R² floors used by the compressor. Constant-output tasks return the parent.
    floor = cutoff if metric == 'r2' else 1 - (np.asarray(cutoff)*scale)**2 / np.where(variance > 0, variance, 1.)
    task = dict(features=features, targets=targets, constants=constants, bounds=evaluator.bounds)
    child, report = compress(parent, evaluator.X, evaluator.Y, scale, evaluator.signs, task, cfg,
                             dict(enabled=True, r2_drop=cfg.compression_r2_drop), floor,
                             working_memory=evaluator.working_memory)
    report['parent_id'] = selection['candidate_id']
    if child is parent:
        return None, report
    if (cfg.fixed_training_reference is not None and violations(child.errors, reference, cfg) or
            not evaluator.admissible(child.system.evaluate(evaluator.probes, child.constants))):
        return None, dict(report, accepted=False, reason='protection_or_domain_probes')
    if any(c.system == child.system and np.array_equal(c.constants, child.constants) for c in pool):
        return None, dict(report, accepted=False, reason='already_in_pool')
    report['candidate_id'] = digest(child.to_dict())
    return child, report
