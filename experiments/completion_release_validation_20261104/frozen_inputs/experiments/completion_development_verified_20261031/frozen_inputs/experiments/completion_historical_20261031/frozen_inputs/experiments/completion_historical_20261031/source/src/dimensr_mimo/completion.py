"""Bounded training-only inverse-transform starts after the full base search.

These are new graphs, not equivalence rewrites of any existing model. All
surviving systems still require the ordinary joint fit and native domain audit.
"""
from itertools import chain, combinations
import time

import numpy as np

from ._single.expression import Node
from ._single.units import ZERO, sub, scale
from .genetics import clone_subgraph
from .graph import Builder, in_bounds
from .selection import digest
from .structure import operator, structured_seeds


def _check_deadline(deadline, counts):
    # Cooperative checks cannot interrupt an individual NumPy/BLAS operation.
    if deadline is not None and time.monotonic() >= deadline:
        error = TimeoutError('Compositional initialization deadline reached.')
        error.initialization_diagnostics = dict(counts)
        raise error


def _legal(tree, grammar):
    if tree.size > grammar.max_nodes or tree.depth > grammar.max_depth:
        return False
    try:
        grammar.trace(tree)
    except (ValueError, KeyError):
        return False
    return True


def _score(tree, constants, X, y):
    """Score the actual strict graph in original response space."""
    values = tree.evaluate(X, constants)
    with np.errstate(all='ignore'):
        score = float(np.mean((values-y)**2)) / max(float(np.var(y)), 1e-12) + 1e-8*tree.size
    return score if np.isfinite(values).all() and np.isfinite(score) else None


def _weighted(tree, values, weight, constants, grammar):
    """Construct a weighted basis, absorbing safe nonnegative square weights."""
    if not np.isfinite(weight):
        return None
    if tree.token == 'square' and weight >= 0:
        child = tree.children[0]
        if child.token == 'mul' and any(c.token.startswith('c') for c in child.children):
            # Basis coefficients are independent and in traversal order.
            return tree, values*np.sqrt(weight)
        if ZERO in constants and 'mul' in grammar.tokens:
            return operator('square', operator('mul', constants[ZERO], child)), np.r_[np.sqrt(weight), values]
    if tree.token == 'mul' and any(c.token.startswith('c') for c in tree.children):
        return tree, values*weight
    if ZERO in constants and 'mul' in grammar.tokens:
        return operator('mul', constants[ZERO], tree), np.r_[weight, values]
    if weight == 1.:
        return tree, values.copy()
    return None


def _basis(grammar, X, constants, counts, deadline):
    """Fixed syntax order: variables, squares, cross products, normalizations."""
    variables = [Node(t, units=u) for t, u in grammar.terminals.items() if t.startswith('x')]
    def normalized():
        if 'mul' not in grammar.tokens:
            return
        for v in variables:
            _check_deadline(deadline, counts)
            inverse = constants.get(sub(ZERO, v.units))
            if inverse is not None:
                term = operator('mul', inverse, v)
                yield term, np.array([1.])
                if 'square' in grammar.tokens:
                    yield operator('square', term), np.array([1.])
            elif 'square' in grammar.tokens:
                inverse_square = constants.get(sub(ZERO, scale(v.units, 2)))
                if inverse_square is not None:
                    yield operator('mul', inverse_square, operator('square', v)), np.array([1.])
    raw = chain(((v, np.array([])) for v in variables),
        ((operator('square', v), np.array([])) for v in variables if 'square' in grammar.tokens),
        ((operator('mul', a, b), np.array([])) for a, b in combinations(variables, 2) if 'mul' in grammar.tokens),
        normalized())
    basis, seen = [], set()
    for tree, values in raw:
        _check_deadline(deadline, counts)
        tree, values = tree.canonical(values)
        if tree in seen:
            continue
        seen.add(tree)
        counts['compositional_basis_attempts'] += 1
        if not _legal(tree, grammar):
            counts['compositional_basis_rejections'] += 1
            continue
        counts['compositional_feature_evaluations'] += 1
        prediction = tree.evaluate(X, values)
        with np.errstate(all='ignore'):
            norm = np.linalg.norm(prediction)
        if not np.isfinite(prediction).all() or not np.isfinite(norm) or norm < 1e-12:
            counts['compositional_basis_rejections'] += 1
            continue
        basis.append((tree, values, prediction))
        if len(basis) == 24:
            break
    counts['compositional_basis_count'] = len(basis)
    return variables[:16], basis


def _inverse_target(token, form, x, y):
    """Reject invalid inverse values, including log underflow and lost sqrt sign."""
    if form == 'multiply' and np.any(np.abs(x) <= 1e-12):
        return None, 'near_zero_divisor'
    if form == 'divide' and np.any(np.abs(y) <= 1e-12):
        return None, 'near_zero_divisor'
    with np.errstate(all='ignore'):
        latent = y if form == 'direct' else y/x if form == 'multiply' else x/y
        if not np.isfinite(latent).all():
            return None, 'nonfinite_latent'
        if token == 'sqrt' and np.any(latent < 0):
            return None, 'negative_sqrt_response'
        target = np.exp(latent) if token == 'log' else np.square(latent)
    if not np.isfinite(target).all() or (token == 'log' and np.any(target <= 0)):
        return None, 'invalid_inverse'
    return target, None


def _compositional_proposals(grammar, X, y, counts, deadline):
    constants = {u: Node(t, units=u) for t, u in grammar.terminals.items() if t.startswith('c')}
    variables, basis = _basis(grammar, X, constants, counts, deadline)
    proposals = []
    forms = [('direct', None)]
    if 'mul' in grammar.tokens:
        forms.extend(('multiply', v) for v in variables)
    if 'div' in grammar.tokens:
        forms.extend(('divide', v) for v in variables)
    for token in ('log', 'sqrt'):
        if token not in grammar.tokens:
            continue
        for form, variable in forms:
            _check_deadline(deadline, counts)
            response_units = (grammar.target if form == 'direct' else
                              sub(grammar.target, variable.units) if form == 'multiply' else
                              sub(variable.units, grammar.target))
            if token == 'log' and response_units != ZERO:
                counts['compositional_unit_rejections'] += 1
                continue
            inner_units = ZERO if token == 'log' else scale(response_units, 2)
            counts['compositional_transforms'] += 1
            x = None if variable is None else X[:, int(variable.token[1:])]
            target, reason = _inverse_target(token, form, x, y)
            if reason is not None:
                counts['compositional_inverse_rejections'][reason] = counts['compositional_inverse_rejections'].get(reason, 0)+1
                continue
            available = [(i, b) for i, b in enumerate(basis) if b[0].units == inner_units]
            intercept_constant = constants.get(inner_units) if 'add' in grammar.tokens else None
            choices = [(b,) for b in available]
            if 'add' in grammar.tokens:
                choices.extend(combinations(available, 2))
            choices = [(chosen, intercept) for chosen in choices
                       for intercept in ([None, intercept_constant] if intercept_constant is not None else [None])]
            for chosen, intercept in choices:
                _check_deadline(deadline, counts)
                design = np.column_stack(([np.ones(len(y))] if intercept is not None else []) + [b[2] for _, b in chosen])
                with np.errstate(all='ignore'):
                    norms = np.linalg.norm(design, axis=0)
                if not np.isfinite(design).all() or not np.isfinite(norms).all() or np.any(norms < 1e-12):
                    counts['compositional_linear_rejections'] += 1
                    continue
                counts['compositional_linear_subproblems'] += 1
                try:
                    coefficients = np.linalg.lstsq(design/norms, target, rcond=None)[0]/norms
                except np.linalg.LinAlgError:
                    counts['compositional_linear_rejections'] += 1
                    continue
                _check_deadline(deadline, counts)
                if not np.isfinite(coefficients).all():
                    counts['compositional_linear_rejections'] += 1
                    continue
                terms = [(intercept, coefficients[:1])] if intercept is not None else []
                for (_, (tree, values, _)), weight in zip(chosen, coefficients[int(intercept is not None):]):
                    weighted = _weighted(tree, values, weight, constants, grammar)
                    if weighted is None:
                        break
                    terms.append(weighted)
                else:
                    tree, values = terms[0]
                    for term, c in terms[1:]:
                        tree, values = operator('add', tree, term), np.r_[values, c]
                    unary = operator(token, tree)
                    variants = [(unary, values, False)] if form == 'direct' else [
                        (operator('mul' if form == 'multiply' else 'div', variable, unary), values, False)]
                    if ZERO in constants and 'mul' in grammar.tokens:
                        scaled = constants[ZERO] if form == 'direct' else operator('mul', constants[ZERO], variable)
                        variants.append((operator('mul' if form != 'divide' else 'div', scaled, unary), np.r_[1., values], True))
                    for root, warm, amplitude in variants:
                        _check_deadline(deadline, counts)
                        counts['compositional_proposed_roots'] += 1
                        root, warm = root.canonical(warm)
                        if root.units != grammar.target or not _legal(root, grammar):
                            counts['compositional_structure_rejections'] += 1
                            continue
                        # No validity/error is inherited from an unabsorbed square.
                        counts['compositional_root_evaluations'] += 1
                        score = _score(root, warm, X, y)
                        if score is None:
                            counts['compositional_numeric_rejections'] += 1
                            continue
                        if form == 'divide':
                            divisor = unary.evaluate(X, values)
                            if not np.isfinite(divisor).all() or np.any(np.abs(divisor) <= 1e-12):
                                counts['compositional_numeric_rejections'] += 1
                                continue
                        source = dict(kind='compositional', transform=token, form=form,
                            variable=None if variable is None else variable.token,
                            basis_indices=[i for i, _ in chosen], intercept=intercept is not None,
                            outer_amplitude=amplitude)
                        proposals.append((root, warm, source))
                    continue
                counts['compositional_coefficient_rejections'] += 1
    return proposals


def completion_seeds(grammar, X, y, limit, *, deadline=None):
    """Merge returned old starts and new roots; return their actual winning sources."""
    counts = dict(compositional_basis_attempts=0, compositional_basis_rejections=0,
        compositional_basis_count=0, compositional_feature_evaluations=0,
        compositional_transforms=0, compositional_unit_rejections=0,
        compositional_inverse_rejections={}, compositional_linear_subproblems=0,
        compositional_linear_rejections=0, compositional_coefficient_rejections=0,
        compositional_proposed_roots=0, compositional_structure_rejections=0,
        compositional_root_evaluations=0, compositional_numeric_rejections=0,
        merged_structure_rejections=0, merged_root_evaluations=0,
        merged_numeric_rejections=0, returned_sources=[])
    if limit <= 0:
        return [], counts
    try:
        seeds, old = structured_seeds(grammar, X, y, limit, compound=True, implicit=True, deadline=deadline)
    except TimeoutError as error:
        counts.update(error.initialization_diagnostics)
        error.initialization_diagnostics = counts
        raise
    counts.update(old)
    implicit = set(old.get('implicit_retained_indices', ()))
    counts['structured_implicit_retained_indices'] = sorted(implicit)
    counts['structured_returned_count'] = len(seeds)
    proposals = [(t, c, dict(kind='implicit' if i in implicit else 'structured', original_index=i))
                 for i, (t, c) in enumerate(seeds)]
    try:
        proposals.extend(_compositional_proposals(grammar, X, y, counts, deadline))
        counts['merged_proposal_count'] = len(proposals)
        winners = {}
        for position, (tree, values, source) in enumerate(proposals):
            _check_deadline(deadline, counts)
            tree, values = tree.canonical(values)
            if tree.units != grammar.target or not _legal(tree, grammar):
                counts['merged_structure_rejections'] += 1
                continue
            counts['merged_root_evaluations'] += 1
            score = _score(tree, values, X, y)
            if score is None:
                counts['merged_numeric_rejections'] += 1
                continue
            # Equal-score ties keep the first source, never a union of sources.
            if tree not in winners or score < winners[tree][0]:
                winners[tree] = score, values, dict(source, proposal_index=position)
        ranked = sorted(winners, key=lambda t:(winners[t][0], t.size, repr(t)))[:limit]
        counts['merged_unique_roots'] = len(winners)
        counts['returned_sources'] = [dict(winners[t][2], returned_index=i,
            score=winners[t][0], tree=t.to_dict(), constants=winners[t][1].tolist()) for i, t in enumerate(ranked)]
        counts['implicit_retained_indices'] = [i for i, s in enumerate(counts['returned_sources']) if s['kind']=='implicit']
        counts['compositional_retained_indices'] = [i for i, s in enumerate(counts['returned_sources']) if s['kind']=='compositional']
        return [(t, winners[t][1]) for t in ranked], counts
    except TimeoutError as error:
        error.initialization_diagnostics = counts
        raise


def compositional_completion(base, evaluator, grammar):
    """Always supplement, within the existing global and twelve-system budgets."""
    cfg = evaluator.cfg
    report = dict(mode='compositional_completion_v1', triggered=True,
        training_eligible=sum(bool(np.all(c.errors <= np.asarray(cfg.acceptance['nrmse']))) for c in base),
        seed_limit=min(12, cfg.population_size//evaluator.Y.shape[1]), new_evaluations=0,
        valid=0, protected=0, protection_rejections=0, cache_skips=0, structure_rejections=0,
        nonimplicit_skips=0, missing_roots=0, initialization_diagnostics=[], events=[],
        implicit_returned_roots=0, compositional_returned_roots=0,
        compositional_linear_subproblems=0, implicit_linear_subproblems=0, linear_subproblems=0,
        stop_reason='completed')
    retained = []
    parent = min(base, key=lambda c:(c.error, c.complexity, digest(c.system.to_dict())))
    report['parent'] = parent.to_dict()

    def stopped():
        if evaluator.expired():
            report['stop_reason'] = 'deadline'
        elif evaluator.calls >= cfg.max_evaluations:
            report['stop_reason'] = 'global_budget'
        elif report['new_evaluations'] >= 12:
            report['stop_reason'] = 'completion_budget'
        else:
            return False
        return True

    def diagnostics(stats):
        report['initialization_diagnostics'].append(stats)
        report['compositional_linear_subproblems'] += stats.get('compositional_linear_subproblems', 0)
        report['implicit_linear_subproblems'] += stats.get('implicit_linear_subproblems', 0)
        report['linear_subproblems'] += stats.get('linear_subproblems', 0)+stats.get('compositional_linear_subproblems', 0)
        for source in stats.get('returned_sources', ()):
            if source['kind'] in ('implicit', 'compositional'):
                report[source['kind']+'_returned_roots'] += 1

    lists, sources = [], []
    for j, g in enumerate(grammar.plain):
        if stopped():
            return retained, report
        try:
            seeds, stats = completion_seeds(g, evaluator.X, evaluator.Y[:, j], report['seed_limit'], deadline=evaluator.deadline)
        except TimeoutError as error:
            diagnostics(error.initialization_diagnostics)
            report['stop_reason'] = 'deadline'
            return retained, report
        diagnostics(stats)
        lists.append(seeds)
        sources.append(stats['returned_sources'])
    for k in range(max(map(len, lists), default=0)):
        if stopped():
            break
        implicit_outputs = [j for j, s in enumerate(sources) if k < len(s) and s[k]['kind']=='implicit']
        compositional_outputs = [j for j, s in enumerate(sources) if k < len(s) and s[k]['kind']=='compositional']
        if not implicit_outputs and not compositional_outputs:
            report['nonimplicit_skips'] += 1
            continue
        builder = Builder()
        root_sources = []
        for j, seeds in enumerate(lists):
            if k < len(seeds):
                tree, constants = seeds[k]
                builder.roots.append(builder.tree(tree, constants))
                root_sources.append(sources[j][k])
            else:
                builder.roots.append(clone_subgraph(builder, parent.system, parent.constants, parent.system.roots[j]))
                report['missing_roots'] += 1
                root_sources.append(dict(kind='parent'))
        graph, warm = builder.finish()
        if graph in evaluator.cache:
            report['cache_skips'] += 1
            continue
        if not in_bounds(graph, cfg):
            report['structure_rejections'] += 1
            continue
        before = evaluator.calls
        child = evaluator(graph, warm)
        new = evaluator.calls > before
        protected = new and child is not None and bool(np.all(child.errors <= parent.errors*1.1+1e-8))
        report['new_evaluations'] += evaluator.calls-before
        report['valid'] += int(new and child is not None)
        report['protected'] += int(protected)
        report['protection_rejections'] += int(new and child is not None and not protected)
        report['events'].append(dict(seed_index=k, parent=parent.to_dict(), graph=graph.to_dict(),
            warm=warm.tolist(), child=None if child is None else child.to_dict(), protected=protected,
            new_evaluation=new, implicit_outputs=implicit_outputs, compositional_outputs=compositional_outputs,
            root_sources=root_sources))
        if protected:
            retained.append(child)
    return retained, report
