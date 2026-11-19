"""Joint neural/GP exploration and constrained NSGA-II, adapted from DimenSR."""
from dataclasses import asdict, replace
from copy import deepcopy
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import warnings
import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted
from .config import SearchConfig
from ._single.units import ZERO, unit, encode
from ._single.seeding import typed_seeds
from ._single.structure import structured_seeds
from ._single.pareto import fronts
from .generation import SystemGrammar
from .graph import System, Builder, in_bounds
from .policy import Policy
from .genetics import offspring, clone_subgraph
from .evaluation import Evaluator, Candidate, search_score
from .pareto import ordered, diverse, violations, retain_parameters
from .audit import data_digest
from .stopping import target_vector

PACKAGE_VERSION='1.8.0'


def arrays(X, Y):
    X, Y = check_array(X, dtype=float), check_array(Y, dtype=float)
    if len(X) != len(Y):
        raise ValueError('X and Y must have the same row count; Y is a 2D output matrix.')
    return X, Y


def predict_system(system, constants, X, cfg):
    if cfg.final_selection != 'pareto':
        return system.evaluate(X, constants)
    bytes_per_row = 16*(len(system.nodes)+len(system.roots)+8)
    batch = max(1, int(cfg.max_working_memory_mb*1024**2)//bytes_per_row)
    prediction = np.empty((len(X), len(system.roots)))
    for start in range(0, len(X), batch):
        prediction[start:start+batch] = system.evaluate(X[start:start+batch], constants)
    return prediction


class DimenSRMIMORegressor(RegressorMixin, BaseEstimator):
    def __init__(self, config=None, feature_units=None, target_units=None,
                 constant_units=None, feature_names=None, extra_units=None):
        self.config, self.feature_units, self.target_units = config, feature_units, target_units
        self.constant_units, self.feature_names, self.extra_units = constant_units, feature_names, extra_units

    def fit(self, X, Y, *, X_val=None, Y_val=None, input_bounds=None, X_domain=None, callback=None,
            checkpoint_path=None, resume=False, stop_requested=None):
        for name in list(vars(self)):
            if name.endswith('_'):
                delattr(self, name)
        start, cpu_start = time.monotonic(), time.process_time()
        cfg = replace(self.config) if self.config is not None else SearchConfig()
        cfg.acceptance = deepcopy(cfg.acceptance)
        cfg.stage_evaluations = deepcopy(cfg.stage_evaluations)
        cfg.portfolio_settings = deepcopy(cfg.portfolio_settings)
        cfg.enhanced_prefix = deepcopy(cfg.enhanced_prefix)
        cfg.validate()
        X, Y = arrays(X, Y)
        if len(X) < 5:
            raise ValueError('At least five rows are required.')
        p, q = X.shape[1], Y.shape[1]
        if cfg.fixed_training_reference is not None and len(cfg.fixed_training_reference) != q:
            raise ValueError('Fixed training reference must match the output count.')
        target = target_vector(cfg.stop_metric, cfg.stop_target, q)
        if target is not None:
            cfg.stop_target = target.tolist()
            if cfg.enhanced_prefix is not None:
                cfg.enhanced_prefix['stop_target'] = target.tolist()
        pareto_mode = cfg.final_selection == 'pareto'
        if cfg.final_selection=='precision':
            from .acceptance import validate_limits
            validate_limits(cfg.acceptance,q)
        features = tuple(unit(u) for u in self.feature_units) if self.feature_units is not None else (ZERO,) * p
        targets = tuple(unit(u) for u in self.target_units) if self.target_units is not None else (ZERO,) * q
        constants = tuple(unit(u) for u in self.constant_units) if self.constant_units is not None else (ZERO,)
        extra = tuple(unit(u) for u in self.extra_units) if self.extra_units is not None else ()
        names = list(self.feature_names) if self.feature_names is not None else [f'x{i}' for i in range(p)]
        if len(features) != p or len(targets) != q or len(names) != p:
            raise ValueError('Feature/target units and names must match input/output columns.')
        if any(not isinstance(n, str) or not n for n in names) or len(set(names)) != p:
            raise ValueError('Feature names must be nonempty unique strings.')
        if not cfg.use_units:
            features, targets, constants, extra = (ZERO,) * p, (ZERO,) * q, ((ZERO,) if constants else ()), ()
        constants = tuple(dict.fromkeys(constants))
        if (X_val is None) != (Y_val is None):
            raise ValueError('Provide both X_val and Y_val.')
        if X_val is None:
            split = np.random.default_rng(np.random.SeedSequence([cfg.random_state, 911])).permutation(len(X))
            nv = max(1, int(np.ceil(len(X) * cfg.validation_fraction)))
            if len(X) - nv < 2:
                raise ValueError('Validation split leaves too few training rows.')
            self.validation_indices_, self.training_indices_ = split[:nv], split[nv:]
            X_val, Y_val = X[split[:nv]], Y[split[:nv]]
            X, Y = X[split[nv:]], Y[split[nv:]]
        else:
            X_val, Y_val = arrays(X_val, Y_val)
            if X_val.shape[1] != p or Y_val.shape[1] != q:
                raise ValueError('Validation feature/output count mismatch.')
            self.training_indices_, self.validation_indices_ = np.arange(len(X)), None
        if cfg.budget_mode == 'evaluations':
            if cfg.search_schedule == 'portfolio':
                if cfg.enhanced_prefix is not None:
                    from .enhanced_portfolio import EnhancedPortfolioSearch as Search
                else:
                    from .portfolio_search import PortfolioSearch as Search
            elif cfg.search_schedule == 'stagnation':
                from .stagnation_search import StagnationSearch as Search
            else:
                from .evaluation_search import Search
            context = Search(self, cfg, X, Y, X_val, Y_val, features, targets, constants, extra,
                input_bounds, X_domain, callback, checkpoint_path, resume, stop_requested).run()
            evaluator = context['evaluator']
            policy = context['policy']
            start = context['start']
            cpu_start = context['cpu_start']
            stage_limit = context['stage_limit']
            reference = context['reference']
            reference_system = context['reference_system']
            reference_constants = context['reference_constants']
            pool = context['pool']
            closure = context['closure']
            archive = context['archive']
            seeds = context['seeds']
            seed_stats = context['seed_stats']
            operations = context['operations']
            changed = context['changed']
            novel = context['novel']
            successful = context['successful']
        else:
            if checkpoint_path is not None or resume or stop_requested is not None:
                raise ValueError('Recoverable checkpoints require evaluation-budget mode.')
            rng = np.random.default_rng(cfg.random_state)
            deadline = start + cfg.wall_seconds if cfg.wall_seconds is not None else None
            base_deadline = start + cfg.wall_seconds * .6 if cfg.accuracy_search else deadline
            exploration_deadline = (start + cfg.wall_seconds * (1 - cfg.selection_budget_fraction if cfg.use_gp else 1)
                                    if cfg.time_allocation=='phased' else None)
            if cfg.accuracy_search and exploration_deadline is not None:
                exploration_deadline = start + (base_deadline-start) * (1-cfg.selection_budget_fraction if cfg.use_gp else 1)
            grammar = SystemGrammar(features, targets, constants, cfg, extra)
            evaluator = Evaluator(X, Y, cfg, input_bounds=input_bounds, X_domain=X_domain, deadline=base_deadline)
            policy = Policy(grammar.n_tokens, cfg.hidden_size, cfg.learning_rate, cfg.random_state,
                            grammar.context_size) if cfg.use_rnn else None
            self.policy_, self.history_ = policy, []
            operations, changed, novel, successful = Counter(), Counter(), Counter(), Counter()
            previous_calls, previous_time = 0, time.monotonic()
            reference = None
            stage_limit = int(cfg.max_evaluations * (1 - cfg.selection_budget_fraction)) if cfg.use_gp else cfg.max_evaluations

            def record(phase, step, population, **update):
                nonlocal previous_calls, previous_time
                now = time.monotonic()
                entry = dict(phase=phase, step=step, evaluations=evaluator.calls,
                             new_evaluations=evaluator.calls - previous_calls,
                             valid_systems=sum(c is not None for c in evaluator.cache.values()),
                             population=len(population), unique_sizes=len({c.system.size for c in population}),
                             best_train_mean_nrmse=min((c.error for c in population), default=None),
                             best_train_search_score=min((search_score(c.errors, cfg) for c in population), default=None),
                             phase_seconds=now - previous_time, elapsed_seconds=now - start,
                             constant_nfev=evaluator.stats['constant_nfev'], **update)
                self.history_.append(entry)
                if cfg.acceptance is not None:
                    entry['training_accuracy_met'] = any(bool(np.all(c.errors <= np.asarray(cfg.acceptance['nrmse']))) for c in population)
                previous_calls, previous_time = evaluator.calls, now
                if callback:
                    callback(entry.copy())

            def phase_expired(time_limit):
                return evaluator.expired() or (time_limit is not None and time.monotonic()>=time_limit)

            def evolve(population, multiobjective, limit, time_limit=None):
                ranked = ordered(population, cfg, reference) if multiobjective else sorted(population, key=lambda c: (violations(c.errors, reference, cfg), search_score(c.errors, cfg)))
                children = []
                for _ in range(cfg.population_size):
                    if evaluator.calls >= limit or phase_expired(time_limit):
                        break
                    for _ in range(cfg.variation_attempts):
                        if time_limit is not None and phase_expired(time_limit):
                            break
                        parent = ranked[int(rng.integers(len(ranked), size=3).min())]
                        donor = ranked[int(rng.integers(len(ranked), size=3).min())]
                        system, warm, op = offspring(parent, donor, grammar, rng)
                        operations[op] += 1
                        changed[op] += int(system != parent.system)
                        if system in evaluator.cache or not in_bounds(system, cfg):
                            continue
                        novel[op] += 1
                        candidate = evaluator(system, warm)
                        if candidate is not None:
                            children.append(candidate)
                            successful[op] += 1
                        break
                merged = list({c.system: c for c in population + children}.values())
                survivors = (retain_parameters(ordered(merged, cfg, reference), cfg.population_size, cfg, reference) if multiobjective else
                             diverse(ordered(merged, cfg, reference), cfg.population_size, cfg, reference))
                return survivors, children, len(merged)

            # Training-only typed/structured seeds, combined as complete systems.
            typed_generator,structure_generator=typed_seeds,structured_seeds
            if cfg.initialization in ('dimensional','compound','implicit'):
                from .seeds import generator
                from .structure import structured_seeds as dimensioned_structures
                typed_generator,structure_generator=generator(cfg,q),dimensioned_structures
                if cfg.initialization in ('compound','implicit'):
                    from functools import partial
                    structure_generator=partial(dimensioned_structures,compound=True,implicit=cfg.initialization=='implicit')
            seed_lists, seed_stats, implicit_slots = [], [], []
            for j, g in enumerate(grammar.plain):
                seeds = [(t, np.ones(t.n_constants)) for t in typed_generator(g, rng)] if cfg.linear_seeds else []
                slots = set()
                if cfg.structured_seeds:
                    if cfg.accuracy_search:
                        try:
                            ss, stats = structure_generator(g, X, Y[:, j], max(1, cfg.population_size // q),
                                deadline=min(base_deadline, time.monotonic()+cfg.wall_seconds*.15/q))
                        except TimeoutError as error:
                            ss, stats = [], dict(error.initialization_diagnostics, timed_out=True)
                    else:
                        ss, stats = structure_generator(g, X, Y[:, j], max(1, cfg.population_size // q))
                    slots = {len(seeds)+i for i in stats.get('implicit_retained_indices', ())}
                    seeds.extend(ss)
                    seed_stats.append(stats)
                if not seeds:
                    t, _ = g.generate(rng)
                    seeds = [(t, np.ones(t.n_constants))]
                seed_lists.append(seeds)
                implicit_slots.append(slots)
            population = []
            implicit_jobs = dict(implicit_seed_jobs=0, implicit_seed_new_evaluations=0, implicit_seed_valid=0)
            init_limit = max(1, int(cfg.max_evaluations * cfg.initialization_fraction))
            for k in range(min(init_limit, max(map(len, seed_lists)))):
                if evaluator.expired():
                    break
                b = Builder()
                for j in range(q):
                    t, c = seed_lists[j][k % len(seed_lists[j])]
                    b.roots.append(b.tree(t, c))
                before = evaluator.calls
                candidate = evaluator(*b.finish())
                if cfg.initialization=='implicit' and any(k%len(seed_lists[j]) in implicit_slots[j] for j in range(q)):
                    implicit_jobs['implicit_seed_jobs'] += 1
                    implicit_jobs['implicit_seed_new_evaluations'] += evaluator.calls-before
                    implicit_jobs['implicit_seed_valid'] += int(candidate is not None and evaluator.calls>before)
                if candidate is not None:
                    population.append(candidate)
            if not population:
                for _ in range(cfg.population_size):
                    system, warm, _ = grammar.generate(rng)
                    c = evaluator(system, warm)
                    if c is not None:
                        population.append(c)
                    if evaluator.expired():
                        break
            if not population:
                raise RuntimeError('No valid initialization system within the budget/domain.')
            # Assemble and freeze per-output initial reference roots without cross-root binding.
            b = Builder()
            for j in range(q):
                c = min(population, key=lambda c: c.errors[j])
                b.roots.append(clone_subgraph(b, c.system, c.constants, c.system.roots[j]))
            reference_system, reference_constants = b.finish()
            reference = np.sqrt(np.mean(((reference_system.evaluate(X, reference_constants) - Y) / evaluator.scales) ** 2, axis=0))
            ref_rad, ref_scaled = (0., 0.)
            ref_candidate = evaluator(reference_system, reference_constants)
            if ref_candidate is not None:
                population.append(ref_candidate)
            population = diverse(population, cfg.population_size, cfg, reference)
            record('initialization', 0, population, reference=reference.tolist(),
                   **(implicit_jobs if cfg.initialization=='implicit' else {}))

            for epoch in range(cfg.epochs):
                if cfg.accuracy_search and epoch:
                    cfg.max_nodes += 16
                    cfg.max_depth += 1
                    grammar = SystemGrammar(features, targets, constants, cfg, extra)
                    record('capacity_growth', epoch, population, max_nodes=cfg.max_nodes, max_depth=cfg.max_depth)
                neural_deadline=gp_deadline=None
                if exploration_deadline is not None:
                    epoch_start=time.monotonic()
                    if epoch_start>=exploration_deadline:
                        break
                    window=(exploration_deadline-epoch_start)/(cfg.epochs-epoch)
                    neural_deadline=epoch_start+window*(.5 if cfg.use_gp else .85)
                    gp_deadline=epoch_start+window*.85
                    record('phase_budget',epoch,population,epoch_allowance_seconds=window,
                           neural_until_seconds=neural_deadline-start,gp_until_seconds=gp_deadline-start,
                           exploration_until_seconds=exploration_deadline-start)
                traces, rewards, sampled = [], [], []
                before = evaluator.calls
                for _ in range(cfg.population_size):
                    if evaluator.calls >= stage_limit or phase_expired(neural_deadline):
                        break
                    system, warm, trace = grammar.generate(rng, policy)
                    candidate = evaluator(system, warm)
                    traces.append(trace)
                    rewards.append(0. if candidate is None else 1 / (1 + search_score(candidate.errors, cfg) + violations(candidate.errors, reference, cfg)))
                    if candidate is not None:
                        sampled.append(candidate)
                population = diverse(population + sampled, cfg.population_size, cfg, reference)
                record('neural' if policy else 'random', epoch, population, proposals=len(traces),
                       neural_valid=len(sampled) if policy else 0, unique=evaluator.calls - before,
                       sampled_shared=sum(c.system.sharing['shared_nodes'] > 0 or c.system.sharing['shared_parameters'] > 0 for c in sampled))
                if cfg.use_gp:
                    for generation in range(cfg.gp_generations):
                        population, children, merged = evolve(population, False, stage_limit,gp_deadline)
                        record('gp', epoch * cfg.gp_generations + generation, population,
                               valid_offspring=len(children), merged_population=merged)
                if policy and traces:
                    elites, skipped = [], 0
                    for c in diverse([c for c in evaluator.cache.values() if c is not None],
                                     max(1, int(cfg.population_size * cfg.risk_fraction)), cfg, reference):
                        try:
                            elites.append(grammar.trace(c.system))
                        except (ValueError, KeyError):
                            skipped += 1
                    before_params = torch.cat([p.detach().flatten().clone() for p in policy.parameters()])
                    update = policy.update(traces, rewards, elites, cfg.risk_fraction,
                                           cfg.entropy_weight, cfg.imitation_weight)
                    delta = float(torch.linalg.vector_norm(torch.cat([p.detach().flatten() for p in policy.parameters()]) - before_params))
                    record('policy_update', epoch, population, parameter_delta=delta,
                           imitation_systems=len(elites), imitation_unrepresentable=skipped, **update)
                if evaluator.calls >= stage_limit or phase_expired(exploration_deadline):
                    break

            archive = [c for c in evaluator.cache.values() if c is not None]
            count = max(cfg.min_candidates, int(np.ceil(cfg.candidate_fraction * len(archive))))
            seeds = list({c.system: c for c in diverse(archive, max(1, count // 2), cfg, reference) + ordered(archive, cfg, reference)}.values())[:count]
            if len(seeds) < cfg.min_candidates:
                warnings.warn(f'Only {len(seeds)} valid unique systems; requested {cfg.min_candidates}.', UserWarning)
            population, pool = retain_parameters(ordered(seeds, cfg, reference), cfg.population_size, cfg, reference), list(seeds)
            if cfg.use_gp:
                for generation in range(cfg.selection_generations):
                    population, children, merged = evolve(population, True, cfg.max_evaluations)
                    pool.extend(children)
                    record('nsga2', generation, population, valid_offspring=len(children), merged_population=merged,
                           nondominated_fronts=len(fronts([(search_score(c.errors, cfg), c.complexity) for c in population])))
                    if evaluator.expired() or evaluator.calls >= cfg.max_evaluations:
                        break
            for c in diverse(pool, cfg.polish_candidates, cfg, reference) if cfg.polish_candidates else []:
                if evaluator.expired():
                    break
                old_max = cfg.constant_max_nfev
                cfg.constant_max_nfev = cfg.polish_max_nfev
                evaluator(c.system, c.constants, refine=True)
                cfg.constant_max_nfev = old_max
            pool = list({c.system: evaluator.cache[c.system] for c in pool}.values())
            record('polish', 0, pool)
            evaluator.deadline = deadline
            closure = None
            if cfg.sharing_strategy=='closure':
                from .sharing import protected_closure
                pool, closure = protected_closure(pool, evaluator, **({'grammar':grammar} if cfg.precision_repair or cfg.compositional_completion or cfg.accuracy_search else {}))
                repair = closure.get('precision_repair', {})
                completion = closure.get('compositional_completion', {})
                record('sharing_closure', 0, pool, new_systems=closure['new_evaluations'], protected_systems=closure['protected'],
                       **(dict(repair_new_systems=repair['new_evaluations'], repair_protected_systems=repair['protected']) if repair else {}),
                       **(dict(completion_new_systems=completion['new_evaluations'], completion_protected_systems=completion['protected']) if completion else {}))
        # Validation may rank ONLY this frozen pool; it cannot affect later search.
        if pareto_mode:
            pool = (list(evaluator.archive.values()) if cfg.search_schedule in ('stagnation', 'portfolio') else
                    [c for c in evaluator.cache.values() if c is not None])
            if cfg.fixed_training_reference is not None:
                pool = [c for c in pool if violations(c.errors, reference, cfg) == 0]
                if not pool:
                    raise RuntimeError('No candidate satisfies the fixed training protection.')
        frozen_payload = [c.to_dict() for c in pool]
        frozen_hash = hashlib.sha256(json.dumps(frozen_payload, sort_keys=True).encode()).hexdigest()
        ref_val = None if pareto_mode else np.sqrt(np.mean(((reference_system.evaluate(X_val, reference_constants) - Y_val) / evaluator.scales) ** 2, axis=0))
        if ref_val is not None and not np.isfinite(ref_val).all():
            raise RuntimeError('Frozen initialization reference is invalid on validation inputs.')
        scored, valid_base_count = [], 0
        for pool_index, c in enumerate(pool):
            pred = predict_system(c.system, c.constants, X_val, cfg)
            valid = evaluator.admissible(pred)
            if valid:
                with np.errstate(over='ignore', invalid='ignore'):
                    err = np.sqrt(np.mean(((pred - Y_val) / evaluator.scales) ** 2, axis=0))
                violation = 0.0 if pareto_mode else violations(err, ref_val, cfg)
                score = (search_score(err, cfg) if pareto_mode else
                         float(err.mean() + cfg.selection_complexity_weight * c.complexity / (q * cfg.max_nodes)))
                if closure is not None:
                    valid = bool(np.isfinite(err).all() and np.isfinite(violation) and np.isfinite(score))
            if valid:
                scored.append((violation, score, c.complexity, c, err))
                valid_base_count += int(closure is not None and pool_index < closure['base_candidate_count'])
            elif closure is not None:
                if pool_index < closure['base_candidate_count'] and not pareto_mode:
                    raise RuntimeError('A base pool candidate is invalid on validation inputs.')
                from .selection import digest
                closure.setdefault('validation_rejections', []).append(dict(graph_sha256=digest(c.system.to_dict()),
                    reason='Validation predictions or error metrics fail finite/output bounds.'))
        if closure is not None:
            # Validation may reject new hypotheses, never restart their search.
            closure['full_training_pool_sha256'] = frozen_hash
            if pareto_mode:
                closure['validation_retained_candidates'] = len(scored)
            else:
                closure['validation_retained_extensions'] = len(scored)-valid_base_count
            frozen_hash = hashlib.sha256(json.dumps([c.to_dict() for _, _, _, c, _ in scored], sort_keys=True).encode()).hexdigest()
        if not scored:
            raise RuntimeError('All frozen systems are invalid on validation inputs.')
        selected = min(range(len(scored)), key=lambda i: scored[i][:3])
        v, score, _, best, errors = scored[selected]
        self.best_, self.config_ = best, cfg
        self.n_features_in_, self.n_outputs_ = p, q
        self.feature_names_, self.feature_units_, self.target_units_, self.constant_units_ = names, features, targets, constants
        self.scales_, self.validation_nrmse_, self.reference_train_, self.reference_validation_ = evaluator.scales, errors, reference, ref_val
        self.formulas_ = best.system.formulas(best.constants, names)
        self.diagnostics_ = dict(evaluations=evaluator.calls, **dict(evaluator.stats),
            elapsed_seconds=time.monotonic() - start, cpu_seconds=time.process_time() - cpu_start,
            candidate_count=len(scored) if pareto_mode else len(seeds), candidate_minimum_met=len(seeds) >= cfg.min_candidates,
            valid_systems=len(pool) if pareto_mode else len(archive), frozen_candidate_sha256=frozen_hash,
            validation_feasible=v == 0, validation_violation=v, selected_score=score,
            operations=dict(operations), changed=dict(changed), novel=dict(novel), successful=dict(successful),
            deadline_reached=evaluator.expired() if cfg.budget_mode == 'legacy' else False, stage_one_limit=stage_limit,
            domain=evaluator.domain, interval_checks=cfg.interval_checks,
            initialization_diagnostics=seed_stats, sharing=best.system.sharing,
            training_rows=len(X), validation_rows=len(X_val),
            data_sha256=dict(training=data_digest(X,Y),validation=data_digest(X_val,Y_val)))
        if cfg.budget_mode == 'evaluations':
            self.diagnostics_['evaluation_budget'] = self.budget_report_
        if pareto_mode:
            self.diagnostics_['cpu_seconds_scope'] = 'coordinator_process'
        if closure is not None:
            self.diagnostics_['sharing_closure'] = closure
        self.candidates_ = [dict(c.to_dict(), validation_errors=e.tolist(), violation=vi, score=sc)
                            for vi, sc, _, c, e in scored]
        self.search_fingerprint_ = hashlib.sha256(json.dumps(
            [c.to_dict() for c in evaluator.cache.values() if c is not None], sort_keys=True).encode()).hexdigest()
        if pareto_mode:
            self._apply_pareto_selection()
        elif cfg.final_selection!='weighted':
            from .selection import derive
            from .acceptance import select
            source=self.to_dict()
            selected=select(source,cfg.acceptance) if cfg.final_selection=='precision' else derive(source,'guarded')
            best=selected['best']
            self.best_=Candidate(System.from_dict(best['system']),np.asarray(best['constants']),np.asarray(best['errors']),
                best['complexity'],best['rademacher'],best['rademacher_complexity'])
            self.validation_nrmse_=np.asarray(selected['validation_nrmse'])
            self.diagnostics_=selected['diagnostics'];self.post_selection_=selected['post_selection']
            self.formulas_=self.best_.system.formulas(self.best_.constants,names)
        return self

    @staticmethod
    def _candidate(record):
        return Candidate(System.from_dict(record['system']), np.asarray(record['constants']),
                         np.asarray(record['errors']), record['complexity'], record['rademacher'],
                         record['rademacher_complexity'])

    def _apply_pareto_selection(self, expected_report=None, expected_best=None):
        from .pareto_selection import select_pareto
        from .selection import digest, FIELDS
        if digest([{k: c[k] for k in FIELDS} for c in self.candidates_]) != self.diagnostics_['frozen_candidate_sha256']:
            raise ValueError('Frozen candidate pool identity does not match.')
        report = select_pareto(self.candidates_, self.config_.complexity_weights)
        report['scored_pool_sha256'] = digest(self.candidates_)
        record = self.candidates_[report['recommended_index']]
        if expected_report is not None and expected_report != report:
            raise ValueError('Saved Pareto selection record does not match the frozen pool.')
        if expected_best is not None and expected_best != {k: record[k] for k in FIELDS}:
            raise ValueError('Saved recommended model does not match Pareto selection.')
        self.best_ = self._candidate(record)
        self.recommended_ = self.best_
        accuracy = self.candidates_[report['best_accuracy_index']]
        self.best_accuracy_ = self._candidate(accuracy)
        self.best_accuracy_validation_nrmse_ = np.asarray(accuracy['validation_errors'])
        self.validation_nrmse_ = np.asarray(record['validation_errors'])
        self.selection_report_ = self.post_selection_ = report
        metrics = {entry['candidate_index']: entry for entry in report['candidate_metrics']}
        self.pareto_front_ = [deepcopy(metrics[index]) for index in report['pareto_indices']]
        self.diagnostics_.pop('validation_feasible', None)
        self.diagnostics_.pop('validation_violation', None)
        self.diagnostics_.update(selection_policy='accuracy_pareto', selected_score=float(self.validation_nrmse_.mean()),
                                 sharing=self.best_.system.sharing)
        self.formulas_ = self.best_.system.formulas(self.best_.constants, self.feature_names_)

    def _prediction_candidate(self, selection):
        if selection == 'recommended':
            return self.best_
        if selection == 'best_accuracy' and hasattr(self, 'best_accuracy_'):
            return self.best_accuracy_
        if isinstance(selection, (int, np.integer)) and not isinstance(selection, (bool, np.bool_)) and 0 <= selection < len(self.candidates_):
            return self._candidate(self.candidates_[selection])
        raise ValueError('selection must be recommended, best_accuracy in Pareto mode, or a frozen candidate index.')

    def formulas_for(self, selection='recommended'):
        check_is_fitted(self, 'best_')
        candidate = self._prediction_candidate(selection)
        return candidate.system.formulas(candidate.constants, self.feature_names_)

    def predict(self, X, *, selection='recommended'):
        check_is_fitted(self, 'best_')
        X = check_array(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError('Feature count mismatch.')
        candidate = self._prediction_candidate(selection)
        y = predict_system(candidate.system, candidate.constants, X, self.config_)
        if not np.isfinite(y).all():
            raise ValueError('Raw formula is undefined on these inputs.')
        if self.config_.output_bounds is not None:
            lo, hi = self.config_.output_bounds
            if (lo is not None and np.any(y < lo)) or (hi is not None and np.any(y > hi)):
                raise ValueError('Output bounds violated.')
        return y

    def audit_domain(self, input_bounds, max_boxes=128):
        check_is_fitted(self, 'best_')
        bounds = np.asarray(input_bounds, float)
        if bounds.shape != (self.n_features_in_, 2):
            raise ValueError('Input bounds must match the features.')
        audits = self.best_.system.audit(self.best_.constants, bounds, max_boxes, self.config_.output_bounds)
        status = 'invalid' if any(a['status'] == 'invalid' for a in audits) else 'unresolved' if any(a['status'] == 'unresolved' for a in audits) else 'enclosed'
        return dict(status=status, outputs=audits, guarantee='numerical screening, not a formal proof')

    def to_dict(self):
        check_is_fitted(self, 'best_')
        data = dict(format='dimensr-mimo-dag', schema_version=1, package_version=PACKAGE_VERSION,
                    config=asdict(self.config_), n_features=self.n_features_in_, n_outputs=self.n_outputs_,
                    feature_names=self.feature_names_, feature_units=[encode(u) for u in self.feature_units_],
                    target_units=[encode(u) for u in self.target_units_], constant_units=[encode(u) for u in self.constant_units_],
                    best=self.best_.to_dict(), scales=self.scales_.tolist(), validation_nrmse=self.validation_nrmse_.tolist(),
                    reference_train=self.reference_train_.tolist(),
                    reference_validation=None if self.reference_validation_ is None else self.reference_validation_.tolist(),
                    history=self.history_, diagnostics=self.diagnostics_, candidates=self.candidates_,
                    search_fingerprint=self.search_fingerprint_)
        if hasattr(self,'post_selection_'):data['post_selection']=self.post_selection_
        return deepcopy(data)

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text(encoding='utf-8'))
        if d.get('format') != 'dimensr-mimo-dag' or d.get('schema_version') != 1:
            raise ValueError('Unsupported model schema.')
        cfg = SearchConfig(**d['config'])
        cfg.validate()
        model = cls(cfg, d['feature_units'], d['target_units'], d['constant_units'], d['feature_names'])
        model.config_ = cfg
        model.n_features_in_, model.n_outputs_ = d['n_features'], d['n_outputs']
        model.feature_units_, model.target_units_, model.constant_units_ = [tuple(unit(u) for u in d[k]) for k in ('feature_units', 'target_units', 'constant_units')]
        model.feature_names_ = d['feature_names']
        b = d['best']
        system, constants = System.from_dict(b['system']), np.asarray(b['constants'], float)
        if (constants.shape != (system.n_parameters,) or not np.isfinite(constants).all() or
                len(system.roots) != model.n_outputs_ or len(model.feature_units_) != model.n_features_in_ or
                len(model.feature_names_) != model.n_features_in_ or len(model.target_units_) != model.n_outputs_ or
                not in_bounds(system, cfg)):
            raise ValueError('Inconsistent model dimensions or parameters.')
        for j, r in enumerate(system.roots):
            if system.nodes[r].units != model.target_units_[j]:
                raise ValueError('Output units mismatch.')
        for v in system.nodes:
            if not v.children:
                units = model.feature_units_ if v.token.startswith('x') else model.constant_units_
                i = int(v.token[1:])
                if i >= len(units) or v.units != units[i]:
                    raise ValueError('Terminal metadata mismatch.')
        model.best_ = Candidate(system, constants, np.asarray(b['errors']), b['complexity'], b['rademacher'], b['rademacher_complexity'])
        model.scales_, model.validation_nrmse_, model.reference_train_ = [np.asarray(d[k], float) for k in ('scales', 'validation_nrmse', 'reference_train')]
        model.reference_validation_ = None if d['reference_validation'] is None else np.asarray(d['reference_validation'], float)
        target_vector(cfg.stop_metric, cfg.stop_target, model.n_outputs_)
        if model.scales_.shape != (model.n_outputs_,) or not np.isfinite(model.scales_).all() or np.any(model.scales_ <= 0):
            raise ValueError('Invalid output normalization.')
        model.history_, model.diagnostics_, model.candidates_ = d['history'], d['diagnostics'], d['candidates']
        model.search_fingerprint_ = d['search_fingerprint']
        model.formulas_ = system.formulas(constants, model.feature_names_)
        if 'post_selection' in d:model.post_selection_=d['post_selection']
        if cfg.final_selection == 'pareto':
            if 'post_selection' not in d:
                raise ValueError('Pareto models require a frozen selection record.')
            index = d['post_selection']['recommended_index']
            if d['validation_nrmse'] != d['candidates'][index]['validation_errors']:
                raise ValueError('Saved recommended validation metrics do not match the frozen pool.')
            model._apply_pareto_selection(d['post_selection'], d['best'])
            model.budget_report_ = model.diagnostics_['evaluation_budget']
            if cfg.search_schedule == 'portfolio':
                model.portfolio_report_ = deepcopy(model.budget_report_['portfolio'])
        elif cfg.final_selection!='weighted':
            from .selection import choose,derive,FIELDS
            from .acceptance import select
            original=deepcopy(d);original.pop('post_selection',None)
            original['diagnostics'].pop('application_acceptance',None)
            anchor=original['candidates'][choose(original['candidates'],'original')['candidate_index']]
            original['best']={k:anchor[k] for k in FIELDS}
            original['validation_nrmse']=anchor['validation_errors']
            original['diagnostics'].update(validation_feasible=anchor['violation']==0,validation_violation=anchor['violation'],
                selected_score=anchor['score'],sharing=System.from_dict(anchor['system']).sharing)
            expected=select(original,cfg.acceptance) if cfg.final_selection=='precision' else derive(original,'guarded')
            if expected['best']!=d['best'] or expected['post_selection']!=d.get('post_selection') or expected['diagnostics']!=d['diagnostics']:
                raise ValueError('Saved selection or acceptance record does not match the frozen pool.')
            # Verify the saved record first, then rebase provenance onto the new
            # serialized defaults/version so old native models can be resaved.
            if original['config']!=asdict(cfg) or original.get('package_version')!=PACKAGE_VERSION:
                original.update(config=asdict(cfg),package_version=PACKAGE_VERSION)
                migrated=select(original,cfg.acceptance) if cfg.final_selection=='precision' else derive(original,'guarded')
                model.diagnostics_=migrated['diagnostics'];model.post_selection_=migrated['post_selection']
        return model
