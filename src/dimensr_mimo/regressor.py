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
from .evaluation import Evaluator, Candidate
from .pareto import ordered, diverse, violations

PACKAGE_VERSION='1.3.0'


def arrays(X, Y):
    X, Y = check_array(X, dtype=float), check_array(Y, dtype=float)
    if len(X) != len(Y):
        raise ValueError('X and Y must have the same row count; Y is a 2D output matrix.')
    return X, Y


class DimenSRMIMORegressor(RegressorMixin, BaseEstimator):
    def __init__(self, config=None, feature_units=None, target_units=None,
                 constant_units=None, feature_names=None, extra_units=None):
        self.config, self.feature_units, self.target_units = config, feature_units, target_units
        self.constant_units, self.feature_names, self.extra_units = constant_units, feature_names, extra_units

    def fit(self, X, Y, *, X_val=None, Y_val=None, input_bounds=None, X_domain=None, callback=None):
        for name in list(vars(self)):
            if name.endswith('_'):
                delattr(self, name)
        start, cpu_start = time.monotonic(), time.process_time()
        cfg = replace(self.config) if self.config is not None else SearchConfig()
        cfg.acceptance = deepcopy(cfg.acceptance)
        cfg.validate()
        X, Y = arrays(X, Y)
        if len(X) < 5:
            raise ValueError('At least five rows are required.')
        p, q = X.shape[1], Y.shape[1]
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
        rng = np.random.default_rng(cfg.random_state)
        deadline = start + cfg.wall_seconds if cfg.wall_seconds is not None else None
        exploration_deadline = (start + cfg.wall_seconds * (1 - cfg.selection_budget_fraction if cfg.use_gp else 1)
                                if cfg.time_allocation=='phased' else None)
        grammar = SystemGrammar(features, targets, constants, cfg, extra)
        evaluator = Evaluator(X, Y, cfg, input_bounds=input_bounds, X_domain=X_domain, deadline=deadline)
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
                         phase_seconds=now - previous_time, elapsed_seconds=now - start,
                         constant_nfev=evaluator.stats['constant_nfev'], **update)
            self.history_.append(entry)
            previous_calls, previous_time = evaluator.calls, now
            if callback:
                callback(entry.copy())

        def phase_expired(time_limit):
            return evaluator.expired() or (time_limit is not None and time.monotonic()>=time_limit)

        def evolve(population, multiobjective, limit, time_limit=None):
            ranked = ordered(population, cfg, reference) if multiobjective else sorted(population, key=lambda c: (violations(c.errors, reference, cfg), c.error))
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
            survivors = (ordered(merged, cfg, reference)[:cfg.population_size] if multiobjective else
                         diverse(ordered(merged, cfg, reference), cfg.population_size, cfg))
            return survivors, children, len(merged)

        # Training-only typed/structured seeds, combined as complete systems.
        typed_generator,structure_generator=typed_seeds,structured_seeds
        if cfg.initialization=='dimensional':
            from .seeds import generator
            from .structure import structured_seeds as dimensioned_structures
            typed_generator,structure_generator=generator(cfg,q),dimensioned_structures
        seed_lists, seed_stats = [], []
        for j, g in enumerate(grammar.plain):
            seeds = [(t, np.ones(t.n_constants)) for t in typed_generator(g, rng)] if cfg.linear_seeds else []
            if cfg.structured_seeds:
                ss, stats = structure_generator(g, X, Y[:, j], max(1, cfg.population_size // q))
                seeds.extend(ss)
                seed_stats.append(stats)
            if not seeds:
                t, _ = g.generate(rng)
                seeds = [(t, np.ones(t.n_constants))]
            seed_lists.append(seeds)
        population = []
        init_limit = max(1, int(cfg.max_evaluations * cfg.initialization_fraction))
        for k in range(min(init_limit, max(map(len, seed_lists)))):
            if evaluator.expired():
                break
            b = Builder()
            for j in range(q):
                t, c = seed_lists[j][k % len(seed_lists[j])]
                b.roots.append(b.tree(t, c))
            candidate = evaluator(*b.finish())
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
        population = diverse(population, cfg.population_size, cfg)
        record('initialization', 0, population, reference=reference.tolist())

        for epoch in range(cfg.epochs):
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
                rewards.append(0. if candidate is None else 1 / (1 + candidate.error + violations(candidate.errors, reference, cfg)))
                if candidate is not None:
                    sampled.append(candidate)
            population = diverse(population + sampled, cfg.population_size, cfg)
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
                                 max(1, int(cfg.population_size * cfg.risk_fraction)), cfg):
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
        seeds = list({c.system: c for c in diverse(archive, max(1, count // 2), cfg) + ordered(archive, cfg, reference)}.values())[:count]
        if len(seeds) < cfg.min_candidates:
            warnings.warn(f'Only {len(seeds)} valid unique systems; requested {cfg.min_candidates}.', UserWarning)
        population, pool = ordered(seeds, cfg, reference)[:cfg.population_size], list(seeds)
        if cfg.use_gp:
            for generation in range(cfg.selection_generations):
                population, children, merged = evolve(population, True, cfg.max_evaluations)
                pool.extend(children)
                record('nsga2', generation, population, valid_offspring=len(children), merged_population=merged,
                       nondominated_fronts=len(fronts([(c.error, c.complexity) for c in population])))
                if evaluator.expired() or evaluator.calls >= cfg.max_evaluations:
                    break
        for c in diverse(pool, cfg.polish_candidates, cfg) if cfg.polish_candidates else []:
            if evaluator.expired():
                break
            old_max = cfg.constant_max_nfev
            cfg.constant_max_nfev = cfg.polish_max_nfev
            evaluator(c.system, c.constants, refine=True)
            cfg.constant_max_nfev = old_max
        pool = list({c.system: evaluator.cache[c.system] for c in pool}.values())
        record('polish', 0, pool)
        # Validation may rank ONLY this frozen pool; it cannot affect later search.
        frozen_payload = [c.to_dict() for c in pool]
        frozen_hash = hashlib.sha256(json.dumps(frozen_payload, sort_keys=True).encode()).hexdigest()
        ref_val = np.sqrt(np.mean(((reference_system.evaluate(X_val, reference_constants) - Y_val) / evaluator.scales) ** 2, axis=0))
        if not np.isfinite(ref_val).all():
            raise RuntimeError('Frozen initialization reference is invalid on validation inputs.')
        scored = []
        for c in pool:
            pred = c.system.evaluate(X_val, c.constants)
            if evaluator.admissible(pred):
                err = np.sqrt(np.mean(((pred - Y_val) / evaluator.scales) ** 2, axis=0))
                violation = violations(err, ref_val, cfg)
                score = float(err.mean() + cfg.selection_complexity_weight * c.complexity / (q * cfg.max_nodes))
                scored.append((violation, score, c.complexity, c, err))
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
            candidate_count=len(seeds), candidate_minimum_met=len(seeds) >= cfg.min_candidates,
            valid_systems=len(archive), frozen_candidate_sha256=frozen_hash,
            validation_feasible=v == 0, validation_violation=v, selected_score=score,
            operations=dict(operations), changed=dict(changed), novel=dict(novel), successful=dict(successful),
            deadline_reached=evaluator.expired(), stage_one_limit=stage_limit,
            domain=evaluator.domain, interval_checks=cfg.interval_checks,
            initialization_diagnostics=seed_stats, sharing=best.system.sharing,
            training_rows=len(X), validation_rows=len(X_val))
        self.candidates_ = [dict(c.to_dict(), validation_errors=e.tolist(), violation=vi, score=sc)
                            for vi, sc, _, c, e in scored]
        self.search_fingerprint_ = hashlib.sha256(json.dumps(
            [c.to_dict() for c in evaluator.cache.values() if c is not None], sort_keys=True).encode()).hexdigest()
        if cfg.final_selection!='weighted':
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

    def predict(self, X):
        check_is_fitted(self, 'best_')
        X = check_array(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError('Feature count mismatch.')
        y = self.best_.system.evaluate(X, self.best_.constants)
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
                    reference_train=self.reference_train_.tolist(), reference_validation=self.reference_validation_.tolist(),
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
        model.scales_, model.validation_nrmse_, model.reference_train_, model.reference_validation_ = [np.asarray(d[k], float) for k in ('scales', 'validation_nrmse', 'reference_train', 'reference_validation')]
        if model.scales_.shape != (model.n_outputs_,) or not np.isfinite(model.scales_).all() or np.any(model.scales_ <= 0):
            raise ValueError('Invalid output normalization.')
        model.history_, model.diagnostics_, model.candidates_ = d['history'], d['diagnostics'], d['candidates']
        model.search_fingerprint_ = d['search_fingerprint']
        model.formulas_ = system.formulas(constants, model.feature_names_)
        if 'post_selection' in d:model.post_selection_=d['post_selection']
        if cfg.final_selection!='weighted':
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
