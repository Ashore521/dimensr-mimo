"""Finite stage scheduling, with no clock-based search decisions."""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from .audit import data_digest
from .checkpoint import Checkpoint, JournalEvaluator, SearchStopped
from .evaluation import search_score
from .generation import SystemGrammar
from .genetics import clone_subgraph, offspring
from .graph import Builder, in_bounds
from .pareto import diverse, ordered, retain_parameters, violations
from .policy import Policy
from .residual import residual_search
from .seeds import generator
from .selection import digest
from .sharing import protected_closure
from .structure import structured_seeds
from .stopping import target_vector, observe, completed

STAGES = ('initialization', 'exploration', 'nsga2', 'residual', 'sharing')


class Search:
    evaluator_type = JournalEvaluator

    def __init__(self, owner, cfg, X, Y, X_val, Y_val, features, targets, constants, extra,
                 input_bounds, X_domain, callback, checkpoint_path, resume, stop_requested):
        self.owner, self.cfg, self.X, self.Y = owner, cfg, X, Y
        self.pareto_mode = cfg.final_selection == 'pareto'
        self.variance = np.var(Y, axis=0)
        self.original_threads = torch.get_num_threads()
        self.units = features, targets, constants, extra
        self.callback, self.stop_requested = callback, stop_requested
        package = Path(__file__).parent
        binding = dict(schema=1, config=asdict(cfg), training=data_digest(X, Y),
                       validation=data_digest(X_val, Y_val), units=repr(self.units),
                       bounds=None if input_bounds is None else np.asarray(input_bounds).tolist(),
                       domain=None if X_domain is None else data_digest(np.asarray(X_domain), np.empty((len(X_domain), 0))),
                       source={str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(package.rglob('*.py'))})
        self.store = Checkpoint(checkpoint_path, binding, resume)
        self.s = self.store.load()
        elapsed = self.s['elapsed_seconds'] if self.s is not None else 0.
        last = self.store.db.execute('SELECT payload FROM evaluations ORDER BY id DESC LIMIT 1').fetchone()
        if last is not None:
            elapsed = max(elapsed, self.store.decode(last[0])['elapsed_seconds'])
        self.started = time.monotonic() - elapsed
        self.cpu_started = time.process_time() - (self.s.get('cpu_seconds', 0.) if self.s else 0.)
        if self.s is not None:
            for key, value in self.s['dynamic_config'].items():
                setattr(cfg, key, value)
        self.rng = np.random.default_rng(cfg.random_state)
        self.grammar = self.make_grammar()
        self.policy = Policy(self.grammar.n_tokens, cfg.hidden_size, cfg.learning_rate,
                             cfg.random_state, self.grammar.context_size) if cfg.use_rnn else None
        self.evaluator = self.evaluator_type(X, Y, cfg, input_bounds=input_bounds, X_domain=X_domain,
            deadline=None, checkpoint=self.store, stop_requested=stop_requested, elapsed=self.elapsed)
        if self.s is None:
            targets = target_vector(cfg.stop_metric, cfg.stop_target, Y.shape[1])
            self.s = dict(phase='initialization', step=0, population=[], history=[], reports={}, seed_stats=[],
                          operations=Counter(), changed=Counter(), novel=Counter(), successful=Counter(),
                          attempts=Counter(), reference=None,
                          target=dict(metric=cfg.stop_metric, values=None if targets is None else targets.tolist(),
                              additional_epochs=cfg.stop_after_n_epochs, first_hit_epoch=None, finish_after_epoch=None,
                              completed_epochs=0, partial_epoch=None,
                              trigger='disabled' if targets is None else 'not_reached'))
            self.save()
        else:
            self.rng.bit_generator.state = self.s['rng']
            torch.set_rng_state(self.s['torch_rng'])
            if self.policy is not None:
                self.policy.load_state_dict(self.s['policy'])
                self.policy.optimizer.load_state_dict(self.s['optimizer'])
                self.policy.generator.set_state(self.s['policy_rng'])
            e = self.s['evaluator']
            self.evaluator.cache, self.evaluator.calls = e['cache'], e['calls']
            self.evaluator.stats.update(e['stats'])
            self.evaluator.cursor = e['cursor']
            self.evaluator.restore_execution_state(e.get('execution', {}))
        owner.policy_, owner.history_ = self.policy, self.s['history']

    def make_grammar(self):
        f, t, c, extra = self.units
        return SystemGrammar(f, t, c, self.cfg, extra)

    def elapsed(self):
        return time.monotonic() - self.started

    def check_stop(self):
        if self.stop_requested is not None and self.stop_requested():
            raise SearchStopped('Stop requested during bounded feature work; last safe state is persisted.')

    def observe_target(self, candidates, epoch):
        if observe(self.s['target'], candidates, epoch, self.cfg, self.evaluator.scales, self.variance):
            self.record('training_target_reached', target_epoch=epoch)

    def save(self):
        s, e = self.s, self.evaluator
        s.update(dynamic_config=asdict(self.cfg), rng=self.rng.bit_generator.state,
                 torch_rng=torch.get_rng_state(), elapsed_seconds=self.elapsed(),
                 cpu_seconds=time.process_time()-self.cpu_started,
                 policy=None if self.policy is None else self.policy.state_dict(),
                 policy_rng=None if self.policy is None else self.policy.generator.get_state(),
                 optimizer=None if self.policy is None else self.policy.optimizer.state_dict(),
                 evaluator=dict(cache=e.cache, calls=e.calls, stats=dict(e.stats), cursor=e.cursor,
                                execution=e.execution_state()))
        self.store.save(s)
        if self.stop_requested is not None and self.stop_requested():
            raise SearchStopped('Stop requested at a persisted phase boundary.')

    def enter(self, phase):
        self.evaluator.ceiling = sum(self.cfg.stage_evaluations[k] for k in STAGES[:STAGES.index(phase)+1])
        self.s['reports'].setdefault(phase, dict(allocated=self.cfg.stage_evaluations[phase],
            cumulative_ceiling=self.evaluator.ceiling, start_evaluations=self.evaluator.calls,
            start_attempts=self.s['attempts'][phase], elapsed_start=self.elapsed()))

    def finish(self, phase, reason, following):
        r = self.s['reports'][phase]
        r.update(stop_reason=reason, end_evaluations=self.evaluator.calls,
                 evaluations=self.evaluator.calls-r['start_evaluations'],
                 attempts=self.s['attempts'][phase]-r['start_attempts'],
                 unused_cumulative_allowance=self.evaluator.ceiling-self.evaluator.calls,
                 elapsed_seconds=self.elapsed()-r['elapsed_start'])
        self.record(phase, stop_reason=reason)
        self.s['phase'], self.s['step'] = following, 0
        self.save()

    def record(self, phase, **details):
        e, s = self.evaluator, self.s
        population = s['population']
        entry = dict(phase=phase, step=s['step'], evaluations=e.calls,
                     elapsed_seconds=self.elapsed(), population=len(population),
                     best_train_mean_nrmse=min((c.error for c in population), default=None),
                     best_train_search_score=min((search_score(c.errors, self.cfg) for c in population), default=None),
                     constant_nfev=e.stats['constant_nfev'], refinements=e.stats['refinements'],
                     training_accuracy_met=(any(bool(np.all(c.errors <= self.cfg.acceptance['nrmse'])) for c in population)
                         if self.cfg.acceptance is not None else s['target']['first_hit_epoch'] is not None
                         if self.cfg.stop_metric is not None else None),
                     event_index=len(s['history']), **details)
        if self.pareto_mode:
            entry['training_target'] = deepcopy(s['target'])
        s['history'].append(entry)
        if self.callback:
            self.callback(dict(entry, replayed_operations=e.replayed_operations))

    def initialize(self):
        cfg, e, s = self.cfg, self.evaluator, self.s
        self.enter('initialization')
        lists = s.setdefault('initial_seed_lists', [])
        for j in range(len(lists), len(self.grammar.plain)):
            g = self.grammar.plain[j]
            seeds = [(t, np.ones(t.n_constants)) for t in generator(cfg, len(self.units[1]))(g, self.rng)] if cfg.linear_seeds else []
            if cfg.structured_seeds:
                structured, stats = structured_seeds(g, self.X, self.Y[:, j],
                    max(1, cfg.population_size//len(self.units[1])), compound=True, deadline=None,
                    work_callback=self.check_stop)
                seeds.extend(structured)
                s['seed_stats'].append(stats)
            if not seeds:
                tree, _ = g.generate(self.rng)
                seeds = [(tree, np.ones(tree.n_constants))]
            lists.append(seeds)
            self.save()
        population = []
        requests = []
        for k in range(min(cfg.stage_evaluations['initialization'], max(map(len, lists)))):
            if e.expired():
                break
            b = Builder()
            for seeds in lists:
                tree, warm = seeds[k % len(seeds)]
                b.roots.append(b.tree(tree, warm))
            s['attempts']['initialization'] += 1
            if self.pareto_mode:
                requests.append(b.finish())
                continue
            c = e(*b.finish())
            if c is not None:
                population.append(c)
        if self.pareto_mode:
            population = [c for c in e.evaluate_many(requests) if c is not None]
            self.observe_target(population, 0)
        if not population:
            for _ in range(cfg.population_size):
                if e.expired():
                    break
                graph, warm, _ = self.grammar.generate(self.rng)
                s['attempts']['initialization'] += 1
                c = e(graph, warm)
                if c is not None:
                    population.append(c)
                    if self.pareto_mode:
                        self.observe_target([c], 0)
        if not population:
            raise RuntimeError('No valid initialization within its finite evaluation/attempt budget.')
        b = Builder()
        for j in range(len(self.units[1])):
            c = min(population, key=lambda c: c.errors[j])
            b.roots.append(clone_subgraph(b, c.system, c.constants, c.system.roots[j]))
        system, warm = b.finish()
        s['reference_system'], s['reference_constants'] = system, warm
        s['reference'] = np.sqrt(np.mean(((system.evaluate(self.X, warm)-self.Y)/e.scales)**2, axis=0))
        if cfg.fixed_training_reference is not None:
            fixed = np.asarray(cfg.fixed_training_reference, dtype=float)
            if fixed.shape != s['reference'].shape:
                raise ValueError('Fixed training reference must match the output count.')
            s['reference'] = fixed.copy()
        if not e.expired():
            s['attempts']['initialization'] += 1
            c = e(system, warm)
            if c is not None:
                population.append(c)
                if self.pareto_mode:
                    self.observe_target([c], 0)
        s['population'] = diverse(population, cfg.population_size, cfg, s['reference'])
        self.finish('initialization', 'evaluation_budget' if e.expired() else 'seed_schedule_exhausted', 'exploration')

    def evolve(self, population, phase, multiobjective):
        if self.pareto_mode:
            return self.evolve_pareto(population, phase, multiobjective)
        cfg, e, s = self.cfg, self.evaluator, self.s
        ranked = ordered(population, cfg, s['reference']) if multiobjective else sorted(population,
            key=lambda c: (violations(c.errors, s['reference'], cfg), search_score(c.errors, cfg)))
        children = []
        for _ in range(cfg.population_size):
            if e.expired():
                break
            for _ in range(cfg.variation_attempts):
                parent = ranked[int(self.rng.integers(len(ranked), size=3).min())]
                donor = ranked[int(self.rng.integers(len(ranked), size=3).min())]
                graph, warm, op = offspring(parent, donor, self.grammar, self.rng)
                s['attempts'][phase] += 1
                s['operations'][op] += 1
                s['changed'][op] += int(graph != parent.system)
                if graph in e.cache or not in_bounds(graph, cfg):
                    continue
                s['novel'][op] += 1
                child = e(graph, warm)
                if child is not None:
                    children.append(child)
                    s['successful'][op] += 1
                break
        merged = list({c.system: c for c in population+children}.values())
        ranked = ordered(merged, cfg, s['reference'])
        survivors = (retain_parameters(ranked, cfg.population_size, cfg, s['reference']) if multiobjective else
                     diverse(ranked, cfg.population_size, cfg, s['reference']))
        return survivors, children

    def explore(self):
        if self.pareto_mode:
            return self.explore_pareto()
        cfg, e, s = self.cfg, self.evaluator, self.s
        self.enter('exploration')
        while s['step'] < cfg.epochs and not e.expired():
            if s['step']:
                cfg.max_nodes += 16
                cfg.max_depth += 1
                self.grammar = self.make_grammar()
            traces, rewards, sampled = [], [], []
            for _ in range(cfg.population_size):
                if e.expired():
                    break
                graph, warm, trace = self.grammar.generate(self.rng, self.policy)
                s['attempts']['exploration'] += 1
                c = e(graph, warm)
                traces.append(trace)
                rewards.append(0. if c is None else 1/(1+search_score(c.errors, cfg)+violations(c.errors, s['reference'], cfg)))
                if c is not None:
                    sampled.append(c)
            s['population'] = diverse(s['population']+sampled, cfg.population_size, cfg, s['reference'])
            if cfg.use_gp:
                for _ in range(cfg.gp_generations):
                    s['population'], _ = self.evolve(s['population'], 'exploration', False)
            if self.policy is not None and traces:
                elites, skipped = [], 0
                for c in diverse([c for c in e.cache.values() if c is not None],
                                 max(1, int(cfg.population_size*cfg.risk_fraction)), cfg, s['reference']):
                    try:
                        elites.append(self.grammar.trace(c.system))
                    except (ValueError, KeyError):
                        skipped += 1
                update = self.policy.update(traces, rewards, elites, cfg.risk_fraction, cfg.entropy_weight, cfg.imitation_weight)
                self.record('policy_update', imitation_systems=len(elites), imitation_unrepresentable=skipped, **update)
            self.record('exploration_epoch')
            s['step'] += 1
            self.save()
        self.prepare_pool()
        self.finish('exploration', 'evaluation_budget' if e.expired() else 'epoch_limit', 'nsga2')

    def prepare_pool(self):
        cfg, e, s = self.cfg, self.evaluator, self.s
        s['archive'] = [c for c in e.cache.values() if c is not None]
        count = max(cfg.min_candidates, int(np.ceil(cfg.candidate_fraction*len(s['archive']))))
        s['seeds'] = list({c.system: c for c in diverse(s['archive'], max(1, count//2), cfg, s['reference'])+
                         ordered(s['archive'], cfg, s['reference'])}.values())[:count]
        s['population'] = retain_parameters(ordered(s['seeds'], cfg, s['reference']), cfg.population_size, cfg, s['reference'])
        s['pool'] = list(s['seeds'])

    def policy_elites(self, archive):
        from .capacity import imitation
        cfg = self.cfg
        count = max(1, int(cfg.population_size * cfg.risk_fraction))
        preferred = diverse(archive, count, cfg, self.s['reference'])
        if cfg.imitation_scan_limit:
            preferred += diverse(archive, cfg.imitation_scan_limit, cfg, self.s['reference'])
            preferred = list({id(c): c for c in preferred}.values())[:cfg.imitation_scan_limit]
            preferred = [c for c in preferred if violations(c.errors, self.s['reference'], cfg) == 0]
        return imitation(self.grammar, preferred, count)

    def evolve_pareto(self, population, phase, multiobjective):
        cfg, e, s = self.cfg, self.evaluator, self.s
        ranked = ordered(population, cfg, s['reference']) if multiobjective else sorted(population,
            key=lambda c: (violations(c.errors, s['reference'], cfg), search_score(c.errors, cfg)))
        requests, operators, proposed = [], [], set()
        if cfg.compact_elites:
            from .capacity import compact
            for parent in ranked[:max(1, int(cfg.population_size * cfg.elite_fraction))]:
                graph, warm = compact(parent.system, parent.constants)
                if graph != parent.system and graph not in e.cache and graph not in proposed and in_bounds(graph, cfg):
                    requests.append((graph, warm))
                    operators.append('compact')
                    proposed.add(graph)
                    s['attempts'][phase] += 1
                    s['operations']['compact'] += 1
                    s['changed']['compact'] += 1
                    s['novel']['compact'] += 1
        for _ in range(cfg.population_size):
            if e.expired():
                break
            for _ in range(cfg.variation_attempts):
                parent = ranked[int(self.rng.integers(len(ranked), size=3).min())]
                donor = ranked[int(self.rng.integers(len(ranked), size=3).min())]
                s['attempts'][phase] += 1
                try:
                    graph, warm, op = offspring(parent, donor, self.grammar, self.rng)
                except (MemoryError, RecursionError):
                    e.stats['generation_resource_rejections'] += 1
                    continue
                s['operations'][op] += 1
                s['changed'][op] += int(graph != parent.system)
                if graph in e.cache or graph in proposed or not in_bounds(graph, cfg):
                    continue
                proposed.add(graph)
                s['novel'][op] += 1
                requests.append((graph, warm))
                operators.append(op)
                break
        values = e.evaluate_many(requests)
        self.generation_complete = e.last_batch_complete
        children = []
        for child, op in zip(values, operators):
            if child is not None:
                children.append(child)
                s['successful'][op] += 1
        merged = list({c.system: c for c in population+children}.values())
        ranked = ordered(merged, cfg, s['reference'])
        survivors = ranked[:cfg.population_size] if multiobjective else diverse(ranked, cfg.population_size, cfg, s['reference'])
        return survivors, children

    def explore_pareto(self):
        cfg, e, s = self.cfg, self.evaluator, self.s
        self.enter('exploration')
        while s['step'] < cfg.epochs and not e.expired() and not completed(s['target']):
            self.check_stop()
            if s['step'] and cfg.structure_limits is None:
                cfg.max_nodes += 16
                cfg.max_depth += 1
                self.grammar = self.make_grammar()
            requests, traces = [], []
            for _ in range(cfg.population_size):
                s['attempts']['exploration'] += 1
                try:
                    graph, warm, trace = self.grammar.generate(self.rng, self.policy)
                except (MemoryError, RecursionError):
                    e.stats['generation_resource_rejections'] += 1
                    continue
                requests.append((graph, warm))
                traces.append(trace)
            values = e.evaluate_many(requests)
            full_epoch = e.last_batch_complete
            rewards = [0. if c is None else 1/(1+search_score(c.errors, cfg)+
                cfg.reward_violation_weight*violations(c.errors, s['reference'], cfg)) for c in values]
            sampled = [c for c in values if c is not None]
            self.observe_target(sampled, s['step']+1)
            s['population'] = diverse(s['population']+sampled, cfg.population_size, cfg, s['reference'])
            for _ in range(cfg.gp_generations if cfg.use_gp else 0):
                if e.expired():
                    full_epoch = False
                    break
                s['population'], children = self.evolve_pareto(s['population'], 'exploration', False)
                full_epoch = full_epoch and self.generation_complete
                self.observe_target(children, s['step']+1)
            if self.policy is not None and traces:
                elites, reasons = self.policy_elites([c for c in e.cache.values() if c is not None])
                update = self.policy.update(traces, rewards, elites, cfg.risk_fraction, cfg.entropy_weight, cfg.imitation_weight)
                self.record('policy_update', imitation_systems=len(elites), imitation_unrepresentable=sum(reasons.values()), imitation_reasons=reasons, **update)
            s['step'] += 1
            if full_epoch:
                s['target']['completed_epochs'] += 1
            else:
                s['target']['partial_epoch'] = s['step']
            self.record('exploration_epoch', epoch_completed=full_epoch)
            self.save()
        reason = ('global_evaluation_budget' if e.calls >= cfg.max_evaluations else
                  'stage_evaluation_budget' if e.expired() else
                  'target_additional_epochs_complete' if completed(s['target']) else 'epoch_limit')
        s['target']['exit_reason'] = reason
        hit = s['target']['first_hit_epoch']
        s['target']['additional_epochs_completed'] = 0 if hit is None else max(0, s['target']['completed_epochs']-hit)
        self.prepare_pool()
        self.finish('exploration', reason, 'nsga2')

    def nsga(self):
        cfg, e, s = self.cfg, self.evaluator, self.s
        self.enter('nsga2')
        while cfg.use_gp and s['step'] < cfg.selection_generations and not e.expired():
            if self.pareto_mode and cfg.structure_limits is None:
                cfg.max_nodes += 16
                cfg.max_depth += 1
                self.grammar = self.make_grammar()
            s['population'], children = self.evolve(s['population'], 'nsga2', True)
            s['pool'].extend(children)
            self.record('nsga2_generation')
            s['step'] += 1
            self.save()
        self.finish('nsga2', 'disabled' if not cfg.use_gp else 'evaluation_budget' if e.expired() else 'generation_limit', 'polish')

    def polish(self):
        cfg, e, s = self.cfg, self.evaluator, self.s
        e.ceiling = sum(cfg.stage_evaluations.values())
        original = cfg.constant_max_nfev
        try:
            cfg.constant_max_nfev = cfg.polish_max_nfev
            for c in diverse(s['pool'], cfg.polish_candidates, cfg, s['reference']) if cfg.polish_candidates else []:
                e(c.system, c.constants, refine=True)
        finally:
            cfg.constant_max_nfev = original
        s['pool'] = list({c.system: e.cache[c.system] for c in s['pool']}.values())
        s['base_fingerprint'] = digest([c.to_dict() for c in e.cache.values() if c is not None])
        s['base_evaluations'] = e.calls
        self.record('polish', stop_reason='candidate_limit')
        s['phase'] = 'residual'
        self.save()

    def residual(self):
        self.enter('residual')
        s, e = self.s, self.evaluator
        if self.pareto_mode and self.cfg.residual_rounds == 0:
            s['extensions'] = []
            s['residual_report'] = dict(stop_reason='round_limit', completed_rounds=0, new_evaluations=0)
            self.finish('residual', 'round_limit', 'sharing')
            return
        before = e.stats['proposals']
        s['extensions'], s['residual_report'] = residual_search(s['pool'], e, self.grammar,
            max_rounds=self.cfg.residual_rounds, work_callback=self.check_stop)
        s['attempts']['residual'] += e.stats['proposals']-before
        reason = 'stage_evaluation_budget' if e.expired() else s['residual_report']['stop_reason']
        self.finish('residual', reason, 'sharing')

    def sharing(self):
        self.enter('sharing')
        s, e = self.s, self.evaluator
        s['pool'], s['closure'] = protected_closure(s['pool'], e, self.grammar,
            extensions=s['extensions'], max_rounds=self.cfg.sharing_rounds, max_attempts=self.cfg.sharing_attempts)
        s['closure'].update(accuracy_search=s['residual_report'], base_evaluations=s['base_evaluations'],
                            base_search_fingerprint=s['base_fingerprint'])
        s['attempts']['sharing'] += s['closure']['proposals']
        self.finish('sharing', s['closure']['stop_reason'], 'done')

    def run(self):
        if self.pareto_mode:
            torch.set_num_threads(1)
        try:
            while self.s['phase'] != 'done':
                {'initialization': self.initialize, 'exploration': self.explore, 'nsga2': self.nsga,
                 'polish': self.polish, 'residual': self.residual, 'sharing': self.sharing}[self.s['phase']]()
            s, e = self.s, self.evaluator
            assert e.calls <= self.cfg.max_evaluations
            assert e.stats['refinements'] <= self.cfg.polish_candidates
            self.owner.budget_report_ = dict(mode='evaluations', time_is_stop_condition=False,
                global_evaluation_limit=self.cfg.max_evaluations, actual_evaluations=e.calls,
                stages=s['reports'], generation_attempts=dict(s['attempts']),
                refinements=e.stats['refinements'], journal_operations=e.cursor,
                replayed_operations=e.replayed_operations, completed=True)
            if self.pareto_mode:
                self.owner.budget_report_.update(training_target=deepcopy(s['target']), execution=e.execution_report())
            return dict(evaluator=e, policy=self.policy, start=self.started, cpu_start=self.cpu_started,
                        stage_limit=sum(self.cfg.stage_evaluations[k] for k in STAGES[:2]),
                        **{k: s[k] for k in ('reference', 'reference_system', 'reference_constants', 'pool', 'closure',
                            'archive', 'seeds', 'seed_stats', 'operations', 'changed', 'novel', 'successful')})
        finally:
            try:
                self.evaluator.close()
            finally:
                self.store.close()
                if self.pareto_mode:
                    torch.set_num_threads(self.original_threads)
