"""Optional stagnation-controlled search over a single persistent global budget."""
from copy import deepcopy
import math
import time

import numpy as np
import torch

from .archive_evaluation import ArchiveEvaluator
from .evaluation_search import Search
from .evaluation import search_score
from .graph import Builder
from .genetics import clone_subgraph
from .pareto import diverse, violations
from .residual_session import ResidualSession
from .seeds import appended_seeds, balanced_prefix
from .selection import digest
from .sharing import protected_closure
from .stagnation import Patience, settings
from .structure import structured_seeds


class StagnationSearch(Search):
    evaluator_type = ArchiveEvaluator

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.controls = settings(self.cfg.stagnation_settings)
        self.s.setdefault('cycle', 0)
        self.s.setdefault('restarts', [])
        self.s.setdefault('patience', None)
        self.s.setdefault('initial_capacity', dict(max_nodes=self.cfg.max_nodes, max_depth=self.cfg.max_depth))
        self.evaluator.on_progress = self.progress
        if self.s['reference'] is not None:
            self.evaluator.monitor.initialize(self.s['reference'])

    def progress(self, metrics):
        if self.s['patience'] is not None and (not self.worst_progress() or 'B' in metrics):
            self.s['patience'].stale = 0
        self.record('effective_training_progress', improved_metrics=metrics, stage=self.s['phase'])

    def worst_progress(self):
        return self.cfg.search_objective == 'worst' and self.s['phase'] in ('exploration', 'nsga2', 'residual')

    def progress_sequence(self):
        monitor = self.evaluator.monitor
        return monitor.metric_sequences['B'] if self.worst_progress() else monitor.sequence

    def enter(self, phase):
        self.display_event(phase, action='开始或继续')
        e, s = self.evaluator, self.s
        e.ceiling = self.cfg.max_evaluations
        e.stage_tag, e.cycle = phase, s['cycle']
        key = str(s['cycle']) + ':' + phase
        if key not in s['reports']:
            s['reports'][key] = dict(cycle=s['cycle'], phase=phase, start_evaluations=e.calls,
                start_refinements=e.stats['refinements'], start_attempts=s['attempts'][phase],
                elapsed_start=self.elapsed(), monitor_at_entry=deepcopy(e.monitor.report()))
            s['reports'][key]['coordinator_cpu_start'] = time.process_time() - self.cpu_started
            s['reports'][key]['work_at_entry'] = {name:e.stats[name] for name in (
                'constant_residual_calls', 'constant_raw_training_predictions', 'constant_nfev',
                'optimizer_starts', 'optimizer_returns', 'optimizer_incomplete_starts',
                'separable_linear_attempts', 'separable_linear_solves', 'separable_linear_cpu_seconds',
                'fit_wall_seconds', 'fit_cpu_seconds', 'residual_screen_predictions')}
            profile = dict(exploration=('exploration_interval', 'exploration_patience'),
                           nsga2=('nsga_interval', 'nsga_patience'),
                           residual=('residual_interval', 'residual_patience'))
            if phase in profile:
                interval, patience = profile[phase]
                s['patience'] = Patience(self.controls[interval], self.controls[patience], self.progress_sequence(),
                    self.controls['exploration_warmup'] if phase == 'exploration' else 0)
            elif phase == 'sharing':
                s['patience'] = Patience(1, self.controls['sharing_patience'], e.monitor.sequence)
            else:
                s['patience'] = None
        if self.cfg.stage_budgets is not None and phase not in ('initialization', 'polish'):
            report = s['reports'][key]
            allowance = self.cfg.stage_budgets[s['cycle']][phase]
            e.ceiling = min(self.cfg.max_evaluations, report['start_evaluations'] + allowance)
            report.update(first_evaluation_allowance=allowance, ceiling=e.ceiling)

    def finish(self, phase, reason, following):
        e, s = self.evaluator, self.s
        if self.cfg.stage_budgets is not None and phase not in ('initialization', 'polish') and e.calls >= e.ceiling and e.calls < self.cfg.max_evaluations:
            reason = 'stage_evaluation_budget'
        patience = s['patience']
        if patience is not None and ((patience.units and
                (not patience.history or patience.history[-1]['units'] != patience.units)) or
                (s['step'] and not patience.history)):
            self.partial_check()
        report = s['reports'][str(s['cycle']) + ':' + phase]
        report.update(stop_reason=reason, evaluations=e.calls - report['start_evaluations'],
            end_evaluations=e.calls, refinements=e.stats['refinements'] - report['start_refinements'],
            attempts=s['attempts'][phase] - report['start_attempts'],
            elapsed_seconds=self.elapsed() - report['elapsed_start'], monitor_at_exit=deepcopy(e.monitor.report()),
            stagnation_checks=[] if s['patience'] is None else deepcopy(s['patience'].history))
        report['coordinator_cpu_seconds'] = time.process_time() - self.cpu_started - report['coordinator_cpu_start']
        report['numerical_work'] = {name:e.stats[name]-value for name,value in report['work_at_entry'].items()}
        self.record(phase, stop_reason=reason)
        s['phase'], s['step'], s['patience'] = following, 0, None
        if phase == 'restart' and following == 'exploration':
            s['cycle'] += 1
        e.ceiling = self.cfg.max_evaluations
        self.save()

    def record(self, phase, **details):
        e, s = self.evaluator, self.s
        monitor = e.monitor.report()
        entry = dict(phase=phase, step=s['step'], cycle=s.get('cycle', 0), evaluations=e.calls,
            elapsed_seconds=self.elapsed(), population=len(s['population']),
            best_train_mean_nrmse=None if monitor['A'] is None else monitor['A']['mean_nrmse'],
            best_train_maximum_nrmse=None if monitor['B'] is None else monitor['B']['maximum_nrmse'],
            constant_nfev=e.stats['constant_nfev'], refinements=e.stats['refinements'],
            training_monitor=monitor, event_index=len(s['history']), **details)
        entry['training_protection_reference'] = None if s['reference'] is None else s['reference'].tolist()
        entry['normalization_scales'] = e.scales.tolist()
        s['history'].append(entry)
        if self.callback:
            self.callback(dict(entry, replayed_operations=e.replayed_operations))

    def initialize(self):
        cfg, e, s = self.cfg, self.evaluator, self.s
        self.enter('initialization')
        limit = min(self.controls['initialization_evaluations'], cfg.max_evaluations)
        e.ceiling = limit
        lists = s.setdefault('initial_seed_lists', [])
        for j in range(len(lists), len(self.grammar.plain)):
            self.check_stop()
            g = self.grammar.plain[j]
            seeds = [(tree, np.ones(tree.n_constants)) for tree in balanced_prefix(
                list(appended_seeds(g, self.rng)), self.controls['initialization_seed_limit'])] if cfg.linear_seeds else []
            if cfg.compact_seed_terms:
                from .residual_guided import compact_seeds
                compact, stats = compact_seeds(g, self.X, self.Y[:, j], e.probes,
                                               cfg.compact_seed_terms, self.check_stop)
                seeds = compact + seeds
                s['seed_stats'].append(stats)
            if cfg.structured_seeds:
                structured, stats = structured_seeds(g, self.X, self.Y[:, j],
                    max(1, cfg.population_size // len(self.units[1])), compound=True, deadline=None,
                    work_callback=self.check_stop)
                seeds.extend(structured)
                s['seed_stats'].append(stats)
            if not seeds:
                tree, _ = g.generate(self.rng)
                seeds = [(tree, np.ones(tree.n_constants))]
            lists.append(seeds)
            self.save()
        requests = []
        for k in range(min(limit, max(map(len, lists)))):
            builder = Builder()
            for seeds in lists:
                tree, warm = seeds[k % len(seeds)]
                builder.roots.append(builder.tree(tree, warm))
            s['attempts']['initialization'] += 1
            requests.append(builder.finish())
        population = [candidate for candidate in e.evaluate_many(requests) if candidate is not None]
        if not population:
            for _ in range(cfg.population_size):
                if e.expired():
                    break
                s['attempts']['initialization'] += 1
                graph, warm, _ = self.grammar.generate(self.rng)
                candidate = e(graph, warm)
                if candidate is not None:
                    population.append(candidate)
        if not population:
            raise RuntimeError('No legal initialization within the finite initialization/global evaluation budget.')
        builder = Builder()
        for j in range(len(self.units[1])):
            candidate = min(population, key=lambda c: c.errors[j])
            builder.roots.append(clone_subgraph(builder, candidate.system, candidate.constants, candidate.system.roots[j]))
        system, warm = builder.finish()
        s['reference_system'], s['reference_constants'] = system, warm
        s['reference'] = np.sqrt(np.mean(((system.evaluate(self.X, warm) - self.Y) / e.scales) ** 2, axis=0))
        if cfg.fixed_training_reference is not None:
            fixed = np.asarray(cfg.fixed_training_reference, dtype=float)
            if fixed.shape != s['reference'].shape:
                raise ValueError('Fixed training reference must match the output count.')
            s['reference'] = fixed.copy()
        if not e.expired():
            s['attempts']['initialization'] += 1
            candidate = e(system, warm)
            if candidate is not None:
                population.append(candidate)
        e.monitor.initialize(s['reference'])
        s['population'] = diverse(population, cfg.population_size, cfg, s['reference'])
        self.finish('initialization', 'global_evaluation_budget' if e.calls >= cfg.max_evaluations else
                    'finite_seed_schedule_complete', 'exploration')

    def check_patience(self, count=1):
        check = self.s['patience'].advance(self.progress_sequence(), count)
        if check is not None:
            self.record('stagnation_check', checked_stage=self.s['phase'], **check)

    def partial_check(self):
        patience = self.s['patience']
        if patience is not None:
            check = patience.check(self.progress_sequence(), partial=True)
            self.record('stagnation_check', checked_stage=self.s['phase'], **check)

    def resource_count(self):
        e = self.evaluator
        return e.stats['resource_screening_rejections'] + e.stats['generation_resource_rejections']

    def window_open(self, phase):
        limit = (self.cfg.stage_windows or {}).get(phase)
        return limit is None or self.s['step'] < limit

    def explore(self):
        cfg, e, s = self.cfg, self.evaluator, self.s
        self.enter('exploration')
        reason = 'stagnation'
        while not s['patience'].stopped and e.calls < e.ceiling and self.window_open('exploration'):
            self.check_stop()
            before, resources = e.calls, self.resource_count()
            if s['step'] and cfg.structure_limits is None:
                previous_capacity = cfg.max_nodes, cfg.max_depth
                cfg.max_nodes += 16
                cfg.max_depth += 1
                try:
                    self.grammar = self.make_grammar()
                except (MemoryError, RecursionError):
                    cfg.max_nodes, cfg.max_depth = previous_capacity
                    e.stats['generation_resource_rejections'] += 1
                    reason = 'resource_protection'
                    break
            requests, traces = [], []
            for _ in range(cfg.population_size):
                s['attempts']['exploration'] += 1
                try:
                    graph, warm, trace = self.grammar.generate(self.rng, self.policy)
                except (MemoryError, RecursionError):
                    e.stats['generation_resource_rejections'] += 1
                    continue
                except (ValueError, KeyError):
                    e.stats['generation_legality_rejections'] += 1
                    continue
                requests.append((graph, warm))
                traces.append(trace)
            values = e.evaluate_many(requests)
            full = e.last_batch_complete
            rewards = [0. if c is None else 1 / (1 + search_score(c.errors, cfg) +
                cfg.reward_violation_weight * violations(c.errors, s['reference'], cfg)) for c in values]
            sampled = [c for c in values if c is not None]
            s['population'] = diverse(s['population'] + sampled, cfg.population_size, cfg, s['reference'])
            self.display_event('neural' if self.policy else 'random', round_number=s['step']+1)
            for generation in range(1, (cfg.gp_generations if cfg.use_gp else 0)+1):
                if e.calls >= e.ceiling:
                    full = False
                    break
                s['population'], children = self.evolve_pareto(s['population'], 'exploration', False)
                self.display_event('gp', round_number=s['step']+1, generation=generation,
                                   valid_offspring=len(children))
                full = full and self.generation_complete
            if self.policy is not None and traces:
                elites, reasons = self.policy_elites(list(e.archive.values()))
                update = self.policy.update(traces, rewards, elites, cfg.risk_fraction,
                    cfg.entropy_weight, cfg.imitation_weight)
                self.record('policy_update', imitation_systems=len(elites), imitation_unrepresentable=sum(reasons.values()), imitation_reasons=reasons, **update)
            s['step'] += 1
            if full:
                self.check_patience()
            else:
                self.partial_check()
            self.record('exploration_epoch', epoch_completed=full)
            self.save()
            if e.calls == before and self.resource_count() > resources:
                reason = 'resource_protection'
                break
        if not self.window_open('exploration'):
            reason = 'capacity_window'
        if e.calls >= cfg.max_evaluations:
            reason = 'global_evaluation_budget'
        self.prepare_pool()
        self.finish('exploration', reason, 'nsga2')

    def nsga(self):
        cfg, e, s = self.cfg, self.evaluator, self.s
        self.enter('nsga2')
        reason = 'stagnation' if cfg.use_gp else 'disabled'
        while cfg.use_gp and not s['patience'].stopped and e.calls < e.ceiling and self.window_open('nsga2'):
            before, resources = e.calls, self.resource_count()
            if cfg.structure_limits is None:
                previous_capacity = cfg.max_nodes, cfg.max_depth
                cfg.max_nodes += 16
                cfg.max_depth += 1
                try:
                    self.grammar = self.make_grammar()
                except (MemoryError, RecursionError):
                    cfg.max_nodes, cfg.max_depth = previous_capacity
                    e.stats['generation_resource_rejections'] += 1
                    reason = 'resource_protection'
                    break
            s['population'], children = self.evolve_pareto(s['population'], 'nsga2', True)
            s['pool'].extend(children)
            s['step'] += 1
            if self.generation_complete:
                self.check_patience()
            else:
                self.partial_check()
            self.record('nsga2_generation', generation_completed=self.generation_complete)
            self.save()
            if e.calls == before and self.resource_count() > resources:
                reason = 'resource_protection'
                break
        if not self.window_open('nsga2'):
            reason = 'capacity_window'
        if e.calls >= cfg.max_evaluations:
            reason = 'global_evaluation_budget'
        self.finish('nsga2', reason, 'polish')

    def polish(self):
        cfg, e, s = self.cfg, self.evaluator, self.s
        self.enter('polish')
        if 'polish_plan' not in s:
            remaining = max(0, cfg.polish_candidates - e.stats['refinements'])
            s['polish_plan'] = diverse(s['pool'], remaining, cfg, s['reference']) if remaining else []
            s['polish_position'] = 0
            self.save()
        while s['polish_position'] < len(s['polish_plan']) and e.calls < e.ceiling:
            candidate = s['polish_plan'][s['polish_position']]
            original = cfg.constant_max_nfev
            try:
                cfg.constant_max_nfev = cfg.polish_max_nfev
                e(candidate.system, candidate.constants, refine=True)
            finally:
                cfg.constant_max_nfev = original
            s['polish_position'] += 1
            self.record('polish_candidate', position=s['polish_position'])
            self.save()
        s['pool'] = list({c.system: e.cache[c.system] for c in s['pool']}.values())
        s['base_fingerprint'] = digest([c.to_dict() for c in e.archive.values()])
        s['base_evaluations'] = e.calls
        self.finish('polish', 'global_evaluation_budget' if e.calls >= cfg.max_evaluations else
                    'finite_refinement_schedule_complete', 'residual')

    def residual(self):
        cfg, e, s = self.cfg, self.evaluator, self.s
        self.enter('residual')
        if e.calls >= e.ceiling:
            s['extensions'], s['residual_report'] = [], dict(stop_reason='stage_evaluation_budget')
            self.finish('residual', 'stage_evaluation_budget', 'sharing')
            return
        if 'residual_session' not in s:
            if cfg.residual_guidance:
                from .residual_guided import GuidedResidualSession
                session_type = GuidedResidualSession
            else:
                session_type = ResidualSession
            s['residual_session'] = session_type(s['pool'], e, self.grammar, self.check_stop)
            s['residual_pending_evaluations'] = 0
            s['residual_empty_attempts'] = 0
            self.save()
        session = s['residual_session']
        session.attach(e, self.grammar, self.save)
        reason = 'stagnation'
        while not s['patience'].stopped and e.calls < e.ceiling and self.window_open('residual'):
            self.check_stop()
            before, resources = e.calls, self.resource_count()
            previous_capacity = cfg.max_nodes, cfg.max_depth
            try:
                proposed = session.step()
            except (MemoryError, RecursionError):
                cfg.max_nodes, cfg.max_depth = previous_capacity
                e.stats['generation_resource_rejections'] += 1
                s['attempts']['residual'] += 1
                reason = 'resource_protection'
                break
            if not proposed:
                reason = session.report['stop_reason']
                break
            s['attempts']['residual'] += 1
            s['step'] = session.rounds
            added = e.calls - before
            s['residual_pending_evaluations'] += added
            s['residual_empty_attempts'] = s['residual_empty_attempts'] + 1 if not added else 0
            if s['residual_pending_evaluations'] >= self.controls['residual_interval']:
                self.check_patience(self.controls['residual_interval'])
                s['residual_pending_evaluations'] = 0
            self.record('residual_round', new_evaluations=added)
            self.save()
            if s['residual_empty_attempts'] >= cfg.variation_attempts:
                reason = 'resource_protection' if self.resource_count() > resources else 'proposal_attempt_limit'
                break
        if not self.window_open('residual'):
            reason = 'capacity_window'
        if e.calls >= cfg.max_evaluations:
            reason = 'global_evaluation_budget'
        if s['residual_pending_evaluations']:
            s['patience'].units += s['residual_pending_evaluations']
            self.partial_check()
            s['residual_pending_evaluations'] = 0
        session.report['stop_reason'] = reason
        s['extensions'], s['residual_report'] = session.retained, session.report
        self.finish('residual', reason, 'sharing')

    def sharing(self):
        cfg, e, s = self.cfg, self.evaluator, self.s
        self.enter('sharing')
        if 'sharing_pool' not in s:
            s['sharing_pool'] = s['pool'] + s['extensions']
            s['sharing_round_reports'] = []
            self.save()
        reason = 'stagnation'
        while not s['patience'].stopped and e.calls < e.ceiling and self.window_open('sharing'):
            self.check_stop()
            before, resources = e.calls, self.resource_count()
            try:
                pool, report = protected_closure(s['sharing_pool'], e, self.grammar, extensions=[],
                    max_rounds=1, max_attempts=cfg.sharing_attempts)
            except (MemoryError, RecursionError) as error:
                e.stats['generation_resource_rejections'] += 1
                s['sharing_round_reports'].append(dict(stop_reason='resource_protection',
                    error=type(error).__name__, proposals_count_incomplete=True, new_evaluations=e.calls - before))
                reason = 'resource_protection'
                break
            s['sharing_pool'] = pool
            s['sharing_round_reports'].append(report)
            s['attempts']['sharing'] += report['proposals']
            s['step'] += 1
            full = e.calls < e.ceiling and e.last_batch_complete
            if full:
                self.check_patience()
            else:
                self.partial_check()
            self.record('sharing_round', new_evaluations=e.calls - before, round_completed=full)
            self.save()
            if e.calls == before:
                reason = 'resource_protection' if self.resource_count() > resources else 'proposal_space_exhausted'
                break
        if not self.window_open('sharing'):
            reason = 'capacity_window'
        if e.calls >= cfg.max_evaluations:
            reason = 'global_evaluation_budget'
        s['pool'] = s['sharing_pool']
        s['closure'] = dict(base_candidate_count=len(s['pool']), accuracy_search=s['residual_report'],
            rounds=s['step'], round_reports=s['sharing_round_reports'], stop_reason=reason,
            base_evaluations=s['base_evaluations'], base_search_fingerprint=s['base_fingerprint'])
        self.finish('sharing', reason, 'restart')

    def restart(self):
        cfg, e, s = self.cfg, self.evaluator, self.s
        if cfg.stage_budgets is not None and s['cycle'] + 1 >= len(cfg.stage_budgets):
            s['exit_reason'], s['phase'] = 'stage_plan_complete', 'done'
            self.save()
            return
        if 'restart_state' not in s and len(s['restarts']) >= self.controls['max_restarts']:
            s['exit_reason'], s['phase'] = 'restart_limit_after_stage_termination', 'done'
            self.save()
            return
        self.enter('restart')
        if 'restart_state' not in s:
            count = max(1, math.ceil(cfg.population_size * self.controls['restart_keep_fraction']))
            monitor = e.monitor
            champions = [key for key in (monitor.best_a, monitor.best_b, monitor.best_c) if key is not None]
            identities = {id(candidate): key for key, candidate in e.archive.items()}
            ranked = champions + [identities[id(candidate)] for candidate in
                diverse(list(e.archive.values()), count, cfg, s['reference'])]
            kept = [e.archive[key] for key in list(dict.fromkeys(ranked))[:count]]
            number = len(s['restarts']) + 1
            seed = int(np.random.SeedSequence([cfg.random_state, number, 0x53544147]).generate_state(1)[0])
            rng = np.random.default_rng(seed)
            s['restart_state'] = dict(number=number, seed=seed, rng=rng.bit_generator.state,
                kept=kept, fresh=[], attempts=0, maximum_attempts=max(1, cfg.population_size - len(kept)) * cfg.variation_attempts)
            s['restarts'].append(dict(number=number, seed=seed, retained=len(kept), start_evaluations=e.calls,
                monitor_before=deepcopy(e.monitor.report()), reference=s['reference'].tolist()))
            cfg.max_nodes, cfg.max_depth = s['initial_capacity']['max_nodes'], s['initial_capacity']['max_depth']
            try:
                self.grammar = self.make_grammar()
            except (MemoryError, RecursionError):
                e.stats['generation_resource_rejections'] += 1
                s['exit_reason'] = 'resource_protection'
                s['restarts'][-1].update(fresh=0, attempts=0, end_evaluations=e.calls,
                                        stop_reason='resource_protection')
                self.finish('restart', 'resource_protection', 'done')
                return
            self.save()
        state = s['restart_state']
        rng = np.random.default_rng()
        rng.bit_generator.state = state['rng']
        while (len(state['kept']) + len(state['fresh']) < cfg.population_size and
               state['attempts'] < state['maximum_attempts'] and e.calls < e.ceiling):
            self.check_stop()
            state['attempts'] += 1
            s['attempts']['restart'] += 1
            try:
                graph, warm, _ = self.grammar.generate(rng)
                candidate = e(graph, warm)
                if candidate is not None and graph not in {c.system for c in state['kept'] + state['fresh']}:
                    state['fresh'].append(candidate)
            except (MemoryError, RecursionError):
                e.stats['generation_resource_rejections'] += 1
            except (ValueError, KeyError):
                e.stats['generation_legality_rejections'] += 1
            state['rng'] = rng.bit_generator.state
            self.save()
        s['population'] = state['kept'] + state['fresh']
        s['restarts'][-1].update(fresh=len(state['fresh']), attempts=state['attempts'],
            end_evaluations=e.calls, monitor_after=deepcopy(e.monitor.report()))
        reason = ('global_evaluation_budget' if e.calls >= cfg.max_evaluations else
                  'diversity_replenished' if len(s['population']) >= cfg.population_size else 'proposal_attempt_limit')
        for key in ('restart_state', 'polish_plan', 'polish_position', 'residual_session',
                    'residual_pending_evaluations', 'residual_empty_attempts', 'sharing_pool', 'sharing_round_reports'):
            s.pop(key, None)
        self.finish('restart', reason, 'exploration')

    def run(self, on_complete=None):
        torch.set_num_threads(1)
        try:
            while self.s['phase'] != 'done':
                if self.evaluator.target_hit is not None:
                    self.s['exit_reason'], self.s['phase'] = 'training_target', 'done'
                    self.record('search_complete', stop_reason='training_target', target_hit=self.evaluator.target_hit)
                    self.save()
                    break
                if self.evaluator.calls >= self.cfg.max_evaluations and self.s['reference'] is not None:
                    self.s['exit_reason'], self.s['phase'] = 'global_evaluation_budget', 'done'
                    self.record('search_complete', stop_reason='global_evaluation_budget')
                    self.save()
                    break
                dict(initialization=self.initialize, exploration=self.explore, nsga2=self.nsga,
                     polish=self.polish, residual=self.residual, sharing=self.sharing,
                     restart=self.restart)[self.s['phase']]()
            s, e = self.s, self.evaluator
            assert e.calls <= self.cfg.max_evaluations
            assert e.stats['refinements'] <= self.cfg.polish_candidates
            archive = list(e.archive.values())
            s.setdefault('archive', archive)
            s.setdefault('seeds', archive)
            s.setdefault('pool', archive)
            s.setdefault('closure', dict(base_candidate_count=len(archive), stop_reason=s.get('exit_reason')))
            self.owner.training_trajectory_ = deepcopy(e.trajectory)
            self.owner.numerical_work_ = deepcopy(e.numerical_work)
            self.owner.budget_report_ = dict(mode='stagnation', time_is_stop_condition=False,
                global_evaluation_limit=self.cfg.max_evaluations, actual_evaluations=e.calls,
                refinements=e.stats['refinements'], stages=s['reports'], restarts=s['restarts'],
                generation_attempts=dict(s['attempts']), training_monitor=e.monitor.report(),
                exit_reason=s.get('exit_reason'), execution=e.execution_report(),
                completed=True, journal_operations=e.cursor, replayed_operations=e.replayed_operations,
                controls=self.controls, ignored_legacy_limits=dict(epochs=self.cfg.epochs,
                    selection_generations=self.cfg.selection_generations, residual_rounds=self.cfg.residual_rounds,
                    sharing_rounds=self.cfg.sharing_rounds),
                target_hit=e.target_hit, stage_budgets=self.cfg.stage_budgets,
                final_cycle_stop_reasons={row['phase']: row['stop_reason'] for row in s['reports'].values()
                    if row['cycle'] == s['cycle']})
            if on_complete is not None:
                on_complete(self)
            return dict(evaluator=e, policy=self.policy, start=self.started, cpu_start=self.cpu_started,
                stage_limit=self.cfg.max_evaluations, archive=archive,
                **{key: s[key] for key in ('reference', 'reference_system', 'reference_constants', 'pool', 'closure',
                    'seeds', 'seed_stats', 'operations', 'changed', 'novel', 'successful')})
        finally:
            try:
                self.evaluator.close()
            finally:
                self.store.close()
                torch.set_num_threads(self.original_threads)
