"""Persistent, serial portfolio over the native complete-system evaluator."""
from copy import deepcopy
import numpy as np
import torch

from .archive_evaluation import ArchiveEvaluator
from .evaluation import search_score
from .evaluation_search import Search
from .portfolio_fitting import fit_request, promotion_evidence
from .portfolio_proposals import (Proposal, Producers, RootArchive, capacity_rejection,
                                  identity)
from .portfolio_scheduler import choose, feedback, initial_state, settings
from .selection import digest
from .stopping import completed
from .structural_discovery import StructuralRootArchive, SupportSearch, recombine


class PortfolioSearch(Search):
    """Each proposal and its exact fit protocol is saved before native work."""
    evaluator_type = ArchiveEvaluator

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.controls = settings(self.cfg.portfolio_settings)
        self.producers = Producers(self.grammar, self.controls)
        bank_type = StructuralRootArchive if self.controls['structural_discovery'] else RootArchive
        self.s.setdefault('portfolio', dict(
            roots=bank_type(len(self.units[1]), self.controls['root_capacity']),
            scheduler=initial_state(self.controls), pending=None, proposal_count=0, seed_index=0,
            accepted=[], promotions=set(), stop_reason=None, policy_batch=[],
            failed_requests=[], promotion_prefer_failure=True))
        self.s['portfolio'].setdefault('completion_seen', set())
        self.s['portfolio'].setdefault('completion_queue', [])
        self.s['portfolio'].setdefault('failure_blocked', set())
        self.s.setdefault('pool', [])
        self.s.setdefault('reference_system', None)
        self.s.setdefault('reference_constants', None)
        self.s['phase'] = 'done' if self.s['portfolio']['stop_reason'] else 'portfolio'
        self.evaluator.ceiling = self.cfg.max_evaluations
        self.evaluator.stage_tag = 'portfolio'

    def total_attempts(self):
        return self.evaluator.calls + self.evaluator.stats['refinements']

    def record(self, phase, **details):
        event = dict(phase=phase, event_index=len(self.s['history']),
                     evaluations=self.evaluator.calls,
                     refinements=self.evaluator.stats['refinements'],
                     total_fit_attempts=self.total_attempts(), **details)
        self.s['history'].append(event)
        if self.callback is not None:
            self.callback(dict(event, replayed_operations=self.evaluator.replayed_operations))

    def promotion(self, force=False):
        state, evaluator = self.s['portfolio'], self.evaluator
        if (evaluator.stats['refinements'] >= self.cfg.polish_candidates or
                not self.total_attempts() or (not force and self.s['pool'] and
                self.total_attempts() % self.controls['refine_every'])):
            return None
        failures = []
        for failed in state['failed_requests']:
            key = digest(failed.system.to_dict())
            for level in ('standard', 'refine'):
                if (key, level) not in state['promotions']:
                    failures.append(Proposal(failed.system, failed.warm.copy(), 'refinement',
                        outputs=failed.outputs, operation='failed_request_promotion',
                        level=level, refine=True, parents=failed.parents))
                    break
        ranked = sorted(self.s['pool'], key=lambda c: (search_score(c.errors, self.cfg),
                                                      c.complexity, identity(c)))
        successes = []
        for candidate in ranked:
            key = digest(candidate.system.to_dict())
            if key in state['failure_blocked']:
                continue
            for level in ('standard', 'refine'):
                if (key, level) not in state['promotions']:
                    successes.append(Proposal(candidate.system, candidate.constants.copy(),
                        'refinement', operation='promotion', level=level, refine=True,
                        parents=(identity(candidate),)))
                    break
        preferred = (failures, successes) if state['promotion_prefer_failure'] else (successes, failures)
        for proposals in preferred:
            if proposals:
                state['promotion_prefer_failure'] = not state['promotion_prefer_failure']
                return proposals[0]
        return None

    def completion_proposal(self):
        from .coefficient_completion import complete_coefficients
        state = self.s['portfolio']
        if state['completion_queue']:
            return state['completion_queue'].pop(0)
        for candidate in self.s['pool']:
            key = identity(candidate)
            if key in state['completion_seen']:
                continue
            state['completion_seen'].add(key)
            proposals, skipped = complete_coefficients(candidate, self.units[2], self.cfg)
            for row in skipped:
                self.record('coefficient_completion_skipped', parent=key, **row)
            state['completion_queue'].extend(proposals)
            # Keep one parent's bounded frontier in memory at a time.
            if state['completion_queue']:
                break
        if state['completion_queue']:
            return state['completion_queue'].pop(0)
        return None

    def propose(self):
        state = self.s['portfolio']
        proposal = self.promotion()
        if proposal is not None:
            return proposal
        if self.controls['structural_discovery'] and state['proposal_count'] % 3 != 0:
            if 'support_queue' not in state:
                stats = self.evaluator.stats
                before = stats['portfolio_feature_predictions']
                screen = SupportSearch(self.grammar, self.X, self.Y, stats)
                queue = []
                seen = set(self.evaluator.cache)
                for index in range(24):
                    item = screen.propose(index, seen, stats)
                    if item is not None:
                        queue.append(item)
                        seen.add(item.system)
                state['support_queue'] = queue
                self.record('structural_supports_prepared', admissible=len(queue),
                            attempted_supports=24, basis_predictions=stats['portfolio_feature_predictions']-before,
                            direction_scores=stats['structural_direction_scores'])
            while state['support_queue']:
                item = state['support_queue'].pop(0)
                if item.system not in self.evaluator.cache:
                    self.s['attempts']['features'] += 1
                    return item
        if (self.controls['coefficient_completion'] and
                self.controls['completion_scheduler'] == 'priority'):
            proposal = self.completion_proposal()
            if proposal is not None:
                return proposal
        name = choose(state['scheduler'], self.controls, state['roots'].version)
        if name is None:
            return self.promotion(force=True)
        self.s['attempts'][name] += 1
        counters = ('portfolio_feature_predictions', 'raw_system_predictions')
        generation_before = sum(self.evaluator.stats[key] for key in counters)
        try:
            if name == 'coefficients':
                proposal = self.completion_proposal()
                if proposal is not None:
                    proposal.producer = name
            elif not self.s['pool'] and name == 'features':
                proposal = self.producers.initial(state['seed_index'], self.rng)
                state['seed_index'] += 1
            elif name == 'roots' and self.controls['structural_discovery']:
                proposal = recombine(state['roots'], self.cfg, self.evaluator.cache)
                if proposal is None:
                    proposal = self.producers.propose(name, dict(rng=self.rng, roots=state['roots'], pool=self.s['pool']))
            else:
                proposal = self.producers.propose(name, dict(
                    rng=self.rng, roots=state['roots'], pool=self.s['pool'],
                    policy=self.policy, index=state['scheduler'][name]['selections'],
                    X=self.X, Y=self.Y, stats=self.evaluator.stats))
        except (ValueError, MemoryError, RecursionError) as error:
            self.record('proposal_generation_rejected', producer=name,
                        reason=type(error).__name__ + ': ' + str(error))
            proposal = None
        if proposal is None:
            feedback(state['scheduler'], name, self.controls, state['roots'].version,
                     empty=True, cost=1. + sum(self.evaluator.stats[key] for key in counters) - generation_before)
            self.record('producer_empty', producer=name)
        else:
            proposal.generation_cost = sum(self.evaluator.stats[key] for key in counters) - generation_before
        return proposal

    def reserve(self, proposal):
        state = self.s['portfolio']
        for rejected in proposal.screening_rejections:
            self.evaluator.stats['portfolio_capacity_rejections'] += 1
            self.record('proposal_capacity_fallback', producer=proposal.producer,
                        outputs=list(proposal.outputs), **rejected)
        reasons, metrics = capacity_rejection(proposal.system, self.cfg)
        duplicate = proposal.system in self.evaluator.cache and not proposal.refine
        if reasons or duplicate:
            if proposal.producer != 'refinement':
                feedback(state['scheduler'], proposal.producer, self.controls,
                         state['roots'].version, empty=True, cost=1. + proposal.generation_cost)
            self.record('proposal_rejected', producer=proposal.producer,
                        operation=proposal.operation, reasons=reasons,
                        reason='capacity' if reasons else 'cached_duplicate', shape=metrics,
                        outputs=list(proposal.outputs))
            return False
        # The full Proposal includes graph, warm values, producer, trace, outputs,
        # fidelity and refinement identity. RNG and policy precede this request.
        state['pending'] = dict(proposal=proposal,
                               best_errors=state['roots'].best_errors(),
                               cursor=self.evaluator.cursor,
                               stats=dict(self.evaluator.stats),
                               archive_version=state['roots'].version)
        self.record('proposal_reserved', producer=proposal.producer,
                    operation=proposal.operation, outputs=list(proposal.outputs),
                    level=proposal.level, refine=proposal.refine,
                    graph_sha256=digest(proposal.system.to_dict()),
                    parents=list(proposal.parents))
        self.save()
        return True

    def complete_pending(self):
        state, evaluator = self.s['portfolio'], self.evaluator
        pending = state['pending']
        proposal = pending['proposal']
        candidate = fit_request(evaluator, proposal.system, proposal.warm,
                                level=proposal.level, outputs=proposal.outputs,
                                refine=proposal.refine)
        before = pending['best_errors']
        if candidate is not None:
            # Native ArchiveEvaluator owns complete accepted versions. This
            # insertion also supports a deterministic injected service in tests.
            evaluator.archive.setdefault(identity(candidate), candidate)
        accepted_ids = set(state['accepted'])
        accepted = []
        for value in evaluator.archive.values():
            key = identity(value)
            if key not in accepted_ids:
                accepted_ids.add(key)
                state['accepted'].append(key)
                state['roots'].add(value)
                accepted.append(value)
        if accepted:
            pool = {identity(c): c for c in self.s['pool']}
            pool.update((identity(value), value) for value in accepted)
            self.s['pool'] = sorted(pool.values(), key=lambda c: (
                search_score(c.errors, self.cfg), c.complexity, identity(c)))[:self.cfg.population_size]
            self.s['population'] = self.s['pool']
            if self.s['reference'] is None:
                reference_candidate = accepted[0]
                self.s['reference'] = (reference_candidate.errors.copy() if self.cfg.fixed_training_reference is None
                                       else np.asarray(self.cfg.fixed_training_reference, float))
                self.s['reference_system'] = reference_candidate.system
                self.s['reference_constants'] = reference_candidate.constants.copy()
                evaluator.monitor.initialize(self.s['reference'])
            self.observe_target(accepted, 0)
        graph_key = digest(proposal.system.to_dict())
        evidence = None
        if self.controls['failure_promotion'] == 'evidence':
            row = evaluator.checkpoint.db.execute(
                'SELECT payload FROM evaluation_work WHERE id=?', (pending['cursor'],)).fetchone()
            work = None if row is None else evaluator.checkpoint.decode(row[0])
            evidence = promotion_evidence(work, pending['cursor'])
            self.record('failure_promotion_classified', graph_sha256=graph_key,
                        level=proposal.level, **evidence)
            if evidence['eligible']:
                state['failure_blocked'].discard(graph_key)
            elif evidence['reason'] != 'native_accepted':
                state['failure_blocked'].add(graph_key)
            else:
                state['failure_blocked'].discard(graph_key)
        native_failure = evidence is not None and evidence['eligible']
        remove_failed = (proposal.level == 'refine' or
                         (candidate is not None if evidence is None else not native_failure))
        if remove_failed:
            state['failed_requests'] = [failed for failed in state['failed_requests']
                                       if digest(failed.system.to_dict()) != graph_key]
        elif (native_failure or (evidence is None and proposal.level == 'screen' and
              proposal.system in evaluator.cache and evaluator.cache[proposal.system] is None)):
            # Only completed native failures have cache entries. Fatal outcomes
            # and unresolved reservations raise before reaching this branch.
            failed = Proposal(proposal.system, proposal.warm.copy(), 'refinement',
                outputs=proposal.outputs, operation='failed_request_promotion',
                level=proposal.level, parents=proposal.parents)
            state['failed_requests'] = [item for item in state['failed_requests']
                                       if digest(item.system.to_dict()) != graph_key]
            state['failed_requests'].append(failed)
            evicted = None
            if len(state['failed_requests']) > self.controls['root_capacity']:
                evicted = digest(state['failed_requests'].pop(0).system.to_dict())
            self.record('failed_request_queued', graph_sha256=graph_key,
                        outputs=list(proposal.outputs), level=proposal.level,
                        queued_requests=len(state['failed_requests']),
                        queue_limit=self.controls['root_capacity'], evicted_graph_sha256=evicted)
        after = state['roots'].best_errors()
        improvement = float(np.sum(np.maximum(0., before[np.isfinite(before)] -
                                             after[np.isfinite(before)])))
        work = evaluator.stats
        # Persisted work units avoid wall-clock timing in scheduling decisions.
        cost = 1. + proposal.generation_cost + sum(max(0., work[key] - pending['stats'].get(key, 0)) for key in (
            'constant_nfev', 'constant_raw_training_predictions',
            'separable_linear_attempts', 'portfolio_feature_predictions'))
        if proposal.producer != 'refinement':
            feedback(state['scheduler'], proposal.producer, self.controls,
                     state['roots'].version, empty=candidate is None,
                     improvement=improvement, cost=cost)
        else:
            state['promotions'].add((graph_key, proposal.level))
        if self.policy is not None and proposal.trace is not None:
            reward = 0. if candidate is None else 1. / (1. + search_score(candidate.errors, self.cfg))
            state['policy_batch'].append((proposal.trace, reward))
            if len(state['policy_batch']) >= self.controls['policy_batch_size']:
                self.update_policy()
        self.s['operations'][proposal.operation] += 1
        self.s['successful'][proposal.operation] += int(candidate is not None)
        self.record('proposal_completed', producer=proposal.producer,
                    operation=proposal.operation, accepted=candidate is not None,
                    level=proposal.level, refine=proposal.refine,
                    training_improvement=improvement, recorded_work_cost=cost,
                    candidate_id=None if candidate is None else identity(candidate),
                    accepted_candidate_ids=[identity(value) for value in accepted],
                    archive_version=state['roots'].version)
        # All controller effects, including policy updates and credits, share
        # this snapshot. A crash before it replays the native journal once.
        state['pending'] = None
        self.save()

    def update_policy(self):
        batch = self.s['portfolio']['policy_batch']
        if not batch:
            return
        traces, rewards = zip(*batch)
        best = max(range(len(batch)), key=lambda j: rewards[j])
        elites = [traces[best]] if rewards[best] > 0 else []
        update = self.policy.update(list(traces), list(rewards), elites, self.cfg.risk_fraction,
                                    self.cfg.entropy_weight, self.cfg.imitation_weight)
        self.record('policy_update', producer='grammar', batch_size=len(batch),
                    reward_source='complete_training_system', imitation_systems=len(elites), **update)
        batch.clear()

    def run(self):
        torch.set_num_threads(1)
        state, evaluator = self.s['portfolio'], self.evaluator
        try:
            while self.s['phase'] != 'done':
                self.check_stop()
                if state['pending'] is not None:
                    self.complete_pending()
                    continue
                if self.total_attempts() >= self.cfg.max_evaluations:
                    state['stop_reason'] = 'global_fit_budget'
                    break
                if completed(self.s['target']):
                    state['stop_reason'] = 'training_target_reached'
                    break
                if state['proposal_count'] >= self.controls['proposal_limit']:
                    state['stop_reason'] = 'proposal_limit'
                    break
                state['proposal_count'] += 1
                proposal = self.propose()
                if proposal is None and choose(state['scheduler'], self.controls, state['roots'].version) is None:
                    proposal = self.promotion(force=True)
                    if proposal is None:
                        state['stop_reason'] = 'all_producers_exhausted'
                        break
                if proposal is None:
                    self.save()
                    continue
                if not self.reserve(proposal):
                    self.save()
            if self.s['phase'] != 'done':
                self.update_policy()
                self.s['phase'] = 'done'
                self.record('portfolio_finished', stop_reason=state['stop_reason'])
                self.save()
            if not evaluator.archive:
                raise RuntimeError('No legal complete system within the portfolio limits.')
            assert self.total_attempts() <= self.cfg.max_evaluations
            self.owner.portfolio_report_ = dict(
                stop_reason=state['stop_reason'], proposals=state['proposal_count'],
                scheduler=deepcopy(state['scheduler']),
                root_counts=[len(entries) for entries in state['roots'].roots],
                accepted_versions=len(state['accepted']),
                pending_failed_requests=len(state['failed_requests']), controls=deepcopy(self.controls))
            self.owner.budget_report_ = dict(
                mode='portfolio', time_is_stop_condition=False,
                global_evaluation_limit=self.cfg.max_evaluations,
                actual_evaluations=evaluator.calls, refinements=evaluator.stats['refinements'],
                total_fit_attempts=self.total_attempts(), completed=True,
                stop_reason=state['stop_reason'], journal_operations=evaluator.cursor,
                replayed_operations=evaluator.replayed_operations,
                portfolio=deepcopy(self.owner.portfolio_report_), execution=evaluator.execution_report())
            pool = list(evaluator.archive.values())
            return dict(evaluator=evaluator, policy=self.policy, start=self.started,
                        cpu_start=self.cpu_started, stage_limit=self.cfg.max_evaluations,
                        reference=self.s['reference'], reference_system=self.s['reference_system'],
                        reference_constants=self.s['reference_constants'], pool=pool, closure=None,
                        archive=pool, seeds=pool, seed_stats=[],
                        **{key: self.s[key] for key in ('operations', 'changed', 'novel', 'successful')})
        finally:
            try:
                evaluator.close()
            finally:
                self.store.close()
                torch.set_num_threads(self.original_threads)
