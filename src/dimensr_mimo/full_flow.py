"""Run the released stagnation stages with an outer-round search threshold."""
import numpy as np

from dimensr_mimo.archive_evaluation import ArchiveEvaluator
from dimensr_mimo.portfolio_proposals import identity
from dimensr_mimo.stagnation_search import StagnationSearch
from dimensr_mimo.stopping import metric_values
from dimensr_mimo.structural_discovery import StructuralRootArchive, SupportSearch, recombine


class FullFlowEvaluator(ArchiveEvaluator):
    first_hit = None

    def evaluate_many(self, requests, refine=False):
        previous = len(self.numerical_work)
        result = super().evaluate_many(requests, refine=refine)
        if self.first_hit is None:
            for row in self.numerical_work[previous:]:
                index = row['operation']
                payload = self.checkpoint.db.execute(
                    'SELECT payload FROM evaluation_work WHERE id=?', (index,)).fetchone()[0]
                candidate = self.checkpoint.decode(payload)['outcome']['candidate']
                if candidate is None:
                    continue
                values = metric_values(candidate, 'r2', self.scales, self.Y.var(axis=0))
                if np.isfinite(values).all() and (values >= .90).all():
                    self.first_hit = dict(operation=index, cycle=self.cycle, phase=self.stage_tag,
                                          cumulative_first=row['evaluations'],
                                          training_r2=values.tolist(), candidate_id=identity(candidate))
                    break
        return result

    def execution_state(self):
        return {**super().execution_state(), 'first_flow_hit': self.first_hit}

    def restore_execution_state(self, state):
        super().restore_execution_state(state)
        self.first_hit = state.get('first_flow_hit')


class FullFlowSearch(StagnationSearch):
    evaluator_type = FullFlowEvaluator
    search_threshold = 5000
    structural_root_limit = 24

    def search_evaluations(self):
        e = self.evaluator
        return sum(row['end_evaluations'] - row['start_evaluations']
                   if 'end_evaluations' in row else e.calls - row['start_evaluations']
                   for row in self.s['reports'].values()
                   if row['phase'] in ('initialization', 'exploration'))

    def window_open(self, phase):
        if phase == 'exploration':
            return self.search_evaluations() < self.search_threshold and self.evaluator.first_hit is None
        return super().window_open(phase)

    def finish(self, phase, reason, following):
        if phase == 'exploration':
            if self.search_evaluations() >= self.search_threshold:
                reason = 'search_threshold_after_complete_round'
            elif self.evaluator.first_hit is not None:
                reason = 'training_target_after_complete_round'
        if phase == 'initialization' and self.evaluator.first_hit is not None:
            reason = 'training_target_after_initialization'
        super().finish(phase, reason, following)

    def evolve_pareto(self, population, phase, multiobjective):
        result = super().evolve_pareto(population, phase, multiobjective)
        if phase == 'exploration' and self.generation_complete:
            self.s['current_gp_generations'] = self.s.get('current_gp_generations', 0) + 1
        return result

    def record(self, phase, **details):
        if phase == 'exploration_epoch':
            end = self.search_evaluations()
            start = self.s.get('previous_round_search_end', end)
            if self.s['step'] == 1:
                current = self.s['reports'][str(self.s['cycle']) + ':exploration']
                start = end - (self.evaluator.calls - current['start_evaluations'])
            details.update(search_start=start, search_end=end,
                           gp_generations=self.s.pop('current_gp_generations', 0))
            self.s['previous_round_search_end'] = end
        super().record(phase, **details)

    def nsga(self):
        self.add_structural_candidates()
        super().nsga()

    def add_structural_candidates(self):
        s, e = self.s, self.evaluator
        if s.get('structural_completed'):
            return
        self.enter('structural')
        if 'structural_queue' not in s:
            screen = SupportSearch(self.grammar, self.X, self.Y, e.stats)
            seen = set(e.cache)
            queue = []
            for index in range(24):
                proposal = screen.propose(index, seen, e.stats)
                if proposal is not None:
                    queue.append(proposal)
                    seen.add(proposal.system)
            s.update(structural_queue=queue, structural_index=0, structural_assemblies=0)
            self.save()
        while s['structural_index'] < len(s['structural_queue']):
            proposal = s['structural_queue'][s['structural_index']]
            self.evaluate_structural(proposal)
            s['structural_index'] += 1
            self.save()
        bank = StructuralRootArchive(self.Y.shape[1], 8)
        for candidate in e.archive.values():
            bank.add(candidate)
        while s['structural_assemblies'] < self.structural_root_limit:
            proposal = recombine(bank, self.cfg, e.cache.keys() | s.get('structural_rejected', set()))
            if proposal is None:
                break
            candidate = self.evaluate_structural(proposal)
            if candidate is not None:
                bank.add(candidate)
            s['structural_assemblies'] += 1
            self.save()
        s['structural_completed'] = True
        self.prepare_pool()
        self.finish('structural', 'finite_structural_proposals_complete', 'nsga2')

    def evaluate_structural(self, proposal):
        e = self.evaluator
        self.s['attempts']['structural'] += 1
        before = e.calls
        candidate = e(proposal.system, proposal.warm)
        if proposal.system not in e.cache:
            self.s.setdefault('structural_rejected', set()).add(proposal.system)
        self.record('structural_proposal', operation=proposal.operation, parents=list(proposal.parents),
                    attempted=e.calls > before, accepted=candidate is not None)
        return candidate
