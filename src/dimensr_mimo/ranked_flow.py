"""Training-only admission before the engineering flow's error/complexity selection."""
from math import ceil

from dimensr_mimo.pareto import ordered, retain_parameters
from dimensr_mimo.portfolio_proposals import identity
from dimensr_mimo.stopping import metric_values
from .full_flow import FullFlowSearch


def shortlist(archive, score, fraction, minimum):
    """Use the archive's model ID as a deterministic tie-breaker at the cutoff."""
    limit = min(len(archive), max(minimum, ceil(len(archive) * fraction)))
    return [archive[k] for k in sorted(archive, key=lambda k: (score(archive[k]), k))[:limit]]


class RankedFlowSearch(FullFlowSearch):
    def prepare_pool(self):
        if self.evaluator.first_hit is not None:
            return super().prepare_pool()
        cfg, state = self.cfg, self.s
        archive = self.evaluator.archive
        seeds = shortlist(archive, self.training_rank, cfg.candidate_fraction, cfg.min_candidates)
        state['archive'] = list(archive.values())
        state['seeds'] = seeds
        state['pool'] = list(seeds)
        # Only admitted systems reach the first accuracy/complexity sort.
        ranked = ordered(seeds, cfg, state['reference'])
        state['population'] = retain_parameters(ranked, cfg.population_size, cfg, state['reference'])
        if state['phase'] != 'exploration' or state.get('structural_completed'):
            self.display_event('admission', note=f'有效 {len(archive)} 组，入围 {len(seeds)} 组')
        state.setdefault('admission_reports', []).append(dict(
            cycle=state['cycle'], stage=state['phase'], available=len(archive), admitted=len(seeds),
            minimum_met=len(seeds) >= cfg.min_candidates,
            lowest_admitted_min_training_r2=-self.training_rank(seeds[-1])[0] if seeds else None))

    def training_rank(self, candidate):
        r2 = metric_values(candidate, 'r2', self.evaluator.scales, self.variance)
        return -float(r2.min()), candidate.error

    def restart(self):
        # A new cycle replaces state['pool']. Keep earlier admitted versions and
        # their descendants eligible for delivery across restarts and resumes.
        previous = self.s.setdefault('previous_cycle_candidates', {})
        for candidate in self.s.get('pool', ()):
            previous[identity(candidate)] = candidate
        super().restart()

    def delivery_candidates(self):
        if self.evaluator.first_hit is not None:
            return self.evaluator.archive.values()
        if not self.s.get('admission_reports'):
            # A budget or resource stop may end construction before prepare_pool.
            return shortlist(self.evaluator.archive, self.training_rank,
                             self.cfg.candidate_fraction, self.cfg.min_candidates)
        admitted = dict(self.s.get('previous_cycle_candidates', {}))
        for candidate in self.s['pool']:
            admitted[identity(candidate)] = candidate
        return admitted.values()
