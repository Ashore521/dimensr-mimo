"""Preserve candidate versions and training-only metrics across replay and restarts."""
from copy import deepcopy

from .checkpoint import JournalEvaluator
from .selection import digest
from .stagnation import HistoricalMonitor, settings


class ArchiveEvaluator(JournalEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.archive = {}
        self.monitor = HistoricalMonitor(settings(self.cfg.stagnation_settings),
            self.cfg.degradation_relative, self.cfg.degradation_absolute)
        self.ingested_cursor = 0
        self.trajectory = []
        self.numerical_work = []
        self.stage_tag = 'initialization'
        self.cycle = 0
        self.on_progress = None
        self.target_hit = None

    def evaluate_many(self, requests, refine=False):
        try:
            return super().evaluate_many(requests, refine=refine)
        finally:
            rows = self.checkpoint.db.execute(
                'SELECT id,payload FROM evaluations WHERE id>=? AND id<? ORDER BY id',
                (self.ingested_cursor, self.cursor)).fetchall()
            for index, payload in rows:
                recorded = self.checkpoint.decode(payload)
                work = self.checkpoint.db.execute('SELECT payload FROM evaluation_work WHERE id=?', (index,)).fetchone()
                outcome = None if work is None else self.checkpoint.decode(work[0]).get('outcome')
                progress = []
                if outcome is not None:
                    candidate = outcome.get('candidate')
                    if candidate is not None:
                        identity = digest(dict(system=candidate.system.to_dict(),
                            constants=[0. if value == 0 else float(value) for value in candidate.constants]))
                        if identity not in self.archive:
                            self.archive[identity] = candidate
                            progress = self.monitor.add(identity, candidate)
                            if progress and self.on_progress is not None:
                                self.on_progress(progress)
                    stats = outcome.get('stats', {})
                    limit_hits = sum(stats.get(key, 0) for key in ('optimizer_status_0',
                        'constant_prediction_limit_hits', 'linear_solver_limit_hits', 'deadline_interruptions'))
                    converged = sum(stats.get('optimizer_status_' + str(k), 0) for k in (1, 2, 3, 4))
                    if limit_hits:
                        solver_status = 'evaluation_limited'
                    elif outcome['status'] != 'completed':
                        solver_status = outcome['status']
                    elif converged:
                        solver_status = 'converged'
                    elif stats.get('linear_fits', 0):
                        solver_status = 'affine_solve'
                    else:
                        solver_status = 'feasible_start_or_parameter_free'
                    self.numerical_work.append(dict(operation=index, cycle=self.cycle, stage=self.stage_tag,
                        evaluations=recorded['calls'], status=outcome['status'],
                        solver_status=solver_status,
                        solver_limit_hits=limit_hits,
                        converged_starts=converged,
                        constant_nfev=stats.get('constant_nfev', 0),
                        optimizer_nfev_complete=outcome['status'] != 'worker_failure' and not bool(stats.get('optimizer_incomplete_starts', 0)),
                        optimizer_starts=stats.get('optimizer_starts', 0),
                        optimizer_returns=stats.get('optimizer_returns', 0),
                        optimizer_incomplete_starts=stats.get('optimizer_incomplete_starts', 0),
                        first_structure_evaluation=bool(self.checkpoint.decode(work[0]).get('cost', 0)),
                        refinement=bool(self.checkpoint.decode(work[0]).get('refinement', 0)),
                        fit_wall_seconds=stats.get('fit_wall_seconds', 0),
                        fit_cpu_seconds=stats.get('fit_cpu_seconds', 0),
                        inner_linear_solves=stats.get('separable_linear_solves', 0),
                        inner_linear_attempts=stats.get('separable_linear_attempts', 0),
                        inner_linear_wall_seconds=stats.get('separable_linear_wall_seconds', 0),
                        inner_linear_cpu_seconds=stats.get('separable_linear_cpu_seconds', 0),
                        stopping_reasons=[key for key in ('optimizer_status_0', 'constant_prediction_limit_hits',
                            'linear_solver_limit_hits', 'deadline_interruptions', 'optimizer_exceptions') if stats.get(key, 0)],
                        residual_calls=stats.get('constant_residual_calls', 0),
                        optimizer_residual_call_attempts=stats.get('constant_solver_residual_calls', 0),
                        raw_training_predictions=stats.get('constant_raw_training_predictions', 0),
                        direct_affine_solves=stats.get('linear_fits', 0),
                        sample_node_values=stats.get('constant_sample_node_values', 0),
                        affine_coefficient_vectors=stats.get('affine_coefficient_vectors', 0),
                        nonlinear_dimensions=stats.get('nonlinear_dimensions', 0),
                        raw_system_predictions=stats.get('raw_system_predictions', 0),
                        raw_system_sample_node_values=stats.get('raw_system_sample_node_values', 0),
                        domain_tree_predictions=stats.get('domain_tree_predictions', 0),
                        domain_tree_sample_node_values=stats.get('domain_tree_sample_node_values', 0),
                        affine_design_predictions=stats.get('affine_design_predictions', 0),
                        error=outcome.get('error')))
                self.trajectory.append(dict(operation=index, cycle=self.cycle, stage=self.stage_tag,
                    evaluations=recorded['calls'], refinements=recorded['stats'].get('refinements', 0),
                    elapsed_seconds=recorded['elapsed_seconds'], effective_progress=progress,
                    **self.monitor.report()))
                self.ingested_cursor = index + 1

    def execution_state(self):
        value = super().execution_state()
        value['historical_archive'] = dict(archive=self.archive, monitor=self.monitor,
            ingested_cursor=self.ingested_cursor, trajectory=self.trajectory,
            numerical_work=self.numerical_work, stage_tag=self.stage_tag, cycle=self.cycle)
        value['historical_archive']['target_hit'] = self.target_hit
        return value

    def restore_execution_state(self, state):
        super().restore_execution_state(state)
        for key, value in state.get('historical_archive', {}).items():
            setattr(self, key, value)

    def execution_report(self):
        report = super().execution_report()
        report.update(historical_candidate_versions=len(self.archive),
                      training_monitor=deepcopy(self.monitor.report()),
                      numerical_limit_hits=sum(row['solver_limit_hits'] for row in self.numerical_work))
        return report
