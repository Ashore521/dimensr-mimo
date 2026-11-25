"""Whole-system training evaluation. Validation labels are not accepted here."""
from dataclasses import dataclass
from collections import Counter
import hashlib
import json
import sys
import time
import numpy as np
from ._single.evaluation import target_scale
from ._single.domains import domain_probes
from .graph import in_bounds
from .fitting import fit_constants
from .complexity import local_rademacher, mixed_complexity


def search_score(errors, cfg):
    """Search ranking only; stored errors and coefficient-fitting loss stay unchanged."""
    errors = np.asarray(errors, float)
    if cfg.search_objective == 'mean':
        return float(errors.mean())
    if cfg.search_objective == 'worst':
        return float(errors.max())
    ratios = errors / np.asarray(cfg.acceptance['nrmse'], float)
    return float(.5 * ratios.mean() + .5 * ratios.max())


@dataclass
class Candidate:
    system: object
    constants: np.ndarray
    errors: np.ndarray
    complexity: float
    rademacher: float
    rademacher_complexity: float

    @property
    def error(self):
        return float(np.mean(self.errors))

    def to_dict(self):
        return dict(system=self.system.to_dict(), constants=self.constants.tolist(),
                    errors=self.errors.tolist(), complexity=self.complexity,
                    rademacher=self.rademacher, rademacher_complexity=self.rademacher_complexity)


class Evaluator:
    def __init__(self, X, Y, cfg, *, input_bounds=None, X_domain=None, deadline=None):
        self.X, self.Y, self.cfg, self.deadline = X, Y, cfg, deadline
        self.scales = np.array([target_scale(y) for y in Y.T])
        self.cache, self.stats = {}, Counter()
        self.calls = 0
        self.probes, self.domain = domain_probes(X, cfg, input_bounds=input_bounds, X_domain=X_domain)
        self.bounds = self.domain['input_bounds'] if cfg.interval_checks else None
        rng = np.random.default_rng(np.random.SeedSequence([cfg.random_state, 721]))
        self.signs = rng.choice([-1., 1.], (cfg.rademacher_draws, Y.size))

    def expired(self):
        return self.deadline is not None and time.monotonic() >= self.deadline

    def admissible(self, pred):
        if not np.isfinite(pred).all():
            return False
        if self.cfg.output_bounds is not None:
            lo, hi = self.cfg.output_bounds
            if (lo is not None and np.any(pred < lo)) or (hi is not None and np.any(pred > hi)):
                return False
        return True

    def working_memory(self, system):
        """Conservative numeric-work estimate, excluding interpreter/library RSS."""
        samples, outputs, parameters = len(self.X), self.Y.shape[1], system.n_parameters
        rows = samples + len(self.probes)
        # DAG values, prediction/residual copies, dense Jacobian/SVD/linear work,
        # and parameter-space factorizations coexist during joint fitting.
        dag_bytes = 8 * rows * len(system.nodes) * 4
        residual_bytes = 8 * rows * outputs * 8
        jacobian_bytes = 8 * samples * outputs * parameters * 6
        parameter_bytes = 8 * parameters * parameters * 6
        signs_bytes = 8 * samples * outputs * self.cfg.rademacher_draws if self.cfg.use_rademacher else 0
        limit = float(getattr(self.cfg, 'max_working_memory_mb', 256)) * 1024 ** 2
        # Fitting/domain auditing expands shared DAG roots to trees. Saturate
        # before counting an exponentially large expansion or allocating it.
        cap = int(limit // 256) + 1
        expanded, depths = [], []
        for node in system.nodes:
            expanded.append(min(cap, 1 + sum(expanded[c] for c in node.children)))
            depths.append(1 + max((depths[c] for c in node.children), default=0))
        expanded_nodes = min(cap, sum(expanded[root] for root in system.roots))
        expanded_bytes = 256 * expanded_nodes
        affine_bytes = 0
        rademacher_jacobian_bytes = jacobian_bytes if self.cfg.use_rademacher else 0
        outer_parameters = parameters
        if self.cfg.constant_fit_mode == 'separable':
            from .capacity import parameter_partition
            linear, nonlinear = parameter_partition(system)
            outer_parameters = len(nonlinear)
            jacobian_bytes = 8 * samples * outputs * outer_parameters * 6
            linear = set(linear)
            dependencies = []
            for node in system.nodes:
                dependencies.append({node.parameter} & linear if node.parameter is not None else
                    set().union(*(dependencies[c] for c in node.children)))
            affine_bytes = 8 * samples * (sum(map(len, dependencies)) + outputs * len(linear) * 5)
        maximum_depth = max(depths[root] for root in system.roots)
        stack_safe = maximum_depth * 4 + 64 < sys.getrecursionlimit()
        # The full-parameter Rademacher SVD runs after fitting, not concurrently
        # with the affine design and reduced outer Jacobian.
        workspace = max(jacobian_bytes + affine_bytes, rademacher_jacobian_bytes)
        total = dag_bytes + residual_bytes + workspace + parameter_bytes + signs_bytes + expanded_bytes
        return dict(estimated_bytes=total, limit_bytes=limit, allowed=total <= limit and stack_safe,
                    reason='recursion_capacity' if not stack_safe else 'working_memory' if total > limit else None,
                    dag_nodes=len(system.nodes), samples=samples, outputs=outputs,
                    parameters=parameters, dag_bytes=dag_bytes, residual_bytes=residual_bytes,
                    jacobian_bytes=jacobian_bytes, parameter_bytes=parameter_bytes,
                    rademacher_signs_bytes=signs_bytes, expanded_nodes=expanded_nodes,
                    expanded_bytes=expanded_bytes, maximum_depth=maximum_depth,
                    affine_bytes=affine_bytes,
                    outer_parameters=outer_parameters, numerical_workspace_bytes=workspace,
                    rademacher_jacobian_bytes=rademacher_jacobian_bytes,
                    expanded_count_saturated=expanded_nodes == cap)

    def __call__(self, system, warm, refine=False):
        self.stats['proposals'] += 1
        if system in self.cache and not refine:
            self.stats['cache_hits'] += 1
            return self.cache[system]
        if self.expired() or (system not in self.cache and self.calls >= self.cfg.max_evaluations):
            return None
        if not in_bounds(system, self.cfg):
            self.stats['structure_bound_rejections'] += 1
            return None
        if self.cfg.final_selection == 'pareto' and not self.working_memory(system)['allowed']:
            self.stats['resource_screening_rejections'] += 1
            return None
        if system not in self.cache:
            self.calls += 1
        else:
            self.stats['refinements'] += 1
        started = time.monotonic()
        def accept(c, pred):
            if not np.all(np.abs(c) <= self.cfg.constant_bound):
                self.stats['parameter_bound_rejections'] += 1
                return False
            started_audit = time.process_time()
            self.stats['point_audit_calls'] += 1
            if not self.admissible(pred) or not self.admissible(system.evaluate(self.probes, c, stats=self.stats)):
                self.stats['point_audit_cpu_seconds'] += time.process_time() - started_audit
                self.stats['point_rejections'] += 1
                return False
            self.stats['point_audit_cpu_seconds'] += time.process_time() - started_audit
            if self.bounds is not None:
                started_audit = time.process_time()
                self.stats['interval_audit_calls'] += 1
                audits = system.audit(c, self.bounds, self.cfg.interval_max_boxes, self.cfg.output_bounds)
                self.stats['interval_audit_cpu_seconds'] += time.process_time() - started_audit
                for a in audits:
                    self.stats['interval_' + a['status']] += 1
                    self.stats['interval_boxes'] += a['boxes_checked']
                if any(a['status'] != 'enclosed' for a in audits):
                    return False
            return True
        digest = hashlib.sha256(json.dumps(system.to_dict(), sort_keys=True).encode()).digest()
        rng = np.random.default_rng(np.random.SeedSequence([self.cfg.random_state, int.from_bytes(digest[:8], 'little')]))
        constants = fit_constants(system, warm, self.X, self.Y, self.scales,
                                  np.vstack((self.X, self.probes)), self.cfg, rng,
                                  accept, self.stats, self.deadline)
        candidate = None
        if constants is not None:
            errors = np.sqrt(np.mean(((system.evaluate(self.X, constants, stats=self.stats) - self.Y) / self.scales) ** 2, axis=0))
            started_rademacher = time.process_time()
            rad, scaled = (local_rademacher(system, constants, self.X, self.scales, self.signs,
                                           self.cfg.perturbation_radius, stats=self.stats) if self.cfg.use_rademacher else (0., 0.))
            self.stats['rademacher_cpu_seconds'] += time.process_time() - started_rademacher
            self.stats['rademacher_systems'] += int(self.cfg.use_rademacher)
            candidate = Candidate(system, constants, errors,
                                  mixed_complexity(system, scaled, self.cfg.complexity_weights), rad, scaled)
            if self.cfg.final_selection == 'pareto' and not all(np.isfinite(value).all() for value in
                    (constants, errors, candidate.error, candidate.complexity, rad, scaled)):
                self.stats['nonfinite_candidate_rejections'] += 1
                self.stats['failed_systems'] += 1
                candidate = None
        else:
            self.stats['failed_systems'] += 1
        previous = self.cache.get(system)
        if previous is not None and (candidate is None or
                (search_score(previous.errors, self.cfg), previous.error) <=
                (search_score(candidate.errors, self.cfg), candidate.error)):
            candidate = previous
        if getattr(self, "observe_candidate", None) is not None:
            self.observe_candidate(candidate)
        self.cache[system] = candidate
        self.stats['evaluation_seconds'] += time.monotonic() - started
        return candidate
