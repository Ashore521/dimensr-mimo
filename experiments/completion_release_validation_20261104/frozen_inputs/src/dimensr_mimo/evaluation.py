"""Whole-system training evaluation. Validation labels are not accepted here."""
from dataclasses import dataclass
from collections import Counter
import hashlib
import json
import time
import numpy as np
from ._single.evaluation import target_scale
from ._single.domains import domain_probes
from .graph import in_bounds
from .fitting import fit_constants
from .complexity import local_rademacher, mixed_complexity


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
        if system not in self.cache:
            self.calls += 1
        else:
            self.stats['refinements'] += 1
        started = time.monotonic()
        def accept(c, pred):
            if not self.admissible(pred) or not self.admissible(system.evaluate(self.probes, c)):
                self.stats['point_rejections'] += 1
                return False
            if self.bounds is not None:
                audits = system.audit(c, self.bounds, self.cfg.interval_max_boxes, self.cfg.output_bounds)
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
            errors = np.sqrt(np.mean(((system.evaluate(self.X, constants) - self.Y) / self.scales) ** 2, axis=0))
            rad, scaled = (local_rademacher(system, constants, self.X, self.scales, self.signs,
                                           self.cfg.perturbation_radius) if self.cfg.use_rademacher else (0., 0.))
            self.stats['rademacher_systems'] += int(self.cfg.use_rademacher)
            candidate = Candidate(system, constants, errors,
                                  mixed_complexity(system, scaled, self.cfg.complexity_weights), rad, scaled)
        else:
            self.stats['failed_systems'] += 1
        previous = self.cache.get(system)
        if previous is not None and (candidate is None or previous.error <= candidate.error):
            candidate = previous
        self.cache[system] = candidate
        self.stats['evaluation_seconds'] += time.monotonic() - started
        return candidate
