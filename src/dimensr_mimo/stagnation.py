"""Training-only historical progress anchors and finite stage patience."""
from fractions import Fraction

import numpy as np

from .pareto_selection import _within


DEFAULTS = dict(initialization_evaluations=301, initialization_seed_limit=290,
    exploration_warmup=20, exploration_interval=5, exploration_patience=6,
    nsga_interval=5, nsga_patience=4, residual_interval=5, residual_patience=4,
    sharing_patience=3, error_absolute=1e-4, error_relative=.005,
    complexity_relative=.01, max_restarts=2, restart_keep_fraction=.5)


def settings(overrides=None):
    values = dict(DEFAULTS)
    if overrides is not None:
        if not isinstance(overrides, dict) or set(overrides) - set(values):
            raise ValueError('Unknown stagnation settings.')
        values.update(overrides)
    for key, value in values.items():
        if key in ('error_absolute', 'error_relative', 'complexity_relative', 'restart_keep_fraction'):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError(key + ' must be finite.')
            if key == 'restart_keep_fraction':
                valid = 0 < value < 1
            elif key == 'error_absolute':
                valid = value > 0
            else:
                valid = 0 < value < 1
            if not valid:
                raise ValueError('Invalid ' + key)
        elif type(value) is not int or value < (0 if key in ('max_restarts', 'exploration_warmup') else 1):
            raise ValueError(key + ' has an invalid finite count.')
    return values


class HistoricalMonitor:
    def __init__(self, controls, protection_relative=.10, protection_absolute=.02):
        self.controls = controls
        self.protection_relative, self.protection_absolute = protection_relative, protection_absolute
        self.records = {}
        self.reference = None
        self.active = False
        self.best_a = self.best_b = self.best_c = None
        self.anchor_a = self.anchor_b = self.anchor_c = None
        self.c_ever_available = False
        self.sequence = 0
        self.metric_sequences = dict(A=0, B=0, C=0)
        self.events = []
        self.rebases = []
        self.minimum_violation = None
        self.protected_count = 0

    def _protection(self, row):
        if self.reference is None:
            row['violation'], row['protected'] = None, False
        else:
            threshold = self.reference * (1 + self.protection_relative) + self.protection_absolute
            row['violation'] = float(np.maximum(np.asarray(row['errors']) - threshold, 0).sum())
            row['protected'] = row['violation'] == 0

    def _eligible(self, identity):
        if identity is None or self.best_a is None:
            return False
        row, reference = self.records[identity], self.records[self.best_a]
        return row['protected'] and _within(row['mean'], reference['mean']) and all(
            _within(value, bound) for value, bound in zip(row['exact_errors'], reference['exact_errors']))

    def _complexity_best(self):
        eligible = [key for key in self.records if self._eligible(key)]
        return min(eligible, key=lambda key: (self.records[key]['complexity'],
                   self.records[key]['mean'], key)) if eligible else None

    def initialize(self, reference):
        if self.reference is not None:
            np.testing.assert_array_equal(self.reference, reference)
            return
        self.reference = np.array(reference, dtype=float, copy=True)
        for row in self.records.values():
            self._protection(row)
        self.protected_count = sum(row['protected'] for row in self.records.values())
        self.minimum_violation = min((row['violation'] for row in self.records.values()), default=None)
        self.best_c = self._complexity_best()
        self.anchor_a = self.records[self.best_a]['mean'] if self.best_a else None
        self.anchor_b = self.records[self.best_b]['maximum'] if self.best_b else None
        self.anchor_c = self.best_c
        self.c_ever_available = self.best_c is not None
        self.active = True

    def add(self, identity, candidate):
        if identity in self.records:
            return []
        errors = tuple(Fraction.from_float(float(value)) for value in candidate.errors)
        row = dict(errors=candidate.errors.tolist(), exact_errors=errors,
                   mean=sum(errors, Fraction()) / len(errors), maximum=max(errors),
                   complexity=float(candidate.complexity))
        self._protection(row)
        self.records[identity] = row
        previous_a = self.best_a
        self.best_a = min((key for key in (self.best_a, identity) if key is not None),
            key=lambda key: (self.records[key]['mean'], self.records[key]['complexity'], key))
        self.best_b = min((key for key in (self.best_b, identity) if key is not None),
            key=lambda key: (self.records[key]['maximum'], self.records[key]['mean'], key))
        if not self.active:
            return []
        self.protected_count += int(row['protected'])
        self.minimum_violation = min(self.minimum_violation, row['violation'])
        if self.best_a != previous_a:
            self.best_c = self._complexity_best()
        elif self._eligible(identity) and (self.best_c is None or
                (row['complexity'], row['mean'], identity) <
                (self.records[self.best_c]['complexity'], self.records[self.best_c]['mean'], self.best_c)):
            self.best_c = identity
        progress = []
        absolute = Fraction(str(self.controls['error_absolute']))
        relative = Fraction(str(self.controls['error_relative']))
        for label, best, anchor_name, metric in (
                ('A', self.best_a, 'anchor_a', 'mean'), ('B', self.best_b, 'anchor_b', 'maximum')):
            anchor = getattr(self, anchor_name)
            value = self.records[best][metric]
            if anchor - value >= max(absolute, anchor * relative):
                setattr(self, anchor_name, value)
                progress.append(label)
        if self.best_c is not None:
            if not self.c_ever_available:
                self.c_ever_available = True
                self.anchor_c = self.best_c
                progress.append('C')
            elif not self._eligible(self.anchor_c):
                self.rebases.append(dict(previous=self.anchor_c, current=self.best_c, reason='accuracy_reference_changed'))
                self.anchor_c = self.best_c
            else:
                anchor = Fraction.from_float(self.records[self.anchor_c]['complexity'])
                value = Fraction.from_float(self.records[self.best_c]['complexity'])
                if anchor - value >= anchor * Fraction(str(self.controls['complexity_relative'])) and value < anchor:
                    self.anchor_c = self.best_c
                    progress.append('C')
        if progress:
            self.sequence += 1
            for label in progress:
                self.metric_sequences[label] += 1
            self.events.append(dict(sequence=self.sequence, metrics=progress, candidate=identity,
                                    best_a=self.best_a, best_b=self.best_b, best_c=self.best_c))
        return progress

    def report(self):
        def candidate(identity):
            if identity is None:
                return None
            row = self.records[identity]
            return dict(candidate_id=identity, mean_nrmse=float(row['mean']), maximum_nrmse=float(row['maximum']),
                        errors=row['errors'], complexity=row['complexity'],
                        protected=row['protected'], violation=row['violation'])
        return dict(A=candidate(self.best_a), B=candidate(self.best_b), C=candidate(self.best_c),
                    anchor_a=None if self.anchor_a is None else float(self.anchor_a),
                    anchor_b=None if self.anchor_b is None else float(self.anchor_b), anchor_c=self.anchor_c,
                    progress_sequence=self.sequence, protected_candidates=self.protected_count,
                    minimum_violation=self.minimum_violation, candidate_versions=len(self.records),
                    reference_frozen=self.reference is not None, complexity_rebases=len(self.rebases))


class Patience:
    def __init__(self, interval, patience, sequence, warmup=0):
        self.interval, self.patience, self.warmup = interval, patience, warmup
        self.units = self.checks = self.stale = 0
        self.last_sequence = sequence
        self.history = []

    def check(self, sequence, partial=False):
        improved = sequence > self.last_sequence
        self.stale = 0 if improved else self.stale + 1
        self.last_sequence = sequence
        self.checks += 1
        entry = dict(units=self.units, check=self.checks, improved=improved,
                     stale_checks=self.stale, partial=partial, progress_sequence=sequence)
        self.history.append(entry)
        return entry

    def advance(self, sequence, count=1):
        self.units += count
        if self.warmup and self.units == self.warmup:
            self.last_sequence = sequence
            return None
        if self.units > self.warmup and (self.units - self.warmup) % self.interval == 0:
            return self.check(sequence)
        return None

    @property
    def stopped(self):
        return self.stale >= self.patience
