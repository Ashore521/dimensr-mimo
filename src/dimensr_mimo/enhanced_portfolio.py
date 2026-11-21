"""Serial Enhanced prefix and Portfolio continuation under one outer identity."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from .portfolio_proposals import identity
from .portfolio_search import PortfolioSearch
from .selection import digest
from .stagnation_search import StagnationSearch


def ledger_identity(path):
    """Hash committed ledger bytes without opening its SQLite or pickle data."""
    checksum, size = hashlib.sha256(), 0
    try:
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                checksum.update(block)
                size += len(block)
    except FileNotFoundError as error:
        raise ValueError('Enhanced prefix ledger is missing.') from error
    return dict(sha256=checksum.hexdigest(), bytes=size)


def prefix_config(outer):
    """Validate a full child configuration and its shared scientific contract."""
    from .config import SearchConfig
    value = outer.enhanced_prefix
    if isinstance(value, dict) and 'compression_r2_drop' not in value:
        value = dict(value, compression_r2_drop=None)
    if not isinstance(value, dict) or set(value) != set(asdict(SearchConfig())):
        raise ValueError('Enhanced prefix requires a complete SearchConfig dictionary.')
    if value.get('enhanced_prefix') is not None:
        raise ValueError('Nested Enhanced prefixes are prohibited.')
    child = SearchConfig(**deepcopy(value))
    child.validate()
    if (child.search_schedule != 'stagnation' or child.polish_candidates != 0 or
            child.max_evaluations > outer.max_evaluations or child.structure_limits is None or
            child.parallel_mode != 'serial' or child.n_cpus != 1):
        raise ValueError('Enhanced prefix needs serial stagnation, fixed capacity, zero refinement and a bounded sub-budget.')
    independent = {
        'search_schedule', 'max_evaluations', 'polish_candidates', 'portfolio_settings',
        'enhanced_prefix', 'stagnation_settings', 'stage_budgets', 'stage_windows',
        'constant_fit_mode', 'constant_max_nfev', 'constant_prediction_limit',
        'constant_jacobian', 'constant_focus_worst', 'residual_guidance',
        'compact_seed_terms', 'compact_elites', 'imitation_scan_limit', 'compression_r2_drop'}
    outer_values = asdict(outer)
    mismatch = [key for key, item in asdict(child).items() if key not in independent and
                json.dumps(item, sort_keys=True) != json.dumps(outer_values[key], sort_keys=True)]
    if mismatch:
        raise ValueError('Enhanced prefix shared configuration mismatch: ' + ', '.join(mismatch))
    child._strict_native_outcomes = True
    return child


class PrefixSearch(StagnationSearch):
    """The actual Enhanced controller, with an explicit empty-prefix boundary."""

    def __init__(self, *args, **kwargs):
        try:
            super().__init__(*args, **kwargs)
        except BaseException:
            if hasattr(self, 'evaluator'):
                self.evaluator.close()
            if hasattr(self, 'store'):
                self.store.close()
            if hasattr(self, 'original_threads'):
                torch.set_num_threads(self.original_threads)
            raise

    def initialize(self):
        try:
            super().initialize()
        except RuntimeError as error:
            if str(error) != 'No legal initialization within the finite initialization/global evaluation budget.':
                raise
            self.s['exit_reason'] = 'no_legal_prefix_initialization'
            self.s.update(reference_system=None, reference_constants=None, pool=[])
            self.record('prefix_empty', stop_reason=self.s['exit_reason'])
            self.finish('initialization', self.s['exit_reason'], 'done')


class EnhancedPortfolioSearch(PortfolioSearch):
    """Child ledger first, then atomic import into the outer Portfolio snapshot.

    The outer binding already includes full source, configuration and data.
    The prefix has its own identically bound ledger. Its consumed calls are
    imported once, leaving total minus actual prefix calls for continuation.
    A reserved child fit without an outcome remains charged and blocks replay.
    """

    def __init__(self, owner, cfg, X, Y, X_val, Y_val, features, targets, constants,
                 extra, input_bounds, X_domain, callback, checkpoint_path, resume, stop_requested):
        self.prefix_path = None if checkpoint_path is None else Path(str(checkpoint_path) + '.enhanced.sqlite')
        if not resume and self.prefix_path is not None and self.prefix_path.exists():
            raise ValueError('Fresh outer checkpoint has an orphan Enhanced prefix ledger.')
        self.resuming = resume
        super().__init__(owner, cfg, X, Y, X_val, Y_val, features, targets, constants,
                         extra, input_bounds, X_domain, callback, checkpoint_path, resume, stop_requested)
        self.prefix_cfg = prefix_config(cfg)
        self.prefix_inputs = (X, Y, X_val, Y_val, features, targets, constants, extra,
                              input_bounds, X_domain)

    def import_prefix(self, child):
        """All accepted versions, binding and accounting commit in one snapshot."""
        state, evaluator = self.s['portfolio'], self.evaluator
        native = child.evaluator
        archive = list(native.archive.values())
        fits = child.store.db.execute('SELECT id,payload FROM evaluation_work ORDER BY id').fetchall()
        charges = []
        for operation, payload in fits:
            work = child.store.decode(payload)
            if work['kind'] == 'fit':
                if work['outcome'] is None:
                    raise RuntimeError('Enhanced handoff has an unresolved reservation.')
                if work['outcome']['status'] == 'worker_failure':
                    raise RuntimeError('Enhanced handoff has a fatal native outcome.')
                charges.append(dict(operation=operation, first=work['cost'], refinement=work['refinement']))
        if (sum(row['first'] for row in charges) != native.calls or
                any(row['refinement'] for row in charges) or
                native.calls > self.prefix_cfg.max_evaluations):
            raise RuntimeError('Enhanced prefix ledger does not reconcile with its explicit allocation.')
        evaluator.calls = native.calls
        evaluator.stats.update(native.stats)
        evaluator.cache.update(native.cache)
        for candidate in archive:
            key = identity(candidate)
            evaluator.archive[key] = candidate
            state['accepted'].append(key)
            state['roots'].add(candidate)
        # The working pool is bounded. The complete evaluator archive is the
        # final selection pool and retains every prefix version and champion.
        from .evaluation import search_score
        self.s['pool'] = sorted(archive, key=lambda c: (
            search_score(c.errors, self.cfg), c.complexity, identity(c)))[:self.cfg.population_size]
        self.s['population'] = self.s['pool']
        for name in ('reference', 'reference_system', 'reference_constants'):
            self.s[name] = child.s.get(name)
        for key, candidate in evaluator.archive.items():
            evaluator.monitor.add(key, candidate)
        if self.s['reference'] is not None:
            evaluator.monitor.initialize(self.s['reference'])
        if self.policy is not None:
            self.policy.load_state_dict(child.policy.state_dict())
            self.policy.optimizer.load_state_dict(child.policy.optimizer.state_dict())
            self.policy.generator.set_state(child.policy.generator.get_state())
        self.rng.bit_generator.state = deepcopy(child.rng.bit_generator.state)
        self.observe_target(archive, 0)
        binding = json.loads(child.store.db.execute('SELECT binding FROM metadata').fetchone()[0])
        handoff = dict(rule='unused_prefix_allowance_transfers_to_portfolio',
            prefix_allocation=self.prefix_cfg.max_evaluations, prefix_actual=native.calls,
            portfolio_allowance=self.cfg.max_evaluations - native.calls,
            global_limit=self.cfg.max_evaluations, prefix_refinements=native.stats['refinements'],
            prefix_binding_sha256=digest(binding),
            prefix_result_sha256=digest([candidate.to_dict() for candidate in archive]),
            prefix_candidate_ids=[identity(candidate) for candidate in archive],
            prefix_journal_operations=native.cursor, prefix_native_charges=charges,
            prefix_stop_reason=child.s.get('exit_reason'),
            prefix_stage_reports=deepcopy(child.s['reports']),
            prefix_execution=native.execution_report(),
            prefix_history=deepcopy(child.s['history']))
        # on_complete runs with the child connection open. Commit every child
        # write before hashing. The remaining child cleanup only closes native
        # services and the SQLite connection, preserving these committed bytes.
        child.store.db.commit()
        handoff['prefix_ledger_identity'] = None if self.prefix_path is None else ledger_identity(self.prefix_path)
        self.s['enhanced_handoff'] = handoff
        self.record('enhanced_handoff', prefix_actual=native.calls,
                    prefix_versions=len(archive), portfolio_allowance=handoff['portfolio_allowance'],
                    prefix_result_sha256=handoff['prefix_result_sha256'])
        self.save()

    def run_prefix(self):
        child_owner = SimpleNamespace()
        exists = self.prefix_path is not None and self.prefix_path.exists()
        if self.prefix_path is not None and self.s.get('enhanced_prefix_started') and not exists:
            raise ValueError('Enhanced prefix ledger is missing after its start was persisted.')
        child_resume = self.resuming and exists
        child = PrefixSearch(child_owner, self.prefix_cfg, *self.prefix_inputs,
            self.callback, self.prefix_path, child_resume, self.stop_requested)
        try:
            if not self.s.get('enhanced_prefix_started') and child.store.db.execute(
                    'SELECT EXISTS(SELECT 1 FROM evaluation_work)').fetchone()[0]:
                raise ValueError('Unstarted Enhanced prefix ledger contains unexpected native work.')
            # A constructed child has a committed, zero-work initialization.
            # Only mark it started after that file exists. No native operation
            # can run before this outer snapshot commits successfully.
            self.s['enhanced_prefix_started'] = True
            self.save()
            child.run(on_complete=self.import_prefix)
        finally:
            child.evaluator.close()
            child.store.close()

    def run(self):
        try:
            if 'enhanced_handoff' in self.s:
                if self.prefix_path is not None and ledger_identity(self.prefix_path) != self.s['enhanced_handoff'].get('prefix_ledger_identity'):
                    raise ValueError('Enhanced prefix ledger content identity does not match the persisted handoff.')
            else:
                torch.set_num_threads(1)
                self.run_prefix()
        except BaseException:
            self.evaluator.close()
            self.store.close()
            torch.set_num_threads(self.original_threads)
            raise
        result = super().run()
        handoff = deepcopy(self.s['enhanced_handoff'])
        self.owner.portfolio_report_['enhanced_handoff'] = handoff
        report = self.owner.budget_report_
        report.update(mode='enhanced_portfolio', enhanced_handoff=handoff,
            prefix_first_evaluations=handoff['prefix_actual'],
            portfolio_first_evaluations=self.evaluator.calls - handoff['prefix_actual'],
            portfolio_journal_operations=self.evaluator.cursor,
            prefix_journal_operations=handoff['prefix_journal_operations'],
            journal_operations_scope='portfolio_continuation_only',
            ledger_accounting=dict(
                prefix=dict(first_evaluations=handoff['prefix_actual'], refinements=0,
                            journal_operations=handoff['prefix_journal_operations']),
                portfolio=dict(first_evaluations=self.evaluator.calls - handoff['prefix_actual'],
                               refinements=self.evaluator.stats['refinements'],
                               journal_operations=self.evaluator.cursor)))
        report['portfolio'] = deepcopy(self.owner.portfolio_report_)
        return result
