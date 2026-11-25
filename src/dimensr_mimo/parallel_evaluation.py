"""Bounded whole-system fitting with a durable, ordered coordinator."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
import copy
import hashlib
import json
import multiprocessing
import os
import time

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from .evaluation import Evaluator, search_score
from .graph import in_bounds


_WORKER = None
_THREAD_LIMITER = None


def _make_evaluator(context):
    evaluator = Evaluator.__new__(Evaluator)
    evaluator.__dict__.update(context)
    evaluator.cache, evaluator.stats, evaluator.calls = {}, Counter(), 0
    evaluator.deadline = None
    return evaluator


def _initialize_worker(context):
    global _WORKER, _THREAD_LIMITER
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    _THREAD_LIMITER = threadpool_limits(limits=1)
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _WORKER = _make_evaluator(context)


def _fit(evaluator, system, warm, cfg):
    cpu_started = time.process_time()
    wall_started = time.perf_counter()
    evaluator.cfg = cfg
    evaluator.cache.clear()
    evaluator.stats.clear()
    evaluator.calls = 0
    try:
        if cfg.search_schedule == 'portfolio':
            from .portfolio_fitting import resolve_fit_config
            evaluator.cfg = resolve_fit_config(system, cfg)
        candidate = Evaluator.__call__(evaluator, system, warm)
        outcome = dict(candidate=candidate,
                    status='completed' if candidate is not None else 'fit_failed', error=None)
    except Exception as error:
        evaluator.stats['fatal_evaluation_errors'] += 1
        evaluator.stats['failed_systems'] += 1
        outcome = dict(candidate=None, status='worker_failure',
                    error=type(error).__name__ + ': ' + str(error))
    evaluator.stats['fit_cpu_seconds'] += time.process_time() - cpu_started
    evaluator.stats['fit_wall_seconds'] += time.perf_counter() - wall_started
    outcome['stats'] = dict(evaluator.stats)
    return outcome


def _worker_fit(system, warm, cfg):
    import torch
    outcome = _fit(_WORKER, system, warm, cfg)
    outcome['threads'] = dict(torch=torch.get_num_threads(),
                             libraries=[dict(api=item['internal_api'], threads=item['num_threads'])
                                        for item in threadpool_info()])
    return outcome


class SystemExecution:
    def __init__(self, evaluator):
        self.owner = evaluator
        self.pool = None
        self.serial = None
        self.diagnostics = {}

    def _context(self):
        e = self.owner
        return {key: getattr(e, key) for key in
                ('X', 'Y', 'cfg', 'scales', 'probes', 'domain', 'bounds', 'signs')}

    def _request(self, system, warm, refine):
        e = self.owner
        payload = dict(system=system.to_dict(), warm=np.asarray(warm).tolist(), refine=refine,
                       max_nodes=e.cfg.max_nodes, max_depth=e.cfg.max_depth,
                       max_nfev=e.cfg.constant_max_nfev, ceiling=e.ceiling)
        if e.cfg.search_schedule == 'portfolio':
            from .portfolio_fitting import request_identity
            payload['fit_profile'] = request_identity(system, e.cfg)
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _reserve(self, system, refine, planned, new_count, refinement_count):
        e = self.owner
        if system in planned:
            return dict(kind='duplicate', source=planned[system], cost=0, refinement=0)
        if system in e.cache and not refine:
            return dict(kind='cache', cost=0, refinement=0)
        cost = int(system not in e.cache)
        portfolio = e.cfg.search_schedule == 'portfolio'
        if portfolio and e.calls + e.stats['refinements'] + new_count + refinement_count >= e.cfg.max_evaluations:
            return dict(kind='budget', cost=0, refinement=0)
        if cost and e.calls + new_count >= min(e.ceiling, e.cfg.max_evaluations):
            return dict(kind='budget', cost=0, refinement=0)
        if refine and e.stats['refinements'] + refinement_count >= e.cfg.polish_candidates:
            return dict(kind='refinement_budget', cost=0, refinement=0)
        if not in_bounds(system, e.cfg):
            work = dict(kind='structure_screening', cost=0, refinement=0)
            if portfolio:
                from .capacity import rejection, shape
                work.update(structure_reason=rejection(system, e.cfg.structure_limits, e.cfg),
                            shape=shape(system, e.cfg))
            return work
        if portfolio:
            from .portfolio_fitting import resolve_fit_config
            original = e.cfg
            try:
                e.cfg = resolve_fit_config(system, original)
                memory = e.working_memory(system)
            finally:
                e.cfg = original
        else:
            memory = e.working_memory(system)
        if not memory['allowed']:
            return dict(kind='resource_screening', cost=0, refinement=0, memory=memory)
        graph_digest = hashlib.sha256(json.dumps(system.to_dict(), sort_keys=True).encode()).hexdigest()
        work = dict(kind='fit', cost=cost, refinement=int(refine), outcome=None,
                    graph_digest=graph_digest, memory=memory,
                    seed_entropy=[e.cfg.random_state, int.from_bytes(bytes.fromhex(graph_digest)[:8], 'little')])
        if portfolio:
            from .portfolio_fitting import request_identity
            work['fit_profile'] = request_identity(system, e.cfg)
        return work

    def _persist_failure(self, index, request, work, error):
        work['outcome'] = dict(candidate=None, status='worker_failure',
                               stats=dict(fatal_evaluation_errors=1, failed_systems=1),
                               error=type(error).__name__ + ': ' + str(error))
        self.owner.checkpoint.work(index, request, work)

    def _execute(self, jobs):
        if not jobs:
            return
        e = self.owner
        cfg = copy.deepcopy(e.cfg)
        if cfg.parallel_mode == 'serial':
            if self.serial is None:
                self.serial = _make_evaluator(self._context())
            for index, request, work, system, warm in jobs:
                with threadpool_limits(limits=1):
                    work['outcome'] = _fit(self.serial, system, warm, cfg)
                e.checkpoint.work(index, request, work)
            return
        futures = {}
        try:
            if self.pool is None:
                self.pool = ProcessPoolExecutor(max_workers=cfg.n_cpus,
                    mp_context=multiprocessing.get_context('spawn'),
                    initializer=_initialize_worker, initargs=(self._context(),))
        except Exception as error:
            for index, request, work, _, _ in jobs:
                self._persist_failure(index, request, work, error)
            return
        broken = False
        for index, request, work, system, warm in jobs:
            try:
                future = self.pool.submit(_worker_fit, system, warm, cfg)
                futures[future] = (index, request, work)
            except Exception as error:
                broken |= isinstance(error, BrokenProcessPool)
                self._persist_failure(index, request, work, error)
        for future in as_completed(futures):
            index, request, work = futures[future]
            try:
                work['outcome'] = future.result()
            except Exception as error:
                broken |= isinstance(error, BrokenProcessPool)
                self._persist_failure(index, request, work, error)
            else:
                e.checkpoint.work(index, request, work)
        if broken:
            self.close()

    def _merge(self, index, request, work, system):
        e = self.owner
        kind = work['kind']
        e.stats['proposals'] += 1
        diagnostic = dict(operation=index, status=kind, **{key: work[key] for key in
            ('memory', 'graph_digest', 'seed_entropy', 'fit_profile', 'structure_reason', 'shape') if key in work})
        candidate = None
        if kind == 'fit':
            outcome = work['outcome']
            if outcome is None:
                raise RuntimeError('Reserved evaluation has no persisted worker outcome.')
            e.calls += work['cost']
            e.stats['refinements'] += work['refinement']
            e.stats.update({key: value for key, value in outcome['stats'].items()
                            if key not in ('proposals', 'cache_hits', 'refinements')})
            candidate = outcome['candidate']
            if getattr(e, 'observe_candidate', None) is not None:
                e.observe_candidate(candidate)
            previous = e.cache.get(system)
            if previous is not None and (candidate is None or
                    (search_score(previous.errors, e.cfg), previous.error) <=
                    (search_score(candidate.errors, e.cfg), candidate.error)):
                candidate = previous
            e.cache[system] = candidate
            diagnostic.update(status=outcome['status'], error=outcome['error'])
            if e.cfg.search_schedule == 'portfolio':
                from .portfolio_fitting import outcome_state
                diagnostic['fit_state'] = outcome_state(outcome)
            if 'threads' in outcome:
                diagnostic['threads'] = outcome['threads']
        elif kind in ('cache', 'duplicate'):
            candidate = e.cache[system]
            e.stats['cache_hits'] += 1
        elif kind in ('budget', 'refinement_budget'):
            e.last_batch_complete = False
            e.stats[kind + '_skips'] += 1
        elif kind == 'resource_screening':
            e.stats['resource_screening_rejections'] += 1
        elif kind == 'structure_screening':
            e.stats['structure_bound_rejections'] += 1
        self.diagnostics[index] = diagnostic
        recorded = dict(candidate=candidate, cached=system in e.cache,
                        cache_value=e.cache.get(system), calls=e.calls, stats=dict(e.stats),
                        elapsed_seconds=e.elapsed(), diagnostic=diagnostic,
                        complete=kind not in ('budget', 'refinement_budget'))
        e.checkpoint.operation(index, request, recorded)
        if (e.cfg.search_schedule == 'portfolio' or getattr(e.cfg, '_strict_native_outcomes', False)) and diagnostic['status'] == 'worker_failure':
            raise RuntimeError('Fatal portfolio worker outcome; automatic retry is disabled: ' + str(diagnostic['error']))
        return candidate

    def evaluate_many(self, requests, refine=False):
        e = self.owner
        requests = list(requests)
        e.last_batch_complete = True
        results = []
        width = e.cfg.n_cpus if e.cfg.parallel_mode == 'process' else 1
        for start in range(0, len(requests), width):
            wave, jobs, planned = [], [], {}
            new_count = refinement_count = 0
            base = e.cursor
            # Replay merged operations before allocating any additional fits.
            for offset, (system, warm) in enumerate(requests[start:start + width]):
                index = base + offset
                request = self._request(system, warm, refine)
                recorded = e.checkpoint.operation(index, request)
                if recorded is not None:
                    if wave:
                        raise ValueError('Completed evaluation journal contains a gap.')
                    results.append(self._replay(index, system, recorded))
                    e.cursor += 1
                    continue
                work = e.checkpoint.work(index, request)
                if work is None:
                    work = self._reserve(system, refine, planned, new_count, refinement_count)
                    e.checkpoint.work(index, request, work)
                elif (e.cfg.search_schedule == 'portfolio' or getattr(e.cfg, '_strict_native_outcomes', False)) and work['kind'] == 'fit' and work['outcome'] is None:
                    raise RuntimeError('Unresolved portfolio fit reservation; automatic retry is disabled.')
                if work['kind'] == 'fit':
                    planned[system] = index
                    new_count += work['cost']
                    refinement_count += work['refinement']
                    if work['outcome'] is None:
                        jobs.append((index, request, work, system, warm))
                wave.append((index, request, work, system))
            self._execute(jobs)
            for index, request, work, system in wave:
                results.append(self._merge(index, request, work, system))
                e.cursor += 1
            if e.stop_requested is not None and e.stop_requested():
                from .checkpoint import SearchStopped
                e.checkpoint.execution_event(dict(status='stop_requested', cursor=e.cursor,
                    evaluations=e.calls, elapsed_seconds=e.elapsed()))
                raise SearchStopped('Stop requested; the finite evaluation wave is persisted.')
            if e.cfg.search_schedule == 'stagnation' and e.cfg.stop_metric is not None:
                from .stopping import metric_values, target_vector
                targets = target_vector(e.cfg.stop_metric, e.cfg.stop_target, e.Y.shape[1])
                for candidate in results[start:]:
                    if candidate is None:
                        continue
                    values = metric_values(candidate, e.cfg.stop_metric, e.scales, np.var(e.Y, axis=0))
                    reached = values >= targets if e.cfg.stop_metric == 'r2' else values <= targets
                    if np.isfinite(values).all() and reached.all():
                        from .selection import digest
                        e.target_hit = dict(metric=e.cfg.stop_metric, values=values.tolist(),
                            candidate_id=digest(dict(system=candidate.system.to_dict(),
                                constants=[0. if v == 0 else float(v) for v in candidate.constants])),
                            evaluations=e.calls, graph=hashlib.sha256(json.dumps(candidate.system.to_dict(), sort_keys=True).encode()).hexdigest())
                        e.ceiling = e.calls
                        e.last_batch_complete = start + width >= len(requests)
                        results.extend([None] * (len(requests) - len(results)))
                        return results
        return results

    def _replay(self, index, system, recorded):
        e = self.owner
        e.calls = recorded['calls']
        e.stats.clear()
        e.stats.update(recorded['stats'])
        if getattr(e, 'observe_candidate', None) is not None:
            e.observe_candidate(recorded['candidate'])
        if recorded['cached']:
            e.cache[system] = recorded['cache_value']
        self.diagnostics[index] = recorded['diagnostic']
        e.last_batch_complete &= recorded['complete']
        e.replayed_operations += 1
        if (e.cfg.search_schedule == 'portfolio' or getattr(e.cfg, '_strict_native_outcomes', False)) and recorded['diagnostic']['status'] == 'worker_failure':
            raise RuntimeError('Previously recorded fatal portfolio outcome; automatic retry is disabled.')
        return recorded['candidate']

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=False)
            self.pool = None
