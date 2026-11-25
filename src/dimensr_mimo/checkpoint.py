"""Local search checkpoints and an atomic journal of completed evaluations.

Only open checkpoints produced by this implementation. Payloads contain Python
state; input/config/source identities are checked before loading that state.
"""
import hashlib
import io
import json
from pathlib import Path
import pickle
import sqlite3

import numpy as np

from .evaluation import Candidate, Evaluator
from .graph import System


class SearchStopped(RuntimeError):
    """A requested stop after the latest completed evaluation was persisted."""


class Checkpoint:
    def __init__(self, path, binding, resume=False):
        if path is None and resume:
            raise ValueError('Resume requires a checkpoint path.')
        if path is not None:
            path = Path(path)
            if path.exists() != resume:
                raise ValueError('Checkpoint exists; use resume=True, or choose a new path.')
            path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(':memory:' if path is None else str(path))
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''CREATE TABLE IF NOT EXISTS metadata(binding TEXT);
            CREATE TABLE IF NOT EXISTS objects(key TEXT PRIMARY KEY, payload BLOB);
            CREATE TABLE IF NOT EXISTS snapshot(id INTEGER PRIMARY KEY, payload BLOB);
            CREATE TABLE IF NOT EXISTS evaluations(id INTEGER PRIMARY KEY, request TEXT, payload BLOB);
            CREATE TABLE IF NOT EXISTS evaluation_work(id INTEGER PRIMARY KEY, request TEXT, payload BLOB);
            CREATE TABLE IF NOT EXISTS execution_events(id INTEGER PRIMARY KEY, payload BLOB);''')
        identity = json.dumps(binding, sort_keys=True)
        row = self.db.execute('SELECT binding FROM metadata').fetchone()
        if row is None:
            with self.db:
                self.db.execute('INSERT INTO metadata VALUES (?)', (identity,))
        elif row[0] != identity:
            self.db.close()
            raise ValueError('Checkpoint data, configuration or source identity does not match.')
        self.loaded = {}

    def encode(self, value):
        owner = self

        class Writer(pickle.Pickler):
            def persistent_id(self, obj):
                if isinstance(obj, (Candidate, System, np.ndarray)):
                    payload = pickle.dumps(obj, protocol=5)
                    key = hashlib.sha256(payload).hexdigest()
                    owner.db.execute('INSERT OR IGNORE INTO objects VALUES (?, ?)', (key, payload))
                    return key
                return None

        stream = io.BytesIO()
        Writer(stream, protocol=5).dump(value)
        return stream.getvalue()

    def decode(self, payload):
        owner = self

        class Reader(pickle.Unpickler):
            def persistent_load(self, key):
                if key not in owner.loaded:
                    row = owner.db.execute('SELECT payload FROM objects WHERE key=?', (key,)).fetchone()
                    if row is None or hashlib.sha256(row[0]).hexdigest() != key:
                        raise ValueError('Checkpoint object is missing or corrupt.')
                    owner.loaded[key] = pickle.loads(row[0])
                return owner.loaded[key]

        return Reader(io.BytesIO(payload)).load()

    def save(self, state):
        with self.db:
            payload = self.encode(state)
            self.db.execute('INSERT OR REPLACE INTO snapshot VALUES (1, ?)', (payload,))

    def load(self):
        row = self.db.execute('SELECT payload FROM snapshot WHERE id=1').fetchone()
        return None if row is None else self.decode(row[0])

    def operation(self, index, request, result=None):
        row = self.db.execute('SELECT request,payload FROM evaluations WHERE id=?', (index,)).fetchone()
        if row is not None:
            if row[0] != request:
                raise ValueError('Search replay diverged from its completed evaluation journal.')
            return self.decode(row[1])
        if result is not None:
            with self.db:
                payload = self.encode(result)
                self.db.execute('INSERT INTO evaluations VALUES (?, ?, ?)', (index, request, payload))
        return None

    def close(self):
        self.db.close()

    def work(self, index, request, value=None):
        """Durably reserve work and persist outcomes before coordinator merge."""
        row = self.db.execute('SELECT request,payload FROM evaluation_work WHERE id=?', (index,)).fetchone()
        if row is not None and row[0] != request:
            raise ValueError('Search replay diverged from its reserved evaluation journal.')
        if value is None:
            return None if row is None else self.decode(row[1])
        with self.db:
            payload = self.encode(value)
            self.db.execute('INSERT OR REPLACE INTO evaluation_work VALUES (?, ?, ?)', (index, request, payload))
        return value

    def execution_event(self, value):
        with self.db:
            self.db.execute('INSERT INTO execution_events(payload) VALUES (?)', (self.encode(value),))

    def execution_events(self):
        return [self.decode(row[0]) for row in self.db.execute('SELECT payload FROM execution_events ORDER BY id')]


class JournalEvaluator(Evaluator):
    def __init__(self, *args, checkpoint, stop_requested, elapsed, **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpoint, self.stop_requested, self.elapsed = checkpoint, stop_requested, elapsed
        self.cursor, self.ceiling, self.refining = 0, self.cfg.max_evaluations, False
        self.replayed_operations = 0
        self.last_batch_complete = True
        self._execution = None

    def expired(self):
        return not self.refining and self.calls >= self.ceiling

    def __call__(self, system, warm, refine=False):
        if self.cfg.final_selection == 'pareto':
            return self.evaluate_many([(system, warm)], refine=refine)[0]
        request = hashlib.sha256(json.dumps(dict(system=system.to_dict(), warm=np.asarray(warm).tolist(),
            refine=refine, max_nodes=self.cfg.max_nodes, max_depth=self.cfg.max_depth,
            max_nfev=self.cfg.constant_max_nfev, ceiling=self.ceiling), sort_keys=True).encode()).hexdigest()
        recorded = self.checkpoint.operation(self.cursor, request)
        if recorded is None:
            self.refining = refine
            try:
                candidate = super().__call__(system, warm, refine=refine)
            except Exception as error:
                self.stats['fatal_evaluation_errors'] += 1
                self.checkpoint.operation(self.cursor, request, dict(candidate=None, cached=system in self.cache,
                    cache_value=self.cache.get(system), calls=self.calls, stats=dict(self.stats),
                    elapsed_seconds=self.elapsed(), fatal_error=type(error).__name__+': '+str(error)))
                raise
            finally:
                self.refining = False
            recorded = dict(candidate=candidate, cached=system in self.cache,
                            cache_value=self.cache.get(system), calls=self.calls,
                            stats=dict(self.stats), elapsed_seconds=self.elapsed())
            self.checkpoint.operation(self.cursor, request, recorded)
        else:
            candidate = recorded['candidate']
            if getattr(self, 'observe_candidate', None) is not None:
                self.observe_candidate(candidate)
            self.calls = recorded['calls']
            self.stats.clear()
            self.stats.update(recorded['stats'])
            if recorded['cached']:
                self.cache[system] = recorded['cache_value']
            self.replayed_operations += 1
            if 'fatal_error' in recorded:
                raise RuntimeError('Previously recorded fatal evaluation; automatic retry is disabled: '+recorded['fatal_error'])
        self.cursor += 1
        if self.stop_requested is not None and self.stop_requested():
            raise SearchStopped('Stop requested; completed evaluation is in the checkpoint journal.')
        return candidate

    def evaluate_many(self, requests, refine=False):
        if self.cfg.final_selection != 'pareto':
            self.last_batch_complete = True
            results = []
            for system, warm in requests:
                cached = system in self.cache
                skipped = not (cached and not refine) and (
                    (not refine and self.calls >= self.ceiling) or
                    (not cached and self.calls >= self.cfg.max_evaluations))
                results.append(self(system, warm, refine=refine))
                self.last_batch_complete &= not skipped
            return results
        if self._execution is None:
            from .parallel_evaluation import SystemExecution
            self._execution = SystemExecution(self)
        return self._execution.evaluate_many(requests, refine=refine)

    def execution_state(self):
        return dict(cache=self.cache, calls=self.calls, stats=dict(self.stats), cursor=self.cursor,
                    last_batch_complete=self.last_batch_complete,
                    diagnostics={} if self._execution is None else self._execution.diagnostics)

    def restore_execution_state(self, state):
        if not state:
            return
        self.close()
        self._execution = None
        self.cache, self.calls, self.cursor = state['cache'], state['calls'], state['cursor']
        self.stats.clear()
        self.stats.update(state['stats'])
        self.last_batch_complete = state.get('last_batch_complete', True)
        if state.get('diagnostics'):
            from .parallel_evaluation import SystemExecution
            self._execution = SystemExecution(self)
            self._execution.diagnostics = dict(state['diagnostics'])

    def execution_report(self):
        return dict(parallel_mode=getattr(self.cfg, 'parallel_mode', 'serial'),
                    n_cpus=getattr(self.cfg, 'n_cpus', 1), evaluations=self.calls,
                    refinements=self.stats['refinements'], last_batch_complete=self.last_batch_complete,
                    replayed_operations=self.replayed_operations,
                    events=self.checkpoint.execution_events(),
                    records=[] if self._execution is None else [self._execution.diagnostics[k]
                        for k in sorted(self._execution.diagnostics)])

    def close(self):
        if self._execution is not None:
            self._execution.close()
