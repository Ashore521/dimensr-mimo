"""Training-only observation of the frozen 1.6.0 search, without decision changes."""
from contextlib import ExitStack
import hashlib
import json
import sys
import time
from unittest.mock import patch

from dimensr_mimo import regressor
from dimensr_mimo.evaluation import Evaluator
from dimensr_mimo.graph import in_bounds


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


class Trace:
    def __init__(self, stream, phase_stream=None):
        self.stream, self.phase_stream = stream, phase_stream
        self.graphs, self.states = {}, {}
        self.serial, self.pending = 0, None
        self.stack = ExitStack()

    def emit(self, kind, **data):
        self.serial += 1
        self.stream.write(json.dumps(dict(serial=self.serial, kind=kind, **data), allow_nan=False) + '\n')
        return self.serial

    def graph(self, system):
        if system not in self.graphs:
            payload = system.to_dict()
            self.graphs[system] = digest(payload)
            self.emit('graph', id=self.graphs[system], system=payload)
        return self.graphs[system]

    def state(self, candidate):
        if candidate is None:
            return None
        key = (candidate.system, tuple(candidate.constants))
        if key not in self.states:
            payload = candidate.to_dict()
            payload['system'] = self.graph(candidate.system)
            self.states[key] = digest(payload)
            self.emit('state', id=self.states[key], candidate=payload)
        return self.states[key]

    def __enter__(self):
        original_evaluate, original_offspring = Evaluator.__call__, regressor.offspring

        def evaluate(evaluator, system, warm, refine=False):
            caller = sys._getframe(1)
            proposal = self.pending[0] if caller.f_code.co_name == 'evolve' and self.pending is not None and self.pending[1] == system else None
            self.pending = None
            cached, calls = system in evaluator.cache, evaluator.calls
            before = dict(evaluator.stats)
            candidate = original_evaluate(evaluator, system, warm, refine)
            self.emit('evaluation', proposal=proposal, source_line=caller.f_lineno,
                graph=self.graph(system), warm=list(map(float, warm)), cached=cached, refine=refine,
                calls_before=calls, calls_after=evaluator.calls, state=self.state(candidate),
                stats_delta={k:v-before.get(k, 0) for k, v in evaluator.stats.items() if v != before.get(k, 0)})
            return candidate

        def offspring(parent, donor, grammar, rng, operator=None):
            result = original_offspring(parent, donor, grammar, rng, operator)
            system, warm, op = result
            serial = self.emit('variation', parent=self.state(parent), donor=self.state(donor),
                operator=op, graph=self.graph(system), warm=list(map(float, warm)),
                in_bounds=in_bounds(system, grammar.cfg))
            self.pending = serial, system
            return result

        self.stack.enter_context(patch.object(Evaluator, '__call__', evaluate))
        self.stack.enter_context(patch.object(regressor, 'offspring', offspring))
        return self

    def phase(self, event):
        caller = sys._getframe(1)
        if caller.f_code.co_name != 'record' or 'population' not in caller.f_locals:
            raise RuntimeError('Observer requires the frozen regressor record callback.')
        self.emit('phase', event=event, population=[self.state(c) for c in caller.f_locals['population']])
        if self.phase_stream is not None:
            self.phase_stream.write(json.dumps(dict(event, output=None, monotonic=time.monotonic())) + '\n')
            self.phase_stream.flush()
        self.pending = None
        self.stream.flush()

    def finish(self, model):
        self.emit('final', pool=[dict(graph=digest(c['system']), constants=c['constants']) for c in model.candidates_],
            selected=dict(graph=digest(model.best_.system.to_dict()), constants=model.best_.constants.tolist()),
            search_fingerprint=model.search_fingerprint_)
        self.stream.flush()

    def __exit__(self, *args):
        self.stack.close()
        self.stream.flush()
