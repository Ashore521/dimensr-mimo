"""One bounded system fit; writes each event immediately. Never reads test data."""
import argparse
import hashlib
from collections import Counter
from dataclasses import asdict, replace
import json
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from dimensr_mimo import DimenSRMIMORegressor, SearchConfig
from dimensr_mimo._single import DimenSRRegressor, SearchConfig as SingleConfig
from cases import design, generate


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def instrument_single(events):
    """Runtime instrumentation only; the vendored baseline source stays byte-identical."""
    from dimensr_mimo._single import regressor as reg, evaluation as ev, grammar as gr
    stats, state = Counter(), {'phase': 'initialization', 'nsga': False}
    restore = []
    def patch(obj, attr, wrapper):
        old = getattr(obj, attr)
        restore.append((obj, attr, old))
        setattr(obj, attr, wrapper(old))
    def timed(name):
        def wrap(fn):
            def call(*a, **k):
                t, cpu = time.monotonic(), time.process_time()
                try:
                    return fn(*a, **k)
                finally:
                    stats[name + '_seconds'] += time.monotonic() - t
                    stats[name + '_cpu_seconds'] += time.process_time() - cpu
                    stats[name + '_calls'] += 1
            return call
        return wrap
    def evaluation(fn):
        def call(self, *a, **k):
            phase = 'polish' if k.get('refine') else state['phase']
            t, before = time.monotonic(), self.calls
            try:
                return fn(self, *a, **k)
            finally:
                stats['evaluation_' + phase + '_seconds'] += time.monotonic() - t
                stats['evaluation_' + phase + '_new'] += self.calls - before
        return call
    def generation(fn):
        def call(self, rng, policy=None, **kw):
            if policy is not None:
                state['phase'] = 'neural'
            return timed('generation_' + state['phase'])(fn)(self, rng, policy, **kw)
        return call
    def variation(fn):
        def call(*a, **k):
            state['phase'] = 'nsga2' if state['nsga'] else 'gp'
            return timed('variation_' + state['phase'])(fn)(*a, **k)
        return call
    def selection(fn):
        def call(candidates, cfg, multiobjective):
            if multiobjective:
                state.update(phase='nsga2', nsga=True)
            return timed('selection')(fn)(candidates, cfg, multiobjective)
        return call
    def policy_update(fn):
        def call(self, *a, **k):
            before = torch.cat([p.detach().flatten().clone() for p in self.parameters()])
            result = timed('policy_update')(fn)(self, *a, **k)
            delta = float(torch.linalg.vector_norm(torch.cat([p.detach().flatten() for p in self.parameters()]) - before))
            events(dict(phase='policy_parameter_evidence', parameter_delta=delta, **result))
            return result
        return call
    patch(ev.Evaluator, '__call__', evaluation)
    patch(gr.Grammar, 'generate', generation)
    patch(reg, 'fitted_offspring', variation)
    patch(reg, '_ordered', selection)
    patch(reg.Policy, 'update', policy_update)
    for attr in ('least_squares', 'linear_solution', 'repair_domains', 'local_rademacher'):
        patch(ev, attr, timed(attr))
    return stats, lambda: [setattr(o, a, f) for o, a, f in reversed(restore)]


def run(folder, job, config):
    torch.set_num_threads(1)
    started, cpu = time.monotonic(), time.process_time()
    X, Y = generate(job['case'], 'train', job['seed'])
    V, Z = generate(job['case'], 'validation', job['seed'])
    spec = design(job['case'])
    output = None
    def event(e):
        e = dict(e, output=output, worker_elapsed=time.monotonic() - started, monotonic=time.monotonic())
        with (folder / 'events.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(e, allow_nan=False) + '\n')
    result = dict(job=job, status='failed', models=[], diagnostics=[], prediction_hashes=[], audits=[])
    def pin_predictions(model):
        result['prediction_hashes'].append({name: hashlib.sha256(model.predict(A).astype('<f8').tobytes()).hexdigest()
                                           for name, A in [('train', X), ('validation', V)]})
    try:
        cfg = SearchConfig(**config)
        cfg.random_state = job['seed']
        if job['method'] == 'independent':
            for output in range(Y.shape[1]):
                event(dict(phase='output_start', allowance_seconds=job['seconds'] / Y.shape[1]))
                settings = {k: v for k, v in asdict(cfg).items() if k in SingleConfig.__dataclass_fields__}
                settings['max_evaluations'] = cfg.max_evaluations // Y.shape[1]
                single = SingleConfig(**settings)
                instrumentation, restore = instrument_single(event)
                try:
                    m = DimenSRRegressor(single, spec['features'], spec['targets'][output], spec['constants'])
                    m.fit(X, Y[:, output], X_val=V, y_val=Z[:, output], input_bounds=spec['bounds'], callback=event)
                    path = f'model-{output}.json'; m.save(folder / path)
                    result['models'].append(path)
                    result['diagnostics'].append(dict(m.diagnostics_, instrumentation=dict(instrumentation)))
                    pin_predictions(m)
                    result['audits'].append(m.audit_domain(input_bounds=spec['bounds']))
                    event(dict(phase='output_complete'))
                finally:
                    restore()
        else:
            if job['method'] == 'no_neural':
                cfg.use_rnn = False
            elif job['method'] == 'no_gp':
                cfg.use_gp = False
            elif job['method'] == 'no_sharing':
                cfg.use_sharing = False
            cfg.wall_seconds = max(1, job['seconds'] - 5)
            m = DimenSRMIMORegressor(cfg, spec['features'], spec['targets'], spec['constants'])
            m.fit(X, Y, X_val=V, Y_val=Z, input_bounds=spec['bounds'], callback=event)
            m.save(folder / 'model.json')
            result['models'].append('model.json')
            result['diagnostics'].append(m.diagnostics_)
            pin_predictions(m)
            result['audits'].append(m.audit_domain(spec['bounds']))
        result['status'] = 'complete'
    except Exception as error:
        result['error'] = repr(error)
        (folder / 'traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
    result.update(fit_wall_seconds=time.monotonic() - started, fit_cpu_seconds=time.process_time() - cpu)
    write(folder / 'result.json', result)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('folder'); p.add_argument('job'); p.add_argument('config')
    a = p.parse_args()
    run(Path(a.folder), json.loads(Path(a.job).read_text()), json.loads(Path(a.config).read_text()))
