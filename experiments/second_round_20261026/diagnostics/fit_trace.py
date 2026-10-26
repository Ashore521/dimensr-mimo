"""Predeclared diagnostic, without test access or production changes.

For original shared_parameter seeds 0/1, fit the known shared structure using
the original 1-start/25-evaluation configuration and a diagnostic 4-start/100
configuration. Then replay the original search with observational wrappers.
Each child has a 60-second process-tree cap; all diagnostics have a 240-second
cap. Existing outputs are never overwritten. Direct-pattern detection accepts
commuted multiplication but makes no algebraic-equivalence claim.
"""
import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FIRST = ROOT / 'experiments' / 'first_round'
sys.path.insert(0, str(FIRST))


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sid(system):
    return hashlib.sha256(json.dumps(system.to_dict(), sort_keys=True).encode()).hexdigest()


def signature(system):
    parameters = {}
    def visit(i):
        node = system.nodes[i]
        if node.parameter is not None:
            if node.parameter not in parameters:
                parameters[node.parameter] = len(parameters)
            return 'p' + str(parameters[node.parameter])
        children = [visit(j) for j in node.children]
        if node.token in ('mul', 'add'):
            children.sort()
        return node.token + ('(' + ','.join(children) + ')' if children else '')
    return tuple(visit(root) for root in system.roots)


def child(seed, folder):
    import numpy as np
    import torch
    from cases import generate, design
    from dimensr_mimo import DimenSRMIMORegressor, SearchConfig
    from dimensr_mimo import regressor as reg, fitting
    from dimensr_mimo.evaluation import Evaluator
    from dimensr_mimo.generation import SystemGrammar
    from dimensr_mimo.graph import System, Vertex, in_bounds
    from dimensr_mimo.pareto import violations

    torch.set_num_threads(1)
    original_path = FIRST / 'runs' / f'shared_parameter_s{seed}_full' / 'model.json'
    original = read(original_path)
    cfg = SearchConfig(**original['config'])
    X, Y = generate('shared_parameter', 'train', seed)
    V, Z = generate('shared_parameter', 'validation', seed)
    spec = design('shared_parameter')
    truth = System((Vertex('c0', parameter=0), Vertex('x0'), Vertex('mul', (0, 1)),
                    Vertex('sin', (2,)), Vertex('x1'), Vertex('mul', (0, 4)),
                    Vertex('cos', (5,))), (3, 6))
    expected = signature(truth)
    fixed = []
    for label, settings in [('original', cfg), ('diagnostic_4x100', replace(cfg, constant_restarts=4, constant_max_nfev=100))]:
        evaluator = Evaluator(X, Y, settings, input_bounds=spec['bounds'])
        started = time.monotonic()
        candidate = evaluator(truth, np.ones(1))
        fixed.append(dict(mode=label, settings=asdict(settings), wall_seconds=time.monotonic()-started,
                          candidate=candidate.to_dict() if candidate is not None else None,
                          stats=dict(evaluator.stats), recovered=candidate is not None and
                          bool(np.allclose(candidate.constants, [1.7], atol=1e-7, rtol=0))))
    write(folder / 'fixed_structure.json', fixed)
    state = dict(evaluator=None, pending=None, active_evaluation=None, proposal_count=0,
                 evaluations=[], proposals=[], optimizers=[], events=[])
    trace = (folder / 'candidates.jsonl').open('x', encoding='utf-8')

    def emit(value):
        trace.write(json.dumps(value, allow_nan=False) + '\n')
        trace.flush()

    def context():
        frame = inspect.currentframe().f_back
        while frame:
            if frame.f_code.co_name == 'evolve':
                return dict(phase='nsga2' if frame.f_locals['multiobjective'] else 'gp')
            if frame.f_code.co_name == 'fit' and frame.f_code.co_filename == reg.__file__:
                return dict(phase='fit', source_line=frame.f_lineno)
            frame = frame.f_back
        return dict(phase='unknown')

    def proposal(system, warm, source):
        state['proposal_count'] += 1
        ev = state['evaluator']
        entry = dict(kind='proposal', proposal=state['proposal_count'], system_id=sid(system),
                     signature=signature(system), system=system.to_dict(), warm=np.asarray(warm).tolist(),
                     direct_target=signature(system) == expected, **source)
        entry['precheck'] = ('cached' if ev is not None and system in ev.cache else
                             'structure_bounds' if not in_bounds(system, cfg) else 'eligible')
        state['pending'] = entry
        state['proposals'].append(entry)
        emit(entry)

    original_init, original_call = Evaluator.__init__, Evaluator.__call__
    original_offspring, original_generate = reg.offspring, SystemGrammar.generate
    original_optimizer = fitting.least_squares

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        state['evaluator'] = self

    def offspring(parent, donor, grammar, rng, operator=None):
        result = original_offspring(parent, donor, grammar, rng, operator)
        proposal(result[0], result[1], dict(**context(), operator=result[2], parent=sid(parent.system), donor=sid(donor.system)))
        return result

    def generate(self, rng, policy=None):
        result = original_generate(self, rng, policy)
        proposal(result[0], result[1], dict(phase='neural' if policy is not None else 'random'))
        return result

    def optimizer(fun, initial, **kwargs):
        try:
            result = original_optimizer(fun, initial, **kwargs)
        except Exception as error:
            emit(dict(kind='optimizer_exception', evaluation=state['active_evaluation'],
                      initial=np.asarray(initial).tolist(), error=repr(error)))
            raise
        entry = dict(kind='optimizer', evaluation=state['active_evaluation'], initial=np.asarray(initial).tolist(),
                     final=result.x.tolist(), status=int(result.status), success=bool(result.success),
                     message=str(result.message), nfev=int(result.nfev), cost=float(result.cost))
        state['optimizers'].append(entry)
        emit(entry)
        return result

    def evaluate(self, system, warm, refine=False):
        before = self.stats.copy()
        pending = state['pending']
        if pending is None or pending['system_id'] != sid(system) or refine:
            proposal(system, warm, dict(**context(), operation='refine' if refine else 'initialization_or_reference'))
            pending = state['pending']
        state['active_evaluation'] = len(state['evaluations'])
        cached = system in self.cache and not refine
        candidate = original_call(self, system, warm, refine)
        delta = {key: value - before[key] for key, value in self.stats.items() if value != before[key]}
        reason = ('cache_hit' if cached else 'valid' if candidate is not None else
                  'deadline' if self.expired() else 'structure_bounds' if not in_bounds(system, cfg) else
                  'evaluation_budget' if self.calls >= cfg.max_evaluations else 'no_accepted_fit')
        entry = dict(kind='evaluation', evaluation=state['active_evaluation'], proposal=pending['proposal'],
                     system_id=sid(system), direct_target=signature(system) == expected,
                     reason=reason, refine=refine, candidate=candidate.to_dict() if candidate is not None else None,
                     stats_delta=delta)
        state['evaluations'].append(entry)
        emit(entry)
        state['pending'] = None
        state['active_evaluation'] = None
        return candidate

    def event(entry):
        state['events'].append(entry)
        emit(dict(kind='phase', **entry))

    Evaluator.__init__, Evaluator.__call__ = init, evaluate
    reg.offspring, SystemGrammar.generate, fitting.least_squares = offspring, generate, optimizer
    started = time.monotonic()
    try:
        model = DimenSRMIMORegressor(cfg, spec['features'], spec['targets'], spec['constants'])
        model.fit(X, Y, X_val=V, Y_val=Z, input_bounds=spec['bounds'], callback=event)
        model.save(folder / 'model.json')
    finally:
        Evaluator.__init__, Evaluator.__call__ = original_init, original_call
        reg.offspring, SystemGrammar.generate, fitting.least_squares = original_offspring, original_generate, original_optimizer
        trace.close()
    ev = state['evaluator']
    pool = {sid(System.from_dict(c['system'])) for c in model.candidates_}
    selected = sid(model.best_.system)
    cache = []
    for system, c in ev.cache.items():
        if c is None:
            cache.append(dict(system_id=sid(system), outcome='fit_failed'))
            continue
        errors = np.sqrt(np.mean(((system.evaluate(V, c.constants)-Z)/ev.scales)**2, axis=0))
        validation_valid = bool(np.isfinite(errors).all())
        cache.append(dict(system_id=sid(system), outcome='selected' if sid(system) == selected else
                          'final_pool' if sid(system) in pool else 'omitted_from_final_pool',
                          train_mean=c.error, validation_mean=float(errors.mean()) if validation_valid else None,
                          violation=float(violations(errors, model.reference_validation_, cfg)) if validation_valid else None,
                          direct_target=signature(system) == expected, sharing=system.sharing))
    viable = [c for c in cache if c.get('validation_mean') is not None]
    best = min(viable, key=lambda c: (c['violation'], c['validation_mean']))
    summary = dict(seed=seed, original_model_sha256=sha(original_path), fixed_structure=fixed,
                   search_wall_seconds=time.monotonic()-started, search_fingerprint=model.search_fingerprint_,
                   original_search_fingerprint=original['search_fingerprint'],
                   fingerprints_equal=model.search_fingerprint_ == original['search_fingerprint'],
                   selected_equal=model.best_.to_dict() == original['best'],
                   deadline_reached=model.diagnostics_['deadline_reached'],
                   proposal_count=len(state['proposals']), evaluation_calls=len(state['evaluations']),
                   unique_evaluated=len(ev.cache), optimizer_calls=len(state['optimizers']),
                   optimizer_status_counts=dict(Counter(o['status'] for o in state['optimizers'])),
                   direct_target_proposals=sum(p['direct_target'] for p in state['proposals']),
                   direct_target_evaluations=sum(p['direct_target'] for p in state['evaluations']),
                   direct_target_valid=sum(c.get('direct_target', False) for c in cache),
                   outcome_counts=dict(Counter(c['outcome'] for c in cache)),
                   selected=next(c for c in cache if c['system_id'] == selected),
                   best_cache_validation=best, cache_outcomes=cache,
                   note='Validation audit of complete cache occurs only after model freeze; no test data. '
                        'Exact direct-pattern match only; intermediate population淘汰原因 not traced. '
                        'Rejection reasons use optimizer and aggregate raw-point/interval rejection counters.')
    assert summary['evaluation_calls'] == ev.stats['proposals']
    assert all(len(o['initial']) == len(o['final']) for o in state['optimizers'])
    write(folder / 'summary.json', summary)


def main():
    import psutil
    started = time.time()
    target = HERE / 'FIT_TRACE.json'
    runroot = HERE / 'trace_runs'
    if target.exists() or runroot.exists():
        raise RuntimeError('Diagnostic outputs already exist; do not overwrite or silently rerun.')
    runroot.mkdir(parents=True)
    sources = {str(p.relative_to(ROOT)): sha(p) for p in (ROOT/'src').rglob('*.py')}
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in [FIRST/'cases.py', FIRST/'worker.py'] +
              [FIRST/'runs'/f'shared_parameter_s{s}_full'/'model.json' for s in (0, 1)]}
    write(runroot/'PROTOCOL.json', dict(protocol=__doc__, start_unix=started, source_hashes=sources,
                                      input_hashes=inputs, script_sha256=sha(Path(__file__)),
                                      test_data_access=False, seeds=[0,1], system_hard_cap_seconds=60,
                                      total_hard_cap_seconds=240))
    rows = []
    for seed in (0, 1):
        folder = runroot/f'seed{seed}'
        folder.mkdir()
        t, peak, reason = time.monotonic(), 0, None
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUTF8='1')
        with (folder/'stdout.log').open('x', encoding='utf-8') as log:
            process = subprocess.Popen([sys.executable, '-X', 'utf8', __file__, '--worker', str(seed), str(folder)],
                                       stdout=log, stderr=subprocess.STDOUT, env=env)
            while process.poll() is None:
                try:
                    tree = [psutil.Process(process.pid)] + psutil.Process(process.pid).children(recursive=True)
                    peak = max(peak, sum(p.memory_info().rss for p in tree if p.is_running()))
                    if time.monotonic()-t > 60 or time.time()-started > 240:
                        reason = 'hard_timeout'
                        for p in reversed(tree):
                            try:
                                p.kill()
                            except psutil.NoSuchProcess:
                                pass
                        process.wait()
                        break
                except psutil.NoSuchProcess:
                    pass
                time.sleep(.2)
        row = dict(seed=seed, returncode=process.returncode, wall_seconds=time.monotonic()-t,
                   peak_tree_rss_bytes=peak, status=reason or ('complete' if process.returncode == 0 else 'failed'))
        if (folder/'summary.json').exists():
            row['summary'] = read(folder/'summary.json')
        write(folder/'resources.json', row | {'summary': None})
        rows.append(row)
    unchanged = all(sha(ROOT/p) == h for p,h in sources.items()) and all(sha(ROOT/p) == h for p,h in inputs.items())
    write(target, dict(rows=rows, wall_seconds=time.time()-started, sources_inputs_unchanged=unchanged,
                       protocol_sha256=sha(runroot/'PROTOCOL.json'), test_data_access=False,
                       status='complete' if unchanged and all(r['status']=='complete' for r in rows) else 'incomplete'))
    print(json.dumps(dict(status=read(target)['status'], wall_seconds=time.time()-started,
                          rows=[{k:v for k,v in r.items() if k!='summary'} for r in rows])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=int)
    parser.add_argument('folder', nargs='?')
    args = parser.parse_args()
    if args.worker is None:
        main()
    else:
        child(args.worker, Path(args.folder))
