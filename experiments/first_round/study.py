"""Persisted sequential supervisor: resource bounds, freezes, then uniform scoring."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import numpy as np
import psutil
from dimensr_mimo import DimenSRMIMORegressor, SearchConfig
from dimensr_mimo._single import DimenSRRegressor
from cases import CASES, design, generate
from worker import write

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
METHODS = ('full', 'independent', 'no_neural', 'no_gp', 'no_sharing')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def budget():
    p = HERE / 'BUDGET.json'
    if not p.exists():
        memory = psutil.virtual_memory()
        write(p, dict(start_unix=time.time(), limit_seconds=7200, timezone='Asia/Shanghai',
                      cpu_count=psutil.cpu_count(), physical_cpus=psutil.cpu_count(logical=False),
                      available_ram_bytes=memory.available, total_ram_bytes=memory.total,
                      python=sys.version, platform=platform.platform(), sequential_workers=1,
                      threads=1))
    d = read(p)
    return d['start_unix'] + d['limit_seconds']


def run_job(job, config, base):
    folder = HERE / base / f"{job['case']}_s{job['seed']}_{job['method']}"
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'resources.json').exists():
        return read(folder / 'resources.json')
    if (folder / 'job.json').exists():
        raise RuntimeError(f'Interrupted cell needs explicit incident review: {folder}')
    write(folder / 'job.json', job)
    write(folder / 'config.json', config)
    remaining = budget() - time.time()
    if remaining < job['seconds'] + 120:
        result = dict(status='not_started_global_budget', job=job)
        write(folder / 'resources.json', result)
        return result
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUTF8='1')
    started, peak, max_cpu, reason = time.monotonic(), 0, 0, None
    last_output_started, output_open = None, False
    cpu_by_pid = {}
    with (folder / 'stdout.log').open('w', encoding='utf-8') as log:
        process = subprocess.Popen([sys.executable, '-X', 'utf8', str(HERE / 'worker.py'), str(folder),
                                    str(folder / 'job.json'), str(folder / 'config.json')],
                                   stdout=log, stderr=subprocess.STDOUT, env=env)
        while process.poll() is None:
            try:
                parent = psutil.Process(process.pid)
                children = parent.children(recursive=True)
                rss = 0
                for child in [parent] + children:
                    try:
                        rss += child.memory_info().rss
                        ct = child.cpu_times()
                        cpu_by_pid[child.pid] = ct.user + ct.system
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                peak = max(peak, rss)
                max_cpu = max(max_cpu, sum(cpu_by_pid.values()))
                if job['method'] == 'independent' and (folder / 'events.jsonl').exists():
                    for line in (folder / 'events.jsonl').read_text().splitlines():
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if event['phase'] == 'output_start':
                            last_output_started = event['monotonic']; output_open = True
                        elif event['phase'] == 'output_complete':
                            output_open = False
                    if output_open and last_output_started is not None and time.monotonic() - last_output_started > job['seconds'] / 2:
                        reason = 'per_output_timeout'
                if time.monotonic() - started >= job['seconds']:
                    reason = 'system_timeout'
                if time.time() >= budget() - 120:
                    reason = 'global_budget'
                if reason:
                    for child in reversed(children):
                        try:
                            child.kill()
                        except psutil.NoSuchProcess:
                            pass
                    process.kill(); break
            except psutil.NoSuchProcess:
                break
            time.sleep(.2)
        process.wait()
    result = read(folder / 'result.json') if (folder / 'result.json').exists() else {}
    resource = dict(status=reason or result.get('status', 'worker_failed'), job=job,
                    wall_seconds=time.monotonic() - started, process_tree_peak_rss_bytes=peak,
                    process_tree_cpu_seconds=max_cpu, returncode=process.returncode,
                    measurement='process-tree sample every 0.2s; CPU at final poll may undercount')
    write(folder / 'resources.json', resource)
    print(folder.name, resource['status'], round(resource['wall_seconds'], 2), flush=True)
    return resource


def mechanism(folder):
    result = read(folder / 'result.json') if (folder / 'result.json').exists() else {}
    events = [json.loads(line) for line in (folder / 'events.jsonl').read_text().splitlines()] if (folder / 'events.jsonl').exists() else []
    method = read(folder / 'job.json')['method']
    outputs = [0, 1] if method == 'independent' else [None]
    evidence = []
    for output in outputs:
        rows = [e for e in events if e['output'] == output]
        nsga = [e for e in rows if e['phase'] == 'nsga2']
        evidence.append(dict(output=output, policy_delta=sum(e.get('parameter_delta', 0) for e in rows),
            neural_valid=sum(e.get('neural_valid', 0) for e in rows), nsga_generations=len(nsga),
            nsga_new_evaluations=sum(e.get('new_evaluations', 0) for e in nsga),
            gp_new_evaluations=sum(e.get('new_evaluations', 0) for e in rows if e['phase'] == 'gp')))
    required = method in ('full', 'independent', 'no_sharing')
    passed = result.get('status') == 'complete' and (not required or all(
        e['policy_delta'] > 0 and e['neural_valid'] > 0 and e['nsga_generations'] >= 2 and e['nsga_new_evaluations'] > 0 for e in evidence))
    return dict(full_mechanism_pass=passed, outputs=evidence)


def calibrate():
    budget()
    cfg = SearchConfig(population_size=24, epochs=4, gp_generations=2, selection_generations=4,
        max_evaluations=1200, max_nodes=15, max_depth=6, min_candidates=30,
        operators=('add', 'sub', 'mul', 'div', 'square', 'sin', 'cos', 'sqrt', 'log'),
        constant_restarts=1, constant_max_nfev=25, domain_restarts=2,
        polish_candidates=3, polish_max_nfev=50)
    if (HERE / 'CALIBRATION.json').exists():
        raise RuntimeError('Calibration already frozen.')
    rows = []
    for case in ('shared_parameter', 'units'):
        for method in ('full', 'independent'):
            job = dict(case=case, seed=17, method=method, seconds=65)
            resource = run_job(job, asdict(cfg), 'calibration')
            folder = HERE / 'calibration' / f'{case}_s17_{method}'
            rows.append(dict(resource=resource, mechanism=mechanism(folder)))
    write(HERE / 'CALIBRATION.json', dict(config=asdict(cfg), runs=rows,
         note='No test labels generated; choose final resource cap using stage completion only.'))


def freeze(seconds, seeds):
    if (HERE / 'FREEZE.json').exists():
        raise RuntimeError('Already frozen.')
    calibration = read(HERE / 'CALIBRATION.json')
    if not all(r['mechanism']['full_mechanism_pass'] for r in calibration['runs']):
        raise RuntimeError('Calibration did not establish complete mechanisms; record amendment before proceeding.')
    jobs = [dict(case=c, seed=s, method=m, seconds=seconds) for c in CASES for s in seeds for m in METHODS]
    files = list((ROOT / 'src').rglob('*.py')) + list(HERE.glob('*.py')) + [HERE / 'PROTOCOL.md', ROOT / 'pyproject.toml', ROOT / 'docs/environment.txt']
    write(HERE / 'FREEZE.json', dict(jobs=jobs, config=calibration['config'],
        files={p.relative_to(ROOT).as_posix(): sha(p) for p in files}, frozen_unix=time.time(),
        test_labels_accessed=False, budget=read(HERE / 'BUDGET.json')))


def train():
    f = read(HERE / 'FREEZE.json')
    for name, h in f['files'].items():
        if sha(ROOT / name) != h:
            raise RuntimeError('Frozen file drift: ' + name)
    for job in f['jobs']:
        run_job(job, f['config'], 'runs')
    files = list((HERE / 'runs').rglob('model*.json')) + list((HERE / 'runs').rglob('result.json')) + list((HERE / 'runs').rglob('resources.json'))
    if not (HERE / 'ALL_MODELS_FROZEN.json').exists():
        write(HERE / 'ALL_MODELS_FROZEN.json', dict(freeze_sha256=sha(HERE / 'FREEZE.json'), frozen_unix=time.time(),
            files={p.relative_to(HERE).as_posix(): sha(p) for p in files}, test_labels_accessed=False))


def evaluate():
    if (HERE / 'SCORES.json').exists():
        raise RuntimeError('Scores already exist; do not overwrite frozen evidence.')
    f = read(HERE / 'ALL_MODELS_FROZEN.json')
    for name, h in f['files'].items():
        if sha(HERE / name) != h:
            raise RuntimeError('Model drift: ' + name)
    rows = []
    for job in read(HERE / 'FREEZE.json')['jobs']:
        folder = HERE / 'runs' / f"{job['case']}_s{job['seed']}_{job['method']}"
        resource = read(folder / 'resources.json')
        row = dict(job=job, resources=resource, mechanism=mechanism(folder))
        if resource['status'] != 'complete':
            rows.append(row); continue
        X, Y = generate(job['case'], 'test', job['seed'])
        T, U = generate(job['case'], 'train', job['seed'])
        V, _ = generate(job['case'], 'validation', job['seed'])
        original = read(folder / 'result.json')
        spec = design(job['case'])
        if job['method'] == 'independent':
            models = [DimenSRRegressor.load(folder / f'model-{j}.json') for j in range(2)]
            pred = np.column_stack([m.predict(X) for m in models])
            reload_pred = np.column_stack([DimenSRRegressor.load(folder / f'model-{j}.json').predict(X) for j in range(2)])
            complexity = sum(m.best_.complexity for m in models)
            nodes = sum(m.best_.tree.size for m in models)
            parameters = sum(len(m.best_.constants) for m in models)
            audit = [m.audit_domain(input_bounds=spec['bounds']) for m in models]
            formulas = [str(m.expression_) for m in models]
        else:
            m = DimenSRMIMORegressor.load(folder / 'model.json')
            pred = m.predict(X); reload_pred = DimenSRMIMORegressor.load(folder / 'model.json').predict(X)
            complexity, nodes, parameters = m.best_.complexity, m.best_.system.size, m.best_.system.n_parameters
            audit = [m.audit_domain(spec['bounds'])]
            formulas = m.formulas_
            row['sharing'] = m.best_.system.sharing
            row['validation_feasible'] = m.diagnostics_['validation_feasible']
            models = [m]
        original_hashes = [{name: hashlib.sha256(m.predict(A).astype('<f8').tobytes()).hexdigest()
                            for name, A in [('train', T), ('validation', V)]} for m in models]
        row['saved_original_predictions_equal'] = original_hashes == original['prediction_hashes']
        row['saved_original_audits_equal'] = audit == original['audits']
        scales = np.std(U, axis=0)
        rmse = np.sqrt(np.mean((pred - Y) ** 2, axis=0)); nrmse = rmse / scales
        r2 = 1 - np.sum((pred - Y) ** 2, axis=0) / np.sum((Y - Y.mean(axis=0)) ** 2, axis=0)
        row.update(rmse=rmse.tolist(), nrmse=nrmse.tolist(), r2=r2.tolist(), mean_nrmse=float(nrmse.mean()),
                   max_nrmse=float(nrmse.max()), complexity=complexity, nodes=nodes, parameters=parameters,
                   audit=audit, formulas=formulas, reload_equal=bool(np.array_equal(pred, reload_pred)))
        rows.append(row)
    write(HERE / 'SCORES.json', dict(rows=rows, all_models_freeze_sha256=sha(HERE / 'ALL_MODELS_FROZEN.json'),
                                    scored_unix=time.time(), total_wall_seconds=time.time() - read(HERE / 'BUDGET.json')['start_unix']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['calibrate', 'freeze', 'train', 'evaluate'])
    parser.add_argument('--seconds', type=int, default=60)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1])
    args = parser.parse_args()
    if args.command == 'freeze':
        freeze(args.seconds, args.seeds)
    else:
        globals()[args.command]()
