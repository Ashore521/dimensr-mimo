"""Forty searches, eighty frozen decisions, then one held-out scoring pass."""
import argparse
import importlib.metadata
from pathlib import Path
import platform
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from experiments.guarded_selection_20261026 import cases, worker, selector

legacy = cases.previous('study', cases=cases, worker=worker.base_worker)
legacy.HERE, legacy.ROOT = HERE, ROOT
read, write, sha = legacy.read, worker.write, legacy.sha
METHODS = ('original', 'no_penalty', 'guarded', 'independent')


def matrix():
    return [dict(case=c, seed=s, method=m, seconds=15 if m in ('no_penalty', 'guarded') else 60)
            for c in cases.CASES for s in cases.SEEDS for m in METHODS]


def budget():
    b = read(HERE / 'BUDGET.json')
    return b['start_unix'] + b['limit_seconds']


def verify_files(files, root=ROOT):
    for name, expected in files.items():
        p = (root / name).resolve()
        if not p.is_relative_to(root.resolve()) or sha(p) != expected:
            raise RuntimeError('Frozen file drift or invalid path: ' + name)


base_mechanism = legacy.mechanism


def mechanism(folder):
    job = read(folder / 'job.json')
    reused = job['method'] in ('no_penalty', 'guarded')
    source = folder.parent / f"{job['case']}_s{job['seed']}_original" if reused else folder
    evidence = base_mechanism(source)
    complete = (folder / 'result.json').exists() and read(folder / 'result.json')['status'] == 'complete'
    evidence['full_mechanism_pass'] = complete and all(
        r['policy_delta'] > 0 and r['neural_valid'] > 0 and r['gp_new_evaluations'] > 0
        and r['nsga_generations'] >= 2 and r['nsga_new_evaluations'] > 0 for r in evidence['outputs'])
    evidence.update(search_reused=reused, source_search=source.name)
    return evidence


legacy.budget, legacy.mechanism = budget, mechanism


def freeze():
    started = time.time()
    if (HERE / 'BUDGET.json').exists() or (HERE / 'FREEZE.json').exists():
        raise RuntimeError('Experiment already started; never reset the clock.')
    config = read(HERE.parent / 'first_round/FREEZE.json')['config']
    memory = legacy.psutil.virtual_memory()
    b = dict(start_unix=started, limit_seconds=3600, training_reserve_seconds=600,
             timezone='Asia/Shanghai', sequential_workers=1, threads=1,
             actual_search_systems=40, derived_decisions=40, final_systems=80,
             python=sys.version, platform=platform.platform(), available_ram_bytes=memory.available,
             packages={n: importlib.metadata.version(n) for n in ('numpy','scipy','torch','sympy','scikit-learn','psutil')})
    write(HERE / 'BUDGET.json', b)
    prior = subprocess.check_output(['git','ls-files','-z','--','experiments/first_round','experiments/second_round_20261026'],cwd=ROOT).decode().split('\0')
    files = [ROOT / n for n in prior if n] + list((ROOT / 'src').rglob('*.py')) + list(HERE.glob('*.py'))
    files += [HERE / n for n in ('PROTOCOL.md','BUDGET.json','PREFLIGHT.json')]
    files += [ROOT / n for n in ('pyproject.toml','docs/environment.txt','tests/test_guarded_selector.py')]
    write(HERE / 'FREEZE.json', dict(jobs=matrix(), config=config, policy=selector.POLICY,
          frozen_unix=time.time(), test_labels_accessed=False, budget=b,
          files={p.relative_to(ROOT).as_posix():sha(p) for p in files},
          base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()))


def checkpoint(folder):
    write(folder / 'CELL_FROZEN.json', dict(frozen_unix=time.time(),files={p.name:sha(p) for p in folder.iterdir() if p.is_file() and p.name!='CELL_FROZEN.json'}))


def train():
    frozen = read(HERE / 'FREEZE.json')
    verify_files(frozen['files'])
    if (HERE / 'ALL_MODELS_FROZEN.json').exists():
        verify_files(read(HERE / 'ALL_MODELS_FROZEN.json')['files'], HERE)
        return
    if not (HERE / 'TRAINING_STARTED.json').exists():
        write(HERE / 'TRAINING_STARTED.json',dict(start_unix=time.time()))
    for job in frozen['jobs']:
        verify_files(frozen['files'])
        folder = HERE / 'runs' / f"{job['case']}_s{job['seed']}_{job['method']}"
        if (folder / 'CELL_FROZEN.json').exists():
            verify_files(read(folder / 'CELL_FROZEN.json')['files'],folder)
            continue
        if (folder / 'job.json').exists():
            if not (folder / 'INTERRUPTION.json').exists():
                write(folder / 'INTERRUPTION.json',dict(status='interrupted_requires_review',automatic_retry=False))
            raise RuntimeError('Unsealed cell requires review: '+str(folder))
        if time.time()+job['seconds'] > budget()-600:
            folder.mkdir(parents=True,exist_ok=True)
            write(folder/'job.json',job); write(folder/'config.json',frozen['config'])
            write(folder/'resources.json',dict(status='not_started_budget',job=job))
        else:
            legacy.run_job(job,frozen['config'],'runs')
        checkpoint(folder)
    files = [p for p in (HERE/'runs').rglob('*') if p.is_file()]
    write(HERE/'ALL_MODELS_FROZEN.json',dict(frozen_unix=time.time(),test_labels_accessed=False,
          freeze_sha256=sha(HERE/'FREEZE.json'),files={p.relative_to(HERE).as_posix():sha(p) for p in files}))


def evaluate():
    if time.time() >= budget():
        raise RuntimeError('Global budget expired before test access.')
    frozen, models = read(HERE/'FREEZE.json'), read(HERE/'ALL_MODELS_FROZEN.json')
    verify_files(frozen['files']); verify_files(models['files'],HERE)
    if models['freeze_sha256'] != sha(HERE/'FREEZE.json'):
        raise RuntimeError('Model freeze mismatch.')
    for job in frozen['jobs']:
        if not (HERE/'runs'/f"{job['case']}_s{job['seed']}_{job['method']}"/'resources.json').exists():
            raise RuntimeError('Incomplete denominator before test release.')
    original = legacy.generate
    def checked(name, split, seed):
        if split == 'test' and time.time() >= budget():
            raise RuntimeError('Global budget expired.')
        return original(name,split,seed)
    legacy.generate = checked
    try:
        legacy.evaluate()
    finally:
        legacy.generate = original


def finish():
    verify_files(read(HERE/'FREEZE.json')['files'])
    end, steps = budget(), []
    for script, command in [('study.py','evaluate'),('verify_report.py','verify')]:
        start = time.time()
        if start >= end:
            steps.append(dict(command=command,status='not_started_budget')); break
        process = subprocess.Popen([sys.executable,'-B',str(HERE/script),command])
        while process.poll() is None and time.time()<end:
            time.sleep(.1)
        expired = process.poll() is None
        if expired:
            try:
                parent=legacy.psutil.Process(process.pid)
                for child in reversed(parent.children(recursive=True)):
                    try: child.kill()
                    except legacy.psutil.NoSuchProcess: pass
                parent.kill()
            except legacy.psutil.NoSuchProcess: pass
        process.wait()
        status='complete' if not expired and process.returncode==0 and time.time()<=end else 'failed_or_budget'
        steps.append(dict(command=command,status=status,returncode=process.returncode,seconds=time.time()-start))
        if status!='complete': break
    complete=len(steps)==2 and all(r['status']=='complete' for r in steps)
    write(HERE/'FINISH_SUPERVISION.json',dict(status='complete' if complete else 'incomplete',steps=steps,
          total_wall_seconds=time.time()-read(HERE/'BUDGET.json')['start_unix']))
    if not complete: raise RuntimeError('Finish incomplete; inspect supervision record.')


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=('freeze','train','evaluate','finish'))
    globals()[parser.parse_args().command]()
