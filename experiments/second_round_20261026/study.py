"""Sequential 60-system replication with pinned inputs and a single test release."""
import argparse
import importlib.metadata
from pathlib import Path
import platform
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.second_round_20261026 import cases, worker
from dimensr_mimo import SearchConfig

legacy = cases.previous('study', cases=cases, worker=worker.previous_worker)
legacy.HERE, legacy.ROOT = HERE, ROOT
read, sha, write = legacy.read, legacy.sha, worker.write
METHODS = ('original', 'proposed', 'independent')


def matrix():
    return [dict(case=c, seed=s, method=m, seconds=60) for c in cases.CASES for s in cases.SEEDS for m in METHODS]


def configs(base, change):
    overrides = change['config_overrides']
    if not overrides or set(overrides) - set(SearchConfig.__dataclass_fields__):
        raise ValueError('NEW_CHANGE must specify known configuration fields.')
    # Budgets and seeds are fixed by the protocol, never by the candidate change.
    if set(overrides) & {'random_state', 'wall_seconds', 'max_evaluations'}:
        raise ValueError('The proposed change cannot alter resource caps or random seeds.')
    proposal = dict(base, **overrides)
    SearchConfig(**base).validate(); SearchConfig(**proposal).validate()
    return dict(original=dict(base), proposed=proposal, independent=dict(base))


def verify_files(files, root=ROOT):
    for name, expected in files.items():
        if sha(root / name) != expected:
            raise RuntimeError('Frozen file drift: ' + name)


def budget():
    b = read(HERE / 'BUDGET.json')
    return b['start_unix'] + b['limit_seconds']


def mechanism(folder):
    evidence = original_mechanism(folder)
    result = read(folder / 'result.json') if (folder / 'result.json').exists() else {}
    evidence['full_mechanism_pass'] = result.get('status') == 'complete' and all(
        e['policy_delta'] > 0 and e['neural_valid'] > 0 and e['gp_new_evaluations'] > 0
        and e['nsga_generations'] >= 2 and e['nsga_new_evaluations'] > 0 for e in evidence['outputs'])
    return evidence


original_mechanism = legacy.mechanism
legacy.budget, legacy.mechanism = budget, mechanism


def freeze():
    started = time.time()
    if (HERE / 'FREEZE.json').exists() or (HERE / 'BUDGET.json').exists():
        raise RuntimeError('Already started; do not reset the experiment budget.')
    base = read(HERE.parent / 'first_round' / 'FREEZE.json')['config']
    change = read(HERE / 'NEW_CHANGE.json')
    settings = configs(base, change)
    old_files = subprocess.check_output(['git', 'ls-files', '-z', '--', 'experiments/first_round'], cwd=ROOT).decode('utf-8').split('\0')
    tracked = [ROOT / path for path in old_files if path]
    files = tracked + list((ROOT / 'src').rglob('*.py')) + list(HERE.rglob('*.py'))
    files += [p for p in (HERE / 'diagnostics').rglob('*') if p.suffix in ('.json', '.jsonl')]
    files += [HERE / 'PROTOCOL.md', HERE / 'NEW_CHANGE.json', ROOT / 'pyproject.toml', ROOT / 'docs/environment.txt',
              ROOT / 'tests/test_second_round.py']
    memory = legacy.psutil.virtual_memory()
    b = dict(start_unix=started, limit_seconds=5400, training_limit_seconds=3600,
             scoring_reserve_seconds=120, timezone='Asia/Shanghai', sequential_workers=1, threads=1,
             cpu_count=legacy.psutil.cpu_count(), total_ram_bytes=memory.total, available_ram_bytes=memory.available,
             python=sys.version, platform=platform.platform(),
             packages={name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'torch', 'sympy', 'scikit-learn', 'psutil')})
    write(HERE / 'BUDGET.json', b)
    fingerprints = {p.relative_to(ROOT).as_posix(): sha(p) for p in files}
    fingerprints[(HERE / 'BUDGET.json').relative_to(ROOT).as_posix()] = sha(HERE / 'BUDGET.json')
    write(HERE / 'FREEZE.json', dict(jobs=matrix(), configs=settings, files=fingerprints,
          frozen_unix=time.time(), test_labels_accessed=False, budget=b,
          base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip()))


def freeze_cell(folder):
    write(folder / 'CELL_FROZEN.json', dict(frozen_unix=time.time(),
          files={p.relative_to(folder).as_posix(): sha(p) for p in folder.rglob('*')
                 if p.is_file() and p.name != 'CELL_FROZEN.json'}))


def verify_cell(folder, job):
    checkpoint = folder / 'CELL_FROZEN.json'
    if not checkpoint.exists():
        incident = folder / 'INTERRUPTION.json'
        if not incident.exists():
            write(incident, dict(status='completed_without_cell_freeze_requires_review', job=job,
                                observed_unix=time.time(), automatic_retry=False))
        raise RuntimeError('Completed run lacks CELL_FROZEN.json; incident review required: ' + str(folder))
    verify_files(read(checkpoint)['files'], folder)


def train():
    frozen = read(HERE / 'FREEZE.json')
    verify_files(frozen['files'])
    if (HERE / 'ALL_MODELS_FROZEN.json').exists():
        verify_files(read(HERE / 'ALL_MODELS_FROZEN.json')['files'], HERE)
        return
    if not (HERE / 'TRAINING_STARTED.json').exists():
        write(HERE / 'TRAINING_STARTED.json', dict(start_unix=time.time()))
    training_deadline = read(HERE / 'TRAINING_STARTED.json')['start_unix'] + 3600
    for job in frozen['jobs']:
        verify_files(frozen['files'])
        folder = HERE / 'runs' / f"{job['case']}_s{job['seed']}_{job['method']}"
        if (folder / 'resources.json').exists():
            verify_cell(folder, job)
            continue
        if (folder / 'job.json').exists() and not (folder / 'resources.json').exists():
            incident = folder / 'INTERRUPTION.json'
            if not incident.exists():
                write(incident, dict(status='interrupted_requires_review', job=job,
                                    observed_unix=time.time(), automatic_retry=False))
            raise RuntimeError('Interrupted run requires incident review: ' + str(folder))
        if not (folder / 'resources.json').exists() and time.time() + job['seconds'] > training_deadline:
            folder.mkdir(parents=True, exist_ok=True)
            write(folder / 'job.json', job)
            write(folder / 'config.json', frozen['configs'][job['method']])
            write(folder / 'resources.json', dict(status='not_started_training_budget', job=job))
        else:
            legacy.run_job(job, frozen['configs'][job['method']], 'runs')
        freeze_cell(folder)
    files = [p for p in (HERE / 'runs').rglob('*') if p.is_file()]
    write(HERE / 'ALL_MODELS_FROZEN.json', dict(freeze_sha256=sha(HERE / 'FREEZE.json'),
          frozen_unix=time.time(), test_labels_accessed=False,
          files={p.relative_to(HERE).as_posix(): sha(p) for p in files}))


def evaluate():
    deadline = budget()
    if time.time() >= deadline:
        raise RuntimeError('Global budget expired before test scoring.')
    frozen = read(HERE / 'FREEZE.json')
    verify_files(frozen['files'])
    model_freeze = read(HERE / 'ALL_MODELS_FROZEN.json')
    if model_freeze['freeze_sha256'] != sha(HERE / 'FREEZE.json'):
        raise RuntimeError('Model freeze refers to a different experiment.')
    verify_files(model_freeze['files'], HERE)
    # The reused evaluator generates test data only after the complete matrix is frozen.
    for job in frozen['jobs']:
        folder = HERE / 'runs' / f"{job['case']}_s{job['seed']}_{job['method']}"
        if not (folder / 'resources.json').exists():
            raise RuntimeError('Missing outcome in the full denominator.')
    original_generate = legacy.generate
    def checked_generate(name, split, seed):
        if split == 'test' and time.time() >= deadline:
            raise RuntimeError('Global budget expired before generating test data.')
        return original_generate(name, split, seed)
    legacy.generate = checked_generate
    try:
        legacy.evaluate()
    finally:
        legacy.generate = original_generate
    finished = time.time()
    if finished > deadline:
        write(HERE / 'SCORING_BUDGET_OVERRUN.json', dict(
            status='scoring_finished_after_global_budget', deadline_unix=deadline,
            finished_unix=finished, overrun_seconds=finished - deadline,
            scores_sha256=sha(HERE / 'SCORES.json'), results_retained=True))
        raise RuntimeError('Scoring finished after the global budget; results retained with an overrun record.')


def finish():
    """Run scoring and verification under the remaining shared hard deadline."""
    verify_files(read(HERE / 'FREEZE.json')['files'])
    timing = read(HERE / 'BUDGET.json')
    deadline, steps = timing['start_unix'] + timing['limit_seconds'], []
    for script, command in [('study.py', 'evaluate'), ('verify_report.py', 'verify')]:
        started = time.time()
        if started >= deadline:
            steps.append(dict(command=command, status='not_started_global_budget', returncode=None))
            break
        process = subprocess.Popen([sys.executable, '-X', 'utf8', str(HERE / script), command])
        while process.poll() is None and time.time() < deadline:
            time.sleep(.1)
        expired = process.poll() is None
        if expired:
            try:
                parent = legacy.psutil.Process(process.pid)
                for child in reversed(parent.children(recursive=True)):
                    try:
                        child.kill()
                    except legacy.psutil.NoSuchProcess:
                        pass
                parent.kill()
            except legacy.psutil.NoSuchProcess:
                pass
        process.wait()
        finished = time.time()
        status = ('global_budget' if expired else 'completed_after_global_budget' if finished > deadline
                  else 'complete' if process.returncode == 0 else 'failed')
        steps.append(dict(command=command, status=status, returncode=process.returncode, wall_seconds=finished - started))
        if status != 'complete':
            break
    complete = len(steps) == 2 and all(step['status'] == 'complete' for step in steps)
    write(HERE / 'FINISH_SUPERVISION.json', dict(status='complete' if complete else 'incomplete', steps=steps,
          deadline_unix=deadline, total_wall_seconds=time.time() - timing['start_unix']))
    if not complete:
        raise RuntimeError('Scoring/verification incomplete; inspect FINISH_SUPERVISION.json for failure or budget expiry.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['freeze', 'train', 'evaluate', 'finish'])
    args = p.parse_args()
    globals()[args.command]()
