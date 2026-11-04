"""External process-tree deadline, including verification; retain any failure."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    if any((HERE/name).exists() for name in ('BUDGET.json', 'supervisor.log', 'EXECUTION.json')):
        raise RuntimeError('Existing execution requires checkpoint/incident inspection; do not restart.')
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUTF8='1')
    start = time.time()
    deadline, reason = start+60, None
    with (HERE/'supervisor.log').open('x', encoding='utf-8') as log:
        process = subprocess.Popen([sys.executable, str(HERE/'study.py'), 'run'], cwd=ROOT,
            env=env, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            if (HERE/'BUDGET.json').exists():
                try:
                    budget = json.loads((HERE/'BUDGET.json').read_text())
                    deadline = budget['start_unix']+budget['limit_seconds']
                except json.JSONDecodeError:
                    pass  # The initial budget write can overlap one observation.
            if time.time() >= deadline:
                reason = 'global_deadline'
                try:
                    children = psutil.Process(process.pid).children(recursive=True)
                    for child in reversed(children):
                        try:
                            child.kill()
                        except psutil.NoSuchProcess:
                            pass
                    process.kill()
                except psutil.NoSuchProcess:
                    pass
                break
            time.sleep(.2)
        code = process.wait()
    verified = (HERE/'VERIFICATION.json').exists() and json.loads((HERE/'VERIFICATION.json').read_text())['status'] == 'passed'
    report = dict(returncode=code, termination_reason=reason, wall_seconds=time.time()-start,
        poll_seconds=.2, verification_passed=verified)
    (HERE/'EXECUTION.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report))
    if reason or code or not verified:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
