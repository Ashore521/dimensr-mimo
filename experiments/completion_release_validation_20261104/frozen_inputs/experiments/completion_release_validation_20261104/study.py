"""Fixed new output pairs under the locked candidate and original batch limits."""
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from experiments.compositional_completion_20261031 import harness
from experiments.completion_release_validation_20261104 import gate,worker
from experiments.completion_development_verified_20261031 import archive_io

SHARED=harness.SHARED
PRIOR=ROOT/'experiments/completion_development_verified_20261031'
STAGE='release_validation'
METHODS=('base','completion')
LIMIT,EXPECTED=2880,32
cases=harness.isolated(HERE/'cases.py','_release_validation_cases')
cases.HERE=HERE
name,mechanism=harness.audited.name,harness.audited.mechanism


def budget():return harness.read(HERE/'BUDGET.json')['start_unix']+LIMIT
def check_candidate_lock():return gate.check_candidate_lock(sys.modules[__name__])
def check_stage_gate():return gate.check(sys.modules[__name__])
def matrix():return harness.matrix(sys.modules[__name__])
def configuration():return harness.configuration(sys.modules[__name__])
def verify_files(files,root=None):return archive_io.verify_files(files,ROOT if root is None else root)
def freeze_inputs():return gate.freeze_inputs(sys.modules[__name__])


def task_binding():
    return dict(local_candidate_lock_sha256=harness.sha(HERE/'CANDIDATE_LOCK.json'),
        parent_candidate_lock_sha256=harness.sha(SHARED/'CANDIDATE_LOCK.json'),
        cases_sha256=harness.sha(HERE/'cases.py'),plan_sha256=harness.sha(HERE/'PLAN.md'),
        design_checks_sha256=harness.sha(HERE/'DESIGN_CHECKS.json'),
        jobs=matrix(),config=configuration(),specs={c:cases.design(c) for c in cases.CASES})


def preflight():
    check_candidate_lock();check_stage_gate()
    gate.validate_task_sources(sys.modules[__name__])
    harness.require(not (HERE/'PREFLIGHT.json').exists(),'Keep existing preflight.')
    jobs=matrix()
    harness.require(len(jobs)==len({name(j) for j in jobs})==EXPECTED,'Wrong release matrix.')
    for job in jobs:harness.effective_config(sys.modules[__name__],job,configuration()).validate()
    rows=[]
    for case in cases.CASES:
        for seed in cases.seeds(case):
            X,Y=cases.training_data(case,'train',seed);V,Z=cases.training_data(case,'validation',seed)
            harness.require(X.shape==Y.shape==(96,2) and V.shape==Z.shape==(48,2),'Wrong release data shape.')
            harness.require(all(np.isfinite(a).all() for a in (X,Y,V,Z)) and np.all(Y.std(axis=0)>0),'Invalid release training data.')
            rows.append(dict(case=case,seed=seed,data_sha256=harness.arrays_hash(X,Y,V,Z)))
    harness.write(HERE/'PREFLIGHT.json',dict(status='passed',rows=rows,test_labels_accessed=False,**task_binding()))


def freeze():
    gate.validate_task_sources(sys.modules[__name__])
    receipt=harness.read(HERE/'PREFLIGHT.json')
    harness.require(receipt['test_labels_accessed'] is False and all(receipt[k]==v for k,v in task_binding().items()),'Release preflight binding drift.')
    check_candidate_lock();check_stage_gate()
    harness.require(not any((HERE/n).exists() for n in ('FREEZE.json','BUDGET.json')),'Already frozen.')
    harness.require(receipt['status']=='passed','Preflight failed.')
    verify_files(harness.read(PRIOR/'FREEZE.json')['files'],PRIOR/'frozen_inputs')
    verify_files(harness.read(PRIOR/'ALL_MODELS_FROZEN.json')['files'],PRIOR)
    files=freeze_inputs()
    for path in files:harness.require(archive_io.disk_path(path).is_file(),'Missing freeze input: '+str(path))
    for path in (ROOT/'src').rglob('*.py'):
        dest=HERE/'source'/path.relative_to(ROOT)
        archive_io.copy_original(path,dest,archive_io.sha(path));files.append(dest)
    b=dict(start_unix=harness.time.time(),limit_seconds=LIMIT,sequential_workers=1,threads=1,system_cap_seconds=60,
        fit_seconds=52,verification_reserve_seconds=600,versions={n:harness.importlib.metadata.version(n)
            for n in ('numpy','scipy','torch','sympy','scikit-learn','psutil')})
    harness.write(HERE/'BUDGET.json',b);files.append(HERE/'BUDGET.json')
    harness.write(HERE/'FREEZE.json',dict(frozen_unix=harness.time.time(),jobs=matrix(),config=configuration(),
        candidate_lock_sha256=archive_io.sha(SHARED/'CANDIDATE_LOCK.json'),
        specs={c:cases.design(c) for c in cases.CASES},
        data_sha256={f'{c}_s{seed}':harness.arrays_hash(*cases.training_data(c,'train',seed),*cases.training_data(c,'validation',seed))
            for c in cases.CASES for seed in cases.seeds(c)},files={p.relative_to(ROOT).as_posix():archive_io.sha(p) for p in files},
        budget=b,test_labels_accessed=False,base_commit=harness.subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
        paired_test_protection=dict(relative=.1,absolute=1e-8)))


def train():return harness.train(sys.modules[__name__])


supervisor=harness.previous('study',cases=cases,worker=worker)
supervisor.HERE,supervisor.ROOT=HERE,ROOT
supervisor.budget=budget


if __name__=='__main__':
    command=sys.argv[1]
    if command=='run':
        preflight();freeze();train()
        from experiments.completion_release_validation_20261104.evaluate import run
        run()
    elif command in ('preflight','freeze','train'):globals()[command]()
    else:raise ValueError('Use preflight, freeze, train or run.')
