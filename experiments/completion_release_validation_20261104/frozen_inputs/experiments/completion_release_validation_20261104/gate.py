"""Source lock and accepted-development prerequisite for new output pairs."""
from pathlib import Path

from experiments.compositional_completion_20261031 import harness
from experiments.completion_development_verified_20261031 import gate as development_gate,study as development
from experiments.native_api_20261026.run import read,sha
from experiments.native_api_20261026.verify import require

RECEIPT_FILES=('FREEZE.json','ALL_MODELS_FROZEN.json','SCORES.json','VERIFICATION.json',
    'SUMMARY.json','EXECUTION.json','FROZEN_INPUT_ARCHIVE.json','POST_RUN_REVIEW.md')


class PendingPrior(RuntimeError):
    status='pending_missing_prior'
    def __init__(self,missing):
        self.missing=missing
        super().__init__(self.status+': '+', '.join(missing))


def check_candidate_lock(s):
    parent=read(s.SHARED/'CANDIDATE_LOCK.json');local=read(s.HERE/'CANDIDATE_LOCK.json')
    actual={p.relative_to(s.ROOT).as_posix() for p in (s.ROOT/'src').rglob('*.py')}
    require(bool(actual) and set(local['source_files'])==actual and local['source_files']==parent['source_files'],'Release source manifest drift.')
    require(parent.get('expected_source_count',len(actual))==len(actual),'Parent source count drift.')
    require(local['parent_candidate_lock_sha256']==sha(s.SHARED/'CANDIDATE_LOCK.json'),'Release parent lock drift.')
    s.verify_files(local['source_files'])
    development_gate.same_config(s.configuration(),parent['shared_configuration'])


def validate_task_sources(s):
    design=read(s.HERE/'DESIGN_CHECKS.json');novelty=read(s.HERE/'NOVELTY_AUDIT.json')
    s.verify_files(design['input_sha256'],s.HERE)
    s.verify_files(design['novelty']['source_freeze_sha256'])
    # The audit version is the archived snapshot; unfrozen live plans may evolve.
    s.verify_files(novelty['source_files'],s.HERE/'history_sources')


def check(s):
    s.check_candidate_lock()
    prior=s.PRIOR
    required=RECEIPT_FILES+('POST_REVIEW_ACCEPTANCE.json','BUDGET.json','TRAINING_STARTED.json')
    missing=[n for n in required if not (prior/n).is_file()]
    if missing:raise PendingPrior(missing)
    acceptance=read(prior/'POST_REVIEW_ACCEPTANCE.json')
    require(acceptance['status']=='accepted_for_release_validation','Development review not accepted for release validation.')
    require(isinstance(acceptance['sha256'],dict) and set(acceptance['sha256'])==set(RECEIPT_FILES),'Development receipt key drift.')
    expected={n:sha(prior/n) for n in RECEIPT_FILES}
    require(acceptance['sha256']==expected,'Development acceptance hash drift.')
    lock_hash=sha(s.SHARED/'CANDIDATE_LOCK.json')
    require(acceptance['candidate_lock_sha256']==lock_hash,'Development receipt source lock drift.')
    frozen=read(prior/'FREEZE.json');sealed=read(prior/'ALL_MODELS_FROZEN.json')
    scores=read(prior/'SCORES.json');verified=read(prior/'VERIFICATION.json')
    summary=read(prior/'SUMMARY.json');execution=read(prior/'EXECUTION.json')
    require(frozen['candidate_lock_sha256']==lock_hash,'Development frozen source lock drift.')
    development_gate.same_config(frozen['config'],s.configuration())
    require(verified['status']=='passed' and verified['budget_met'] is True and summary['technical_pass'] is True,'Development technical gate failed.')
    require(type(summary['adoption_pass']) is bool,'Development science verdict must remain a boolean.')
    require(summary['full_denominator'] is True,'Development denominator incomplete.')
    for value in (verified['expected_systems'],verified['verified_systems'],summary['expected_systems']):development_gate.integer(value,32)
    require(len(frozen['jobs'])==32 and len({(j['case'],j['seed'],j['method']) for j in frozen['jobs']})==32,'Development matrix denominator drift.')
    require(len(verified['rows'])==32 and [r['job'] for r in verified['rows']]==frozen['jobs'] and all(r['verified'] is True for r in verified['rows']),'Development verified row drift.')
    require(len(scores['rows'])==32 and [r['job'] for r in scores['rows']]==frozen['jobs'] and all(r['status']=='complete' for r in scores['rows']),'Development score row drift.')
    require(sealed['freeze_sha256']==expected['FREEZE.json'] and scores['all_models_sha256']==expected['ALL_MODELS_FROZEN.json'],'Development model/score chain drift.')
    require(verified['freeze_sha256']==expected['FREEZE.json'] and verified['models_sha256']==expected['ALL_MODELS_FROZEN.json'] and verified['scores_sha256']==expected['SCORES.json'],'Development verification chain drift.')
    development_gate.integer(execution['returncode'],0)
    require(execution['termination_reason'] is None and execution['verification_passed'] is True,'Development execution failed.')
    budget=read(prior/'BUDGET.json');development_gate.same_config(frozen['budget'],budget)
    for name,value in (('limit_seconds',2880),('verification_reserve_seconds',600),('system_cap_seconds',60),('fit_seconds',52),('sequential_workers',1),('threads',1)):
        development_gate.integer(budget[name],value)
    number=development_gate.number;deadline=number(budget['start_unix'])+2880
    previous=number(read(prior/'TRAINING_STARTED.json')['start_unix'])
    require(number(frozen['frozen_unix'])<=previous,'Development started before freeze.')
    expected_folders={s.name(j) for j in frozen['jobs']}
    require({p.name for p in (prior/'runs').iterdir() if p.is_dir()}==expected_folders,'Extra/missing development attempt folders.')
    for job in frozen['jobs']:
        folder=prior/'runs'/s.name(job);cell=read(folder/'CELL_FROZEN.json')
        require(previous<=number(cell['frozen_unix'])<=number(sealed['frozen_unix']),'Development cell chronology drift.')
        previous=cell['frozen_unix']
        require(sealed['files'].get((folder/'CELL_FROZEN.json').relative_to(prior).as_posix())==sha(folder/'CELL_FROZEN.json'),'Development cell seal missing.')
        s.verify_files(cell['files'],folder)
        for name,digest in cell['files'].items():
            require(sealed['files'].get((folder/name).relative_to(prior).as_posix())==digest,'Development cell artifact missing.')
        require(read(folder/'job.json')==job,'Development job changed.')
        resource=read(folder/'resources.json')
        require(resource['status']=='complete' and type(resource['returncode']) is int and resource['returncode']==0 and 0<=number(resource['wall_seconds'])<=60.5,'Development resource failure.')
    require(previous<=number(sealed['frozen_unix'])<=number(scores['scored_unix'])<=number(verified['verified_unix'])<=deadline,'Development terminal chronology/deadline drift.')
    require(0<=number(verified['elapsed_seconds'])<=2880 and 0<=number(execution['wall_seconds'])<=2940,'Development budget exceeded.')
    development_gate.archive(s,prior,frozen);s.verify_files(sealed['files'],prior)
    # Preserve and recheck the original failed history plus its accepted correction.
    development_gate.check(development)
    return dict(status='passed',prior_technical_pass=True,prior_adoption_pass=summary['adoption_pass'],
        prior_systems=32,prior_input_count=len(frozen['files']),prior_runtime_files=len(sealed['files']))


def status(s):
    try:return check(s)
    except PendingPrior as error:return dict(status=error.status,missing=error.missing)


def freeze_inputs(s):
    # The predecessor's dependency closure includes the original failure/correction.
    files=list(development_gate.freeze_inputs(development))
    for folder in (s.HERE,):
        files+=list(folder.glob('*.py'))+list(folder.glob('*.md'))+list(folder.glob('*.json'))
        for name in ('case_validation','harness_validation','design_review_validation','harness_review_validation'):
            files += [p for p in (folder/name).glob('*') if p.is_file() and p.suffix!='.pyc']
    files += [p for p in (s.HERE/'history_sources').rglob('*') if p.is_file()]
    design=read(s.HERE/'DESIGN_CHECKS.json')
    files += [s.ROOT/n for n in design['novelty']['source_freeze_sha256']]
    files += [s.ROOT/'experiments/conftest.py']
    files += [s.HERE/n for n in ('PLAN.md','PROTOCOL.md','CANDIDATE_LOCK.json','cases.py','test_cases.py',
        'NOVELTY_AUDIT.json','ORACLE_AUDIT.json','DESIGN_REVIEW.md','DESIGN_CHECKS.json','SOURCE_VERSION_NOTE.json','HARNESS_REVIEW.md','TEST_VALIDATION.json','PREFLIGHT.json')]
    files += [s.PRIOR/n for n in RECEIPT_FILES+('POST_REVIEW_ACCEPTANCE.json','BUDGET.json','TRAINING_STARTED.json')]
    if (s.PRIOR/'FREEZE.json').exists():
        frozen=read(s.PRIOR/'FREEZE.json')
        files += [s.ROOT/n for n in frozen['files']]+[s.PRIOR/'frozen_inputs'/n for n in frozen['files']]
    if (s.PRIOR/'ALL_MODELS_FROZEN.json').exists():
        files += [s.PRIOR/n for n in read(s.PRIOR/'ALL_MODELS_FROZEN.json')['files']]
    return sorted(set(files))
