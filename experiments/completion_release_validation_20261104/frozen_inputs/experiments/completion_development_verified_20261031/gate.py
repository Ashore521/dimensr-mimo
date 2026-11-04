"""Accept a reviewed sealed correction while preserving the original failure."""
import json
import math

from experiments.compositional_completion_20261031 import harness
from experiments.completion_verification_repair_20261031.verify_sealed import ARTIFACTS
from experiments.native_api_20261026.run import read,sha
from experiments.native_api_20261026.verify import require

RECEIPT_FILES = ('FREEZE.json','VERIFICATION.json','SUMMARY.json','EXECUTION.json',
                 'FROZEN_INPUT_ARCHIVE.json','POST_RUN_REVIEW.md','CANDIDATE_LOCK.json')
UNUSED_EVIDENCE = ('PREFLIGHT.json','FREEZE.json','BUDGET.json','runs','TRAINING_STARTED.json',
    'ALL_MODELS_FROZEN.json','SCORES.json','SCORING_FAILURE.json','VERIFICATION.json','EXECUTION.json','supervisor.log')


def number(value):
    require(type(value) in (int,float) and math.isfinite(value),'Invalid numeric receipt field.')
    return value


def integer(value,expected):
    require(type(value) is int and value==expected,'Invalid receipt quantity.')


def same_config(a,b):
    require(json.dumps(a,sort_keys=True,allow_nan=False)==json.dumps(b,sort_keys=True,allow_nan=False),'Frozen configuration drift.')


def archive(s,folder,frozen):
    receipt = read(folder/'FROZEN_INPUT_ARCHIVE.json')
    require(receipt['status']=='archived_original_bytes' and receipt['freeze_sha256']==sha(folder/'FREEZE.json'),'Archive chain drift.')
    require(receipt['files']==frozen['files'],'Archive manifest drift.')
    integer(receipt['file_count'],len(frozen['files']))
    s.verify_files(frozen['files'])
    s.verify_files(frozen['files'],folder/'frozen_inputs')
    return receipt


def check(s):
    s.check_candidate_lock()
    historical = s.HERE.parent/'completion_historical_20261031'
    correction = s.HERE.parent/'completion_verification_repair_20261031'
    prepared = s.HERE.parent/'completion_development_20261031'
    require(not any((prepared/n).exists() for n in UNUSED_EVIDENCE),'Original development budget already used or interrupted.')
    lock_path = s.SHARED/'CANDIDATE_LOCK.json';lock = read(lock_path);lock_hash = sha(lock_path)
    receipt = read(correction/'POST_REVIEW_ACCEPTANCE.json')
    require(receipt['status']=='accepted_for_sealed_historical_correction','Independent correction review not accepted.')
    require(isinstance(receipt['sha256'],dict) and set(receipt['sha256'])==set(RECEIPT_FILES),'Correction receipt key drift.')
    expected = {n:sha(correction/n) for n in RECEIPT_FILES if n!='CANDIDATE_LOCK.json'}
    expected['CANDIDATE_LOCK.json'] = lock_hash
    require(receipt['sha256']==expected,'Correction acceptance hash drift.')

    fixed = read(correction/'FREEZE.json');verified = read(correction/'VERIFICATION.json')
    summary = read(correction/'SUMMARY.json');execution = read(correction/'EXECUTION.json')
    require(set(fixed['original_sha256'])==set(ARTIFACTS),'Original failure manifest key drift.')
    require(fixed['original_sha256']=={n:sha(historical/n) for n in ARTIFACTS},'Original failed evidence drift.')
    require(verified['original_sha256']==fixed['original_sha256'],'Corrected verification source drift.')
    require(verified['freeze_sha256']==expected['FREEZE.json'],'Corrected verification freeze drift.')
    require(fixed['no_fit_search_or_rescoring'] is True and fixed['original_verdict']=='failed','Correction training boundary drift.')
    require(verified['status']=='passed' and verified['budget_met'] is True and verified['inputs_unchanged'] is True and verified['original_verdict']=='failed','Correction verification did not pass.')
    integer(verified['expected_systems'],34);integer(verified['verified_systems'],34)
    for name,value in (('expected_systems',34),('expected_pairs',17),('expected_outputs',34),('protected_outputs',34)):
        integer(summary[name],value)
    require(summary['technical_pass'] is True and summary['adoption_pass'] is True and summary['full_denominator'] is True
        and summary['original_verdict']=='failed' and summary['verification_of_sealed_results'] is True,'Correction summary gate failed.')
    require(execution['verification_passed'] is True and execution['no_training_performed'] is True
        and execution['termination_reason'] is None,'Correction execution failed or trained models.')
    integer(execution['returncode'],0)

    original = read(historical/'FREEZE.json');old_summary = read(historical/'SUMMARY.json')
    old_verification = read(historical/'VERIFICATION.json');old_execution = read(historical/'EXECUTION.json')
    old_all = read(historical/'ALL_MODELS_FROZEN.json');old_scores = read(historical/'SCORES.json')
    received = read(historical/'POST_REVIEW_RECEIVED.json')
    require(old_summary['technical_pass'] is False and old_summary['adoption_pass'] is False
        and old_verification['status']=='failed' and old_execution['verification_passed'] is False,'Original failure was changed.')
    require(received['status']=='received_consistent_with_original_failure' and received['original_technical_pass'] is False
        and received['original_adoption_pass'] is False and received['development_authorized'] is False,'Original failed review receipt drift.')
    s.verify_files(received['sha256'],historical)
    integer(len(original['jobs']),34)
    require(len(verified['rows'])==34 and all(r['verified'] is True for r in verified['rows'])
        and [r['job'] for r in verified['rows']]==original['jobs'],'Corrected row denominator/order drift.')
    keys = [(j['case'],j['seed'],j['method']) for j in original['jobs']]
    require(len(set(keys))==34,'Original job identity denominator drift.')
    require(old_all['freeze_sha256']==sha(historical/'FREEZE.json') and old_scores['all_models_sha256']==sha(historical/'ALL_MODELS_FROZEN.json'),'Original all-model/score chain drift.')
    require(old_verification['freeze_sha256']==sha(historical/'FREEZE.json') and old_verification['models_sha256']==sha(historical/'ALL_MODELS_FROZEN.json')
        and old_verification['scores_sha256']==sha(historical/'SCORES.json'),'Original verification chain drift.')

    require(original['candidate_lock_sha256']==lock_hash,'Original source lock drift.')
    same_config(s.configuration(),original['config']);same_config(s.configuration(),lock['shared_configuration'])
    require(lock['stage_plan_sha256']['completion_development_20261031']==sha(prepared/'PLAN.md'),'Original development plan drift.')
    require(lock['stage_plan_sha256']['completion_historical_20261031']==sha(historical/'PLAN.md'),'Original historical plan drift.')
    budget = read(historical/'BUDGET.json');same_config(original['budget'],budget)
    integer(budget['limit_seconds'],2880);integer(fixed['verification_cap_seconds'],600)
    start = number(fixed['frozen_unix']);deadline = number(fixed['deadline_unix'])
    old_deadline = number(budget['start_unix'])+2880
    require(number(fixed['original_deadline_unix'])==old_deadline and start<deadline<=min(old_deadline,start+600),'Correction deadline bound drift.')
    verified_time = number(verified['verified_unix'])
    require(number(original['frozen_unix'])<=number(old_all['frozen_unix'])<=number(old_scores['scored_unix'])
        <=number(old_verification['verified_unix'])<=start<=verified_time<=deadline,'Original/correction chronology drift.')
    require(0<=number(execution['wall_seconds'])<=deadline-start,'Correction wall budget drift.')
    if 'accepted_unix' in receipt:
        require(number(receipt['accepted_unix'])>=verified_time,'Correction accepted before verification.')
    original_archive = archive(s,historical,original)
    require(original_archive['runs_unchanged'] is True,'Original runtime archive drift.')
    integer(original_archive['sealed_run_file_count'],len(old_all['files']))
    s.verify_files(old_all['files'],historical)
    corrected_archive = archive(s,correction,fixed)
    require(corrected_archive['original_historical_inputs_and_runs_unchanged'] is True,'Correction changed historical inputs or models.')
    return dict(original_inputs=len(original['files']),original_runtime_files=len(old_all['files']),
        correction_inputs=len(fixed['files']),candidate_lock_sha256=lock_hash,acceptance_sha256=sha(correction/'POST_REVIEW_ACCEPTANCE.json'))


def freeze_inputs(s):
    inherited = harness.freeze_inputs(s)
    historical = s.HERE.parent/'completion_historical_20261031'
    correction = s.HERE.parent/'completion_verification_repair_20261031'
    obsolete = historical/'POST_REVIEW_ACCEPTANCE.json'
    require(obsolete in inherited,'Frozen predecessor gate dependency changed.')
    files = [p for p in inherited if p!=obsolete]
    for folder in (historical,correction):
        frozen = read(folder/'FREEZE.json')
        files += [s.ROOT/n for n in frozen['files']]+[folder/'frozen_inputs'/n for n in frozen['files']]
    files += [historical/n for n in ARTIFACTS]+[historical/'BUDGET.json']
    files += [historical/n for n in read(historical/'ALL_MODELS_FROZEN.json')['files']]
    files += [historical/n for n in read(historical/'POST_REVIEW_RECEIVED.json')['sha256']]
    files += [p for p in correction.rglob('*') if p.is_file() and p.suffix!='.pyc' and '__pycache__' not in p.parts]
    files += [correction/n for n in RECEIPT_FILES if n!='CANDIDATE_LOCK.json']+[correction/'POST_REVIEW_ACCEPTANCE.json']
    files += [s.ROOT/'conftest.py']
    # Scratch fixtures remain on disk. Freeze the test code, logs and receipts
    # without recursively archiving new copies of those fixtures.
    validation = (s.HERE/'harness_validation', s.HERE/'harness_review_validation')
    return sorted({p for p in files if not any(
        p.is_relative_to(folder) and len(p.relative_to(folder).parts)>1
        for folder in validation)})
