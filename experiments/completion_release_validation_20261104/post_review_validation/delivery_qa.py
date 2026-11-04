"""Read-only final evidence checks; writes only this review's delivery receipt."""
import hashlib,json,sys,time
from pathlib import Path

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent.parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from experiments.completion_release_validation_20261104 import study

def disk(p):
    p=str(Path(p).resolve())
    return Path(p if p.startswith('\\\\?\\') else '\\\\?\\'+p)

def sha(p):return hashlib.sha256(disk(p).read_bytes()).hexdigest()
def read(p):return json.loads(disk(p).read_text(encoding='utf-8'))
checks={}
def check(ok,key):
    checks[key]=bool(ok)
    if not ok:raise AssertionError(key)

out=HERE/'post_review_validation/DELIVERY_QA.json'
assert not out.exists(),'Keep prior delivery checkpoint.'
r=read(HERE/'post_review_checks.json')
f=read(HERE/'FREEZE.json');m=read(HERE/'ALL_MODELS_FROZEN.json')
s=read(HERE/'SUMMARY.json');extra=read(HERE/'post_review_validation/extra_checks.json')
check(r['issues']==[] and r['status']=='independent_reconstruction_consistent_scientific_adoption_failed','final_main_review_status')
check(all(sha(HERE/n)==h for n,h in r['receipt_sha256'].items()),'all_immutable_receipt_hashes')
check(len(f['files'])==8691 and all(sha(ROOT/n)==h for n,h in f['files'].items()),'8691_live_input_hashes')
check(all(sha(HERE/'frozen_inputs'/n)==h for n,h in f['files'].items()),'8691_archived_input_hashes')
archive=disk(HERE/'frozen_inputs')
check({p.relative_to(archive).as_posix() for p in archive.rglob('*') if p.is_file()}==set(f['files']),'exact_archive_members')
check(len(m['files'])==352 and all(sha(HERE/n)==h for n,h in m['files'].items()),'352_run_hashes')
check({p.relative_to(HERE).as_posix() for p in (HERE/'runs').rglob('*') if p.is_file()}==set(m['files']),'exact_run_members')
check({p.name for p in (HERE/'runs').iterdir() if p.is_dir()}=={study.name(j) for j in f['jobs']} and len(f['jobs'])==32,'exact_32_job_directories')
check(study.matrix()==f['jobs'] and study.configuration()==f['config'],'actual_new_matrix_configuration')
lock=read(study.SHARED/'CANDIDATE_LOCK.json')
check(len(lock['source_files'])==41 and all(sha(ROOT/n)==h and sha(HERE/'source'/n)==h for n,h in lock['source_files'].items()),'41_locked_source_and_copies')
check({p.relative_to(ROOT).as_posix() for p in (ROOT/'src').rglob('*.py')}==set(lock['source_files']),'exact_source_members')
check(sha(study.SHARED/'CANDIDATE_LOCK.json')==r['source_lock_sha256'],'source_lock_hash')
prior=study.check_stage_gate()
check(prior==r['upstream_gate'] and prior['status']=='passed','complete_prerequisite_and_historical_chain')
old=ROOT/'experiments/completion_historical_20261031'
check(read(old/'SUMMARY.json')['technical_pass'] is False and read(old/'SUMMARY.json')['adoption_pass'] is False and read(old/'VERIFICATION.json')['status']=='failed','original_failed_verdict_preserved')
check(not (old/'POST_REVIEW_ACCEPTANCE.json').exists(),'no_original_failed_acceptance')
check(s['technical_pass'] is True and s['adoption_pass'] is False and s['protected_outputs']==32 and all(v['both_accepted']==5 for v in s['methods'].values()),'final_scientific_failure_unchanged')
check(not (HERE/'POST_REVIEW_ACCEPTANCE.json').exists(),'review_wrote_no_acceptance')
check(sha(HERE/r['executed_numeric_script_path'])==r['script_sha256'],'executed_original_script_binding')
check(sha(HERE/'post_review_validation/audit.py')==r['corrected_numeric_script_sha256'],'corrected_unexecuted_script_binding')
check(sha(HERE/'post_review_validation/checkpoint_numeric.json')==r['numeric_checkpoint_sha256'],'numeric_checkpoint_binding')
check(sha(HERE/'post_review_validation/resume_review.py')==r['resume_script_sha256'],'resume_script_binding')
check(sha(ROOT/'experiments/new_task_validation_20261030/post_review_audit.py')==r['helper_sha256'],'raw_helper_binding')
check(extra['status']=='passed' and extra['issues']==[] and extra['main_check_receipt_sha256']==sha(HERE/'post_review_checks.json'),'extra_receipt_final_main_binding')
check(extra['script_sha256']==sha(HERE/'post_review_validation/extra_checks.py'),'extra_script_binding')
check(all(p['graph_json_equal'] and p['constant_bits_equal'] and p['best_json_equal'] for p in extra['exact_paired_best']) and len(extra['exact_paired_best'])==16,'all_16_exact_pairs')
report=disk(HERE/'POST_RUN_REVIEW.md').read_text(encoding='utf-8')
check('adoption_pass=false' in report and '5/16' in report and 'not external blind' in report and 'was not rerun' in report,'report_material_scope_and_correction_disclosed')
files=['POST_RUN_REVIEW.md','post_review_checks.json']+[p.relative_to(HERE).as_posix() for p in (HERE/'post_review_validation').iterdir() if p.is_file()]
receipt=dict(status='passed',checked_unix=time.time(),checks=checks,issues=[],upstream_gate=prior,
    live_inputs=8691,archived_inputs=8691,run_artifacts=352,job_directories=32,source_files=41,
    technical_pass=True,adoption_pass=False,no_acceptance_written=True,
    review_output_sha256={n:sha(HERE/n) for n in sorted(files)})
out.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps(dict(status='passed',checks=len(checks),report_sha256=sha(HERE/'POST_RUN_REVIEW.md'),main_checks_sha256=sha(HERE/'post_review_checks.json'),delivery_qa_sha256=sha(out)),indent=2))
