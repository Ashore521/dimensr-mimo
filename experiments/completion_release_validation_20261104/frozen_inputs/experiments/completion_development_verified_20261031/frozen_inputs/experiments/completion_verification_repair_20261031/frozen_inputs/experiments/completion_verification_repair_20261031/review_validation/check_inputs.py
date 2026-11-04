"""Bounded read-only JSON/history checks for the pre-run review."""
import hashlib,json,sys,time
from pathlib import Path

HERE=Path(__file__).resolve().parent.parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
import verify_sealed as repair

read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
historical=repair.HISTORICAL
review=read(historical/'post_review_checks.json')
test=read(HERE/'TEST_VALIDATION.json')
f=read(historical/'FREEZE.json');m=read(historical/'ALL_MODELS_FROZEN.json')
checks={
    'test_receipt_passed_29':test['status']=='passed' and test['tests']==29,
    'test_receipt_hashes':all(sha(HERE/n)==h for n,h in test['sha256'].items()),
    'original_receipt_hashes':all(sha(historical/n)==h for n,h in review['key_sha256'].items()),
    'original_inputs_unchanged':all(sha(ROOT/n)==h for n,h in f['files'].items()),
    'original_archive_unchanged':all(sha(historical/'frozen_inputs'/n)==h for n,h in f['files'].items()),
    'original_runs_unchanged':all(sha(historical/n)==h for n,h in m['files'].items()),
    'original_summary_still_false':read(historical/'SUMMARY.json')['technical_pass'] is False and read(historical/'SUMMARY.json')['adoption_pass'] is False,
    'repair_execution_not_started':not any((HERE/n).exists() for n in ['FREEZE.json','VERIFICATION.json','SUMMARY.json','EXECUTION.json','verification.log']),
    'no_original_acceptance':not (historical/'POST_REVIEW_ACCEPTANCE.json').exists(),
}
pairs=[]
for job in f['jobs']:
    if job['method']!='base':continue
    current=read(historical/'runs'/repair.study.name(job)/'model.json')
    oldjob=dict(job,method='repair')
    prior=read(repair.study.PRIOR/'runs'/repair.study.name(oldjob)/'model.json')
    pairs.append(dict(job=job,phase_preconditions_and_normalized_identity=repair.normalized_model(current)==repair.normalized_model(prior)))
checks['all_17_control_pairs_pass_pure_history_classifier']=len(pairs)==17 and all(p['phase_preconditions_and_normalized_identity'] for p in pairs)
deadline=repair.study.budget()
result=dict(status='checks_passed' if all(checks.values()) else 'findings',checks=checks,control_pairs=pairs,
    checked_unix=time.time(),original_deadline_unix=deadline,remaining_original_seconds=deadline-time.time(),
    sha256={n:sha(HERE/n) for n in ['PLAN.md','verify_sealed.py','test_verify.py','TEST_VALIDATION.json','tests_01.log','tests_02.log']},
    scope='Only JSON/hashes and pure normalized_model on 17 sealed control pairs; no formal verify/run, data generation, fitting, search or full verifier replay.')
(HERE/'PRE_RUN_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='control_pairs'},indent=2))
assert all(checks.values())
