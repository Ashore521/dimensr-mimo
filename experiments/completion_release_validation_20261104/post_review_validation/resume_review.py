"""Resume only the corrected stage lookup; preserve all completed raw calculations."""
import json,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent.parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from experiments.completion_release_validation_20261104 import study,gate
io=study.archive_io
read=lambda p:json.loads(io.disk_path(p).read_text(encoding='utf-8'))
OUT=HERE/'post_review_checks.json'
assert not OUT.exists(),'Keep final review checkpoint.'
checkpoint=HERE/'post_review_validation/checkpoint_numeric.json'
r=read(checkpoint);f=read(HERE/'FREEZE.json');m=read(HERE/'ALL_MODELS_FROZEN.json')
assert r['issues']==['actual_fixed_configuration_matrix'],'Resume only the identified reviewer stage-import issue.'
assert io.sha(HERE/'post_review_validation/audit_01_wrong_stage_import.py')==r['script_sha256']
checks={
    'correct_new_matrix':study.matrix()==f['jobs'],
    'correct_new_configuration':study.configuration()==f['config'],
    'numerical_checkpoint_receipts_unchanged':all(io.sha(HERE/n)==h for n,h in r['receipt_sha256'].items()),
    'all_frozen_inputs_live_and_archive_unchanged':all(io.sha(ROOT/n)==h and io.sha(HERE/'frozen_inputs'/n)==h for n,h in f['files'].items()),
    'all_run_artifacts_unchanged':all(io.sha(HERE/n)==h for n,h in m['files'].items()),
    'reviewed_harness_still_bound':read(HERE/'PRE_RUN_ACCEPTANCE.json')['harness_review_sha256']==io.sha(HERE/'HARNESS_REVIEW.md'),
}
prior_gate=study.check_stage_gate();checks['real_complete_eight_key_gate']=prior_gate['status']=='passed'
gate.validate_task_sources(study)
checks['all_16_exact_best_payloads']=len(r['selected_pair_identity'])==16 and all(p['graph_equal'] and p['constants_equal'] and p['best_payload_equal'] for p in r['selected_pair_identity'])
assert all(checks.values()),checks
r['historical_correction_gate_at_numeric_checkpoint']=r['upstream_gate']
r['upstream_gate']=prior_gate
r['status']='independent_reconstruction_consistent_scientific_adoption_failed'
r['issues']=[]
r['resumed_stage_lookup_checks']=checks
r['numeric_checkpoint_sha256']=io.sha(checkpoint)
r['executed_numeric_script_path']='post_review_validation/audit_01_wrong_stage_import.py'
r['corrected_numeric_script_sha256']=io.sha(HERE/'post_review_validation/audit.py')
r['resume_script_sha256']=io.sha(Path(__file__))
r['resumed_unix']=time.time()
r['reviewer_incident']='Numeric loop completed; the matrix/gate-only lookup had imported predecessor study. Corrected that import, checked the new matrix/config/full gate and all source/archive/run hashes, and reused the unchanged completed numerical checkpoint. Original checkpoint, source and log retained.'
OUT.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps(dict(status=r['status'],issues=r['issues'],resume_checks=checks,prior_gate=prior_gate,receipt_sha256=io.sha(OUT)),indent=2))
