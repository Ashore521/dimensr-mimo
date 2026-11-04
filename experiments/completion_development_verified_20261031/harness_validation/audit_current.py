"""Read-only continuation audit; writes only this stage's implementation receipt."""
import importlib.metadata
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from experiments.completion_development_verified_20261031 import study,gate
from experiments.completion_development_20261031 import study as prepared
from experiments.compositional_completion_20261031 import harness


def main():
    start=time.monotonic()
    result=study.check_stage_gate()
    elapsed=time.monotonic()-start
    assert len(study.matrix())==study.EXPECTED==32 and study.matrix()==prepared.matrix()
    assert study.configuration()==prepared.configuration()
    inactive={s.HERE.name:[n for n in gate.UNUSED_EVIDENCE if (s.HERE/n).exists()] for s in (study,prepared)}
    assert not any(inactive.values()),'A development stage has formal execution evidence.'
    inherited=set(harness.freeze_inputs(study));current=set(study.freeze_inputs())
    obsolete=study.HERE.parent/'completion_historical_20261031/POST_REVIEW_ACCEPTANCE.json'
    assert inherited-current=={obsolete}
    missing=sorted(p.relative_to(ROOT).as_posix() for p in current if not p.is_file())
    expected={f'experiments/{study.HERE.name}/{n}' for n in ('HARNESS_REVIEW.md','PREFLIGHT.json','TEST_VALIDATION.json')}
    assert set(missing)==expected,'Unexpected missing freeze dependency: '+repr(missing)
    runtime_source=Path(r'C:\Users\608\.codex\sessions\2026\10\30\rollout-2026-10-30T17-12-08-01a18c14-b107-7f01-956a-5c654fe78bca.jsonl')
    runtime=None
    with runtime_source.open(encoding='utf-8') as stream:
        for line in stream:
            try:row=json.loads(line)
            except ValueError:continue
            if row.get('type')=='turn_context':
                p=row['payload'];runtime=dict(timestamp=row.get('timestamp'),turn_id=p.get('turn_id'),model=p.get('model'),effort=p.get('effort'))
    assert runtime and runtime['model']=='gpt-6.1-sol' and runtime['effort']=='high'
    runtime.update(session_id='01a18c14-b107-7f01-956a-5c654fe78bca',source_path=str(runtime_source))
    harness.write(study.HERE/'harness_validation/runtime_model.json',runtime)
    owned=[study.HERE/n for n in ('gate.py','study.py','worker.py','evaluate.py','run_bounded.py','test_stage.py','PROTOCOL.md','IMPLEMENTATION.md')]
    evidence=[study.HERE/'PLAN.md',study.HERE/'IMPLEMENTATION_PAUSED.md',Path(__file__),
        study.HERE/'harness_validation/runtime_model.json']
    # The redirected audit stdout changes after this receipt is written.
    evidence+=sorted(p for p in (study.HERE/'harness_validation').glob('*.log') if not p.name.startswith('final_audit_'))
    sources=[study.SHARED/n for n in ('harness.py','verification.py','CANDIDATE_LOCK.json')]
    sources+=[study.HERE.parent/'completion_verification_repair_20261031'/n for n in ('verify_sealed.py','POST_REVIEW_ACCEPTANCE.json','POST_RUN_REVIEW.md')]
    versions={name:importlib.metadata.version(name) for name in ('numpy','scipy','torch','sympy','scikit-learn','psutil','pytest')}
    receipt=dict(status='implementation_constructed_and_readonly_checked_not_independently_reviewed',checked_unix=time.time(),
        python_executable=sys.executable,python_version=sys.version,versions=versions,runtime=runtime,
        commands=[
            '.venv/Scripts/python.exe -m pytest experiments/completion_development_verified_20261031/test_stage.py --basetemp=experiments/completion_development_verified_20261031/harness_validation/pytest_temp_01 -q',
            '.venv/Scripts/python.exe -m pytest experiments/completion_development_verified_20261031/test_stage.py --basetemp=experiments/completion_development_verified_20261031/harness_validation/p2 -q',
            '.venv/Scripts/python.exe experiments/completion_development_verified_20261031/harness_validation/audit_current.py'],
        tests=dict(first=dict(passed=2,setup_errors=47,reason='Win32 long fixture paths; original log retained'),final=dict(passed=49,failed=0,seconds=10.44)),
        real_gate=dict(result,elapsed_seconds=elapsed,status='passed'),formal_artifacts_present=inactive,
        freeze_closure=dict(inherited_paths=len(inherited),current_paths=len(current),removed_only=obsolete.relative_to(ROOT).as_posix(),
            missing_expected_before_review_and_preflight=missing,full_inherited_closure_retained_except_obsolete_acceptance=True),
        owned_sha256={p.relative_to(ROOT).as_posix():harness.sha(p) for p in owned},
        evidence_sha256={p.relative_to(ROOT).as_posix():harness.sha(p) for p in evidence},
        immutable_interface_sha256={p.relative_to(ROOT).as_posix():harness.sha(p) for p in sources},
        no_formal_case_generation=True,no_initializer_fit_search_or_scoring=True,publication_performed=False,
        independent_harness_review_completed=False,scientific_acceptance=False,audit_stdout_not_self_hashed=True,
        remaining=['Independent continuation-harness review','Main-agent final test receipt and freeze acceptance','Authorized single formal development run and its independent post-review'])
    harness.write(study.HERE/'IMPLEMENTATION_CHECKS.json',receipt)
    print(json.dumps({k:receipt[k] for k in ('status','tests','real_gate','formal_artifacts_present','freeze_closure')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
