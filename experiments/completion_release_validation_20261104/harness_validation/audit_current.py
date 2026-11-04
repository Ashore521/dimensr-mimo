"""Read-only release-stage audit; never generates task arrays or starts a run."""
import importlib.metadata
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from experiments.completion_release_validation_20261104 import study,gate
from experiments.compositional_completion_20261031 import harness


def main():
    io=study.archive_io
    study.check_candidate_lock();gate.validate_task_sources(study)
    prior_status=gate.status(study)
    upstream=gate.development_gate.check(gate.development)
    assert len(study.matrix())==study.EXPECTED==32
    assert {j['seed'] for j in study.matrix()}=={10241,10242}
    design=harness.read(study.HERE/'DESIGN_CHECKS.json')
    novelty=harness.read(study.HERE/'NOVELTY_AUDIT.json')
    source_note=harness.read(study.HERE/'SOURCE_VERSION_NOTE.json')
    assert source_note['captured_history_sha256']==novelty['source_files'][source_note['history_source']]
    assert io.sha(study.HERE/'history_sources'/source_note['history_source'])==source_note['captured_history_sha256']
    assert io.sha(ROOT/source_note['history_source'])==source_note['live_plan_sha256_at_note']
    assert len(design['novelty']['source_freeze_sha256'])==design['novelty']['source_freeze_count']
    inherited=set(gate.development_gate.freeze_inputs(gate.development))
    files=set(study.freeze_inputs())
    assert inherited<=files,'An old dependency was dropped.'
    registry_paths={ROOT/n for n in design['novelty']['source_freeze_sha256']}
    assert registry_paths<=files and Path(io.__file__).resolve() in files
    archived_sources={study.HERE/'history_sources'/n for n in novelty['source_files']}
    assert archived_sources<=files
    formal_names=('PREFLIGHT.json','FREEZE.json','BUDGET.json','runs','TRAINING_STARTED.json','ALL_MODELS_FROZEN.json',
                  'SCORES.json','SCORING_FAILURE.json','VERIFICATION.json','EXECUTION.json','supervisor.log')
    formal=[n for n in formal_names if (study.HERE/n).exists()]
    assert not formal,'This implementation package must not start the release matrix.'
    missing=sorted(p.relative_to(ROOT).as_posix() for p in files if not io.disk_path(p).is_file())
    allowed={f'experiments/{study.HERE.name}/{n}' for n in ('HARNESS_REVIEW.md','TEST_VALIDATION.json','PREFLIGHT.json')}
    allowed|={f'experiments/{study.PRIOR.name}/{n}' for n in gate.RECEIPT_FILES+('POST_REVIEW_ACCEPTANCE.json','BUDGET.json','TRAINING_STARTED.json')}
    if prior_status['status']=='pending_missing_prior' and (study.PRIOR/'FREEZE.json').exists():
        prior_frozen=harness.read(study.PRIOR/'FREEZE.json')
        allowed|={f'experiments/{study.PRIOR.name}/frozen_inputs/{n}' for n in prior_frozen['files']}
    assert set(missing)<=allowed,'Unexpected missing execution dependency: '+repr(missing)
    runtime_path=Path(r'C:\Users\608\.codex\sessions\2026\10\30\rollout-2026-10-30T17-12-08-01a18c14-b107-7f01-956a-5c654fe78bca.jsonl')
    runtime=None
    with runtime_path.open(encoding='utf-8') as stream:
        for line in stream:
            try:r=json.loads(line)
            except ValueError:continue
            if r.get('type')=='turn_context':
                p=r['payload'];runtime=dict(timestamp=r.get('timestamp'),turn_id=p.get('turn_id'),model=p.get('model'),effort=p.get('effort'))
    assert runtime and runtime['model']=='gpt-6.1-sol' and runtime['effort']=='high'
    runtime.update(session_id='01a18c14-b107-7f01-956a-5c654fe78bca',source_path=str(runtime_path))
    harness.write(study.HERE/'harness_validation/runtime_model.json',runtime)
    owned=[study.HERE/n for n in ('study.py','gate.py','worker.py','evaluate.py','run_bounded.py','test_stage.py','PROTOCOL.md','IMPLEMENTATION.md')]
    evidence=[study.HERE/n for n in ('PLAN.md','CANDIDATE_LOCK.json','cases.py','test_cases.py','ORACLE_AUDIT.json','NOVELTY_AUDIT.json','DESIGN_REVIEW.md','DESIGN_CHECKS.json','SOURCE_VERSION_NOTE.json')]
    evidence+=[Path(__file__),study.HERE/'harness_validation/runtime_model.json']
    active_stdout=Path(sys.argv[1]).name if len(sys.argv)>1 else None
    evidence+=sorted(p for p in (study.HERE/'harness_validation').glob('*.log') if p.name!=active_stdout)
    immutable=[study.SHARED/n for n in ('harness.py','verification.py','CANDIDATE_LOCK.json')]+[Path(io.__file__).resolve()]
    versions={n:importlib.metadata.version(n) for n in ('numpy','scipy','torch','sympy','scikit-learn','psutil','pytest')}
    receipt=dict(status='constructed_pending_independent_harness_review_and_prior_terminal',checked_unix=time.time(),
        python_executable=sys.executable,python_version=sys.version,versions=versions,runtime=runtime,
        commands=[f'.venv/Scripts/python.exe -m pytest experiments/{study.HERE.name}/test_stage.py --basetemp=experiments/{study.HERE.name}/harness_validation/p{i} -q' for i in range(1,6)]
            +[f'.venv/Scripts/python.exe experiments/{study.HERE.name}/harness_validation/audit_current.py'+(' '+active_stdout if active_stdout else '')],
        tests=[dict(log='pytest_01.log',passed=39,failed=0,seconds=8.77),dict(log='pytest_02.log',passed=46,failed=0,seconds=9.39),
            dict(log='pytest_03.log',passed=47,failed=1,seconds=9.33,reason='Fixture expected RuntimeError before parent helper finalized ValueError; only expectation corrected.'),
            dict(log='pytest_04.log',passed=48,failed=0,seconds=9.22),dict(log='pytest_05.log',passed=51,failed=0,seconds=9.15)],
        candidate_lock_sha256=io.sha(study.HERE/'CANDIDATE_LOCK.json'),parent_candidate_lock_sha256=io.sha(study.SHARED/'CANDIDATE_LOCK.json'),
        source_count=len(harness.read(study.HERE/'CANDIDATE_LOCK.json')['source_files']),
        prior_status=prior_status,upstream_original_and_correction_integrity=upstream,
        historical_source_scope=dict(authority='NOVELTY history_sources archived byte snapshots',
            source_version_note_sha256=io.sha(study.HERE/'SOURCE_VERSION_NOTE.json'),
            live_unfrozen_plan_changes_not_treated_as_archived_source_drift=True),
        freeze_closure=dict(inherited_paths=len(inherited),current_paths=len(files),old_dependencies_dropped=0,
            source_registries=len(registry_paths),historical_source_files=len(archived_sources),missing_expected_pending=missing,
            maximum_logical_archive_path_length=max(len(str(study.HERE/'frozen_inputs'/p.relative_to(ROOT))) for p in files)),
        formal_artifacts_present=formal,owned_sha256={p.relative_to(ROOT).as_posix():io.sha(p) for p in owned},
        evidence_sha256={p.relative_to(ROOT).as_posix():io.sha(p) for p in evidence},
        immutable_interface_sha256={p.relative_to(ROOT).as_posix():io.sha(p) for p in immutable},
        archived_control_comparison='not_applicable_no_same_task_archive',
        no_formal_case_generation=True,no_initializer_fit_search_or_test_scoring=True,no_publication=True,
        independent_harness_review_completed=False,scientific_acceptance=False,audit_stdout_not_self_hashed=True)
    harness.write(study.HERE/'IMPLEMENTATION_CHECKS.json',receipt)
    print(json.dumps({k:receipt[k] for k in ('status','tests','source_count','prior_status','upstream_original_and_correction_integrity','freeze_closure','formal_artifacts_present')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
