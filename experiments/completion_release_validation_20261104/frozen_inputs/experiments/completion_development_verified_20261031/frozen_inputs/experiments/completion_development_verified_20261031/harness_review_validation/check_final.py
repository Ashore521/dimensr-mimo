"""Final read-only chain/closure review; no preflight, freeze, arrays or fit."""
import hashlib,json,os,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent.parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from experiments.completion_development_verified_20261031 import study,gate,evaluate,worker,run_bounded,archive_io
from experiments.completion_development_20261031 import study as prepared
from experiments.compositional_completion_20261031 import harness
from experiments.completion_verification_repair_20261031.verify_sealed import SealedVerifier

sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def independent_disk(path):
    text=str(path.absolute())
    return Path('\\\\?\\'+text) if os.name=='nt' and not text.startswith('\\\\?\\') else path
def independent_disk_sha(path):return hashlib.sha256(independent_disk(path).read_bytes()).hexdigest()
issues=[]
def check(ok,label):
    if not ok:issues.append(label)

def main():
    out=HERE/'HARNESS_REVIEW_CHECKS.json'
    if out.exists():raise RuntimeError('Keep review checkpoint.')
    inputs=[HERE/n for n in ['PLAN.md','PROTOCOL.md','gate.py','study.py','worker.py','evaluate.py','run_bounded.py','test_stage.py','archive_io.py','test_archive_io.py','TEST_VALIDATION.json','INPUT_PREARCHIVE.json','IMPLEMENTATION.md','IMPLEMENTATION_CHECKS.json']]
    inputs += [ROOT/'conftest.py',ROOT/'tests/conftest.py',ROOT/'tests/test_completion_historical_study.py']
    baseline={p.relative_to(ROOT).as_posix():sha(p) for p in inputs}
    started=time.monotonic();real=study.check_stage_gate();gate_seconds=time.monotonic()-started
    check(study.matrix()==prepared.matrix() and study.EXPECTED==32 and study.LIMIT==2880,'unchanged_matrix')
    check(study.configuration()==prepared.configuration(),'unchanged_configuration')
    check(study.cases is not prepared.cases and study.cases.HERE==study.HERE and prepared.cases.HERE==prepared.HERE,'isolated_cases')
    check(isinstance(evaluate.verifier,SealedVerifier) and evaluate.verifier.base.HERE==evaluate.verifier.legacy.HERE==study.HERE,'sealed_verifier_binding')
    check(evaluate.verifier.base.cases is evaluate.verifier.legacy.cases is study.cases and evaluate.verifier.base.study is evaluate.verifier.legacy.study is study,'verifier_case_study_binding')
    check(worker.worker_run is harness.worker_run and run_bounded.bounded is harness.bounded,'unchanged_worker_supervision')
    check(evaluate.verify.__func__ is SealedVerifier.verify and evaluate.summarize.__func__ is SealedVerifier.summarize,'inherited_full_verifier_summary')
    inherited=set(harness.freeze_inputs(study));current=set(study.freeze_inputs())
    obsolete=HERE.parent/'completion_historical_20261031/POST_REVIEW_ACCEPTANCE.json'
    validation=(HERE/'harness_validation',HERE/'harness_review_validation')
    nested=lambda p:any(p.is_relative_to(f) and len(p.relative_to(f).parts)>1 for f in validation)
    removed=inherited-current
    check(obsolete in removed and all(p==obsolete or nested(p) for p in removed),'only_obsolete_and_current_scratch_removed')
    check(not any(nested(p) for p in current),'no_current_nested_scratch_frozen')
    historical=HERE.parent/'completion_historical_20261031';correction=HERE.parent/'completion_verification_repair_20261031'
    mandatory=set()
    for folder in [historical,correction]:
        frozen=read(folder/'FREEZE.json')
        mandatory.update(ROOT/n for n in frozen['files']);mandatory.update(folder/'frozen_inputs'/n for n in frozen['files'])
    mandatory.update(historical/n for n in read(historical/'ALL_MODELS_FROZEN.json')['files'])
    check(mandatory<=current,'all_original_frozen_dependencies_archives_runs_retained')
    check(ROOT/'conftest.py' in current,'new_root_adapter_frozen')
    expected={HERE/n for n in ['HARNESS_REVIEW.md','PREFLIGHT.json']}
    missing=set();bad=[];file_hashes={};read_count=0
    for path in sorted(current):
        if not path.is_file():missing.add(path);continue
        try:
            digest=sha(path);read_count+=1
            if path!=HERE/'harness_review_validation/check_final.log':file_hashes[path.relative_to(ROOT).as_posix()]=digest
        except OSError as error:bad.append(dict(path=str(path),error=repr(error)))
    check(missing==expected,'only_expected_receipts_pending')
    check(not bad,'all_existing_inputs_actually_hash_readable')
    prearchive=read(HERE/'INPUT_PREARCHIVE.json')
    check(prearchive['file_count']==len(prearchive['files'])==3986,'actual_prearchive_count')
    check(set(prearchive['files'])=={p.relative_to(ROOT).as_posix() for p in current if not p.is_relative_to(HERE)},'prearchive_covers_all_stable_external_inputs')
    prearchive_mismatches=[n for n,h in prearchive['files'].items() if independent_disk_sha(ROOT/n)!=h or independent_disk_sha(HERE/'frozen_inputs'/n)!=h]
    check(not prearchive_mismatches,'independent_all_3986_source_archive_hashes')
    physical_archive=archive_io.disk_path(HERE/'frozen_inputs')
    archive_members={p.relative_to(physical_archive).as_posix() for p in physical_archive.rglob('*') if p.is_file()}
    check(archive_members==set(prearchive['files']),'exact_prearchive_members')
    archive_io.verify_files(prearchive['files'],HERE/'frozen_inputs')
    future=sorted([(len(str(HERE/'frozen_inputs'/p.relative_to(ROOT))),p,HERE/'frozen_inputs'/p.relative_to(ROOT)) for p in current if p.is_file()],reverse=True)
    probes=[]
    for length,source,dest in future[:5]:
        target=HERE/'harness_review_validation'/'final_copy_probe'/source.relative_to(ROOT)
        row=dict(source=str(source),future_destination=str(dest),future_length=length,probe_length=len(str(target)))
        try:
            archive_io.copy_original(source,target,independent_disk_sha(source))
            row['copy_read_hash_equal']=independent_disk_sha(target)==independent_disk_sha(source)
        except OSError as error:row.update(copy_read_hash_equal=False,error=repr(error))
        finally:
            physical_target=archive_io.disk_path(target)
            if physical_target.is_file():physical_target.unlink()
        probes.append(row)
    check(all(p['copy_read_hash_equal'] for p in probes),'deepest_archive_copy_read_probes')
    check(all(sha(ROOT/n)==h for n,h in baseline.items()),'reviewed_source_unchanged')
    inactive={folder.name:[n for n in gate.UNUSED_EVIDENCE if (folder/n).exists()] for folder in [HERE,prepared.HERE]}
    check(not any(inactive.values()),'both_development_budgets_unused')
    source_checks=read(HERE/'IMPLEMENTATION_CHECKS.json')
    old_snapshot_mismatches=[n for n,h in source_checks['owned_sha256'].items() if sha(ROOT/n)!=h]
    runtime=read(ROOT/'experiments/completion_release_validation_20261104/DESIGN_CHECKS.json')['runtime_metadata']
    check(runtime['model']=='gpt-6.1-sol' and runtime['effort']=='high','current_child_sol_high')
    adapter=read(HERE/'harness_review_validation/adapter_scope_checks.json')
    check(adapter['status']=='passed' and adapter['adapter_sha256']==sha(ROOT/'conftest.py') and adapter['original_torch_fixture_restored'] is True,'adapter_check_binding')
    result=dict(status='harness_supported' if not issues else 'review_findings',issues=issues,checked_unix=time.time(),reviewed_sha256=baseline,
        implementation_checks_is_prior_revision_snapshot=True,implementation_snapshot_changed_owned_paths=old_snapshot_mismatches,
        real_gate=dict(real,elapsed_seconds=gate_seconds),config=study.configuration(),matrix=study.matrix(),runtime_metadata=runtime,
        freeze_closure=dict(inherited=len(inherited),current=len(current),mandatory_old_dependency_count=len(mandatory),mandatory_old_dependencies_preserved=mandatory<=current,
            removed=[p.relative_to(ROOT).as_posix() for p in sorted(removed)],existing_inputs_hashed=read_count,missing_expected=[p.relative_to(ROOT).as_posix() for p in sorted(missing)],hash_read_errors=bad,
            maximum_future_destination_length=future[0][0],future_destination_count_over_259=sum(row[0]>259 for row in future),copy_read_probes=probes,input_sha256=file_hashes),
        actual_prearchive=dict(count=3986,independent_hash_mismatches=prearchive_mismatches,exact_members=True,old_creation_helper_sha256=prearchive['archive_io_sha256'],current_reverified_helper_sha256=sha(HERE/'archive_io.py')),
        formal_artifacts_present=inactive,adapter_checks=adapter,independent_test_logs={p.name:dict(sha256=sha(p),text=p.read_text(encoding='utf-8').strip()) for p in [HERE/'harness_review_validation/pytest_stage_io_final_review.log',HERE/'harness_review_validation/preseal_adapter_review.log']},
        no_preflight_freeze_case_fit_score=True,script_sha256=sha(Path(__file__)))
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['status','issues','real_gate','formal_artifacts_present']}));print(json.dumps({k:result['freeze_closure'][k] for k in ['inherited','current','existing_inputs_hashed','missing_expected','maximum_future_destination_length','future_destination_count_over_259','copy_read_probes']},indent=2));print('receipt_sha256',sha(out))

if __name__=='__main__':main()
