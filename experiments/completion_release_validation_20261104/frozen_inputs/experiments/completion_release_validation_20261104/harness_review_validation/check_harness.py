"""Static/read-only new-harness audit; predecessor results are not interpreted."""
import ast,hashlib,json,sys,time
from pathlib import Path

HERE=Path(__file__).resolve().parent.parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from experiments.completion_release_validation_20261104 import study,gate,evaluate,worker,run_bounded
from experiments.compositional_completion_20261031 import harness,verification

io=study.archive_io
read=lambda p:json.loads(io.disk_path(p).read_text(encoding='utf-8'))
issues=[]
def check(ok,label):
    if not ok:issues.append(label)

def main():
    out=HERE/'HARNESS_REVIEW_CHECKS.json'
    if out.exists():raise RuntimeError('Preserve review checkpoint.')
    names=['PLAN.md','PROTOCOL.md','CANDIDATE_LOCK.json','cases.py','test_cases.py','NOVELTY_AUDIT.json','ORACLE_AUDIT.json',
           'DESIGN_REVIEW.md','DESIGN_CHECKS.json','SOURCE_VERSION_NOTE.json','study.py','gate.py','worker.py','evaluate.py','run_bounded.py','test_stage.py','IMPLEMENTATION.md','IMPLEMENTATION_CHECKS.json','TEST_VALIDATION.json']
    baseline={n:io.sha(HERE/n) for n in names}
    implementation=read(HERE/'IMPLEMENTATION_CHECKS.json')
    changed_worker_paths=[n for n,h in implementation['owned_sha256'].items() if io.sha(ROOT/n)!=h]
    check(changed_worker_paths==['experiments/completion_release_validation_20261104/gate.py'],'only_declared_post_worker_append')
    addition="    files += [s.ROOT/'experiments/conftest.py']\n"
    current_gate=(HERE/'gate.py').read_text(encoding='utf-8')
    check(current_gate.count(addition)==1 and hashlib.sha256(current_gate.replace(addition,'').encode()).hexdigest()==implementation['owned_sha256']['experiments/completion_release_validation_20261104/gate.py'],'exact_single_dependency_line_change')
    external_adapter=ROOT/'experiments/conftest.py';adapter_hash=io.sha(external_adapter)
    check(all(io.sha(ROOT/n)==h for n,h in implementation['immutable_interface_sha256'].items()),'unchanged_shared_interfaces')
    study.check_candidate_lock();gate.validate_task_sources(study)
    local=read(HERE/'CANDIDATE_LOCK.json');parent=read(study.SHARED/'CANDIDATE_LOCK.json')
    check(len(local['source_files'])==41 and local['source_files']==parent['source_files'],'same_41_source_map')
    check(study.configuration()==parent['shared_configuration'],'same_locked_configuration')
    matrix=study.matrix()
    check(study.EXPECTED==32 and study.LIMIT==2880 and len(matrix)==32 and len({(j['case'],j['seed'],j['method']) for j in matrix})==32,'exact_32_matrix')
    check({j['seed'] for j in matrix}=={10241,10242} and all(j['seconds']==60 for j in matrix),'new_seeds_and_cap')
    check(study.cases.FAMILIES==('log_affine_ratio','reciprocal_quadratic_norm','oscillatory_affine_saturation','exponential_affine_saturation'),'four_planned_families')
    v=evaluate.verifier
    check(v.base.cases is v.legacy.cases is study.cases and v.base.study is v.legacy.study is study and v.base.HERE==v.legacy.HERE==study.HERE,'isolated_native_verifier_binding')
    check(v.base.verify_cell.__globals__['cases'] is study.cases and v.base.evaluate.__globals__['guards'].__self__ is v,'native_callbacks_binding')
    check(worker.worker_run is harness.worker_run and run_bounded.bounded is harness.bounded and study.supervisor.HERE==study.HERE and study.supervisor.budget is study.budget,'unchanged_worker_and_supervisor')
    try:v.control_identity({}, {})
    except RuntimeError as error:check('No archived' in str(error),'explicit_no_archive_error')
    else:check(False,'explicit_no_archive_error')
    tree=ast.parse((HERE/'evaluate.py').read_text(encoding='utf-8'))
    klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='NewTaskVerifier')
    verify=next(n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name=='verify')
    check(not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='control_identity' for n in ast.walk(verify)),'no_archived_control_call')
    # Observe pending without ever opening predecessor quantitative result receipts.
    forbidden={'SCORES.json','VERIFICATION.json','SUMMARY.json','EXECUTION.json','POST_RUN_REVIEW.md','POST_REVIEW_ACCEPTANCE.json'}
    original_read=gate.read;gate_reads=[]
    def guarded_read(path):
        p=Path(path);gate_reads.append(str(p))
        if p.parent==study.PRIOR and p.name in forbidden:raise AssertionError('Prior terminal result interpretation deferred until acceptance.')
        return original_read(path)
    gate.read=guarded_read
    try:prior_status=gate.status(study)
    finally:gate.read=original_read
    check(prior_status['status']=='pending_missing_prior' and 'POST_REVIEW_ACCEPTANCE.json' in prior_status['missing'],'actual_pending_not_accepted')
    # This is the old historical/correction gate, not the running/new development scorer.
    upstream=gate.development_gate.check(gate.development)
    inherited=set(gate.development_gate.freeze_inputs(gate.development));current=set(study.freeze_inputs())
    check(inherited<=current,'zero_inherited_dependencies_dropped')
    design=read(HERE/'DESIGN_CHECKS.json');novelty=read(HERE/'NOVELTY_AUDIT.json');note=read(HERE/'SOURCE_VERSION_NOTE.json')
    check(note['captured_history_sha256']==novelty['source_files'][note['history_source']] and io.sha(HERE/'history_sources'/note['history_source'])==note['captured_history_sha256'],'captured_history_note')
    check(io.sha(ROOT/note['history_source'])==note['live_plan_sha256_at_note'],'later_live_plan_note')
    prior_frozen=read(study.PRIOR/'FREEZE.json')
    check(prior_frozen['files'][note['history_source']]==note['live_plan_sha256_at_note'],'predecessor_final_plan_separately_frozen')
    registry={ROOT/n for n in design['novelty']['source_freeze_sha256']};historical={HERE/'history_sources'/n for n in novelty['source_files']}
    check(len(registry)==13 and registry<=current and len(historical)==54 and historical<=current,'registries_and_snapshot_sources_in_closure')
    check(Path(io.__file__).resolve() in current,'local_archive_io_dependency_frozen')
    check(external_adapter in current,'experiments_adapter_frozen')
    check('experiments/conftest.py' not in prior_frozen['files'],'new_adapter_did_not_overwrite_frozen_prior_input')
    missing=[];bad=[];hashes={};hashed=0
    active=HERE/'harness_review_validation/check_harness.log'
    for path in sorted(current):
        if not io.disk_path(path).is_file():missing.append(path);continue
        try:
            digest=io.sha(path);hashed+=1
            if path!=active:hashes[path.relative_to(ROOT).as_posix()]=digest
        except OSError as error:bad.append(dict(path=str(path),error=repr(error)))
    allowed={HERE/n for n in ['HARNESS_REVIEW.md','TEST_VALIDATION.json','PREFLIGHT.json']}
    allowed|={study.PRIOR/n for n in gate.RECEIPT_FILES+('POST_REVIEW_ACCEPTANCE.json','BUDGET.json','TRAINING_STARTED.json')}
    allowed|={study.PRIOR/'frozen_inputs'/n for n in prior_frozen['files']}
    check(set(missing)<=allowed,'only_declared_pending_dependencies_missing')
    check(not bad,'all_available_dependencies_hash_readable')
    future=sorted([(len(str(HERE/'frozen_inputs'/p.relative_to(ROOT))),p,HERE/'frozen_inputs'/p.relative_to(ROOT)) for p in current if io.disk_path(p).is_file()],reverse=True)
    probes=[]
    for length,source,dest in future[:5]:
        owned=HERE/'harness_review_validation'/'archive_probe'/source.relative_to(ROOT)
        row=dict(source=str(source),future_destination=str(dest),future_length=length,probe_length=len(str(owned)))
        try:
            expected=io.sha(source);io.copy_original(source,owned,expected)
            own_physical=io.disk_path(owned)
            row['independent_copy_read_equal']=hashlib.sha256(own_physical.read_bytes()).hexdigest()==expected
        except OSError as error:row.update(independent_copy_read_equal=False,error=repr(error))
        finally:
            if io.disk_path(owned).is_file():io.disk_path(owned).unlink()
        probes.append(row)
    check(all(p['independent_copy_read_equal'] for p in probes),'deep_archive_copy_read_probes')
    official=['PREFLIGHT.json','FREEZE.json','BUDGET.json','runs','TRAINING_STARTED.json','ALL_MODELS_FROZEN.json','SCORES.json','SCORING_FAILURE.json','VERIFICATION.json','EXECUTION.json','supervisor.log']
    check(not any((HERE/n).exists() for n in official),'no_new_formal_stage_execution')
    check(all(io.sha(HERE/n)==h for n,h in baseline.items()) and io.sha(external_adapter)==adapter_hash,'review_inputs_unchanged')
    runtime_path=Path('C:/Users/608/.codex/sessions/2026/10/30/rollout-2026-10-30T17-07-49-01a18c10-baea-7b10-bd6d-96f104545fe5.jsonl')
    runtime=None
    for number,line in enumerate(runtime_path.open(encoding='utf-8'),1):
        r=json.loads(line)
        if r.get('type')=='turn_context':
            p=r['payload'];runtime=dict(path=str(runtime_path),line=number,timestamp=r.get('timestamp'),model=p.get('model'),effort=p.get('effort'))
    check(runtime['model']=='gpt-6.1-sol' and runtime['effort']=='high','actual_sol_high')
    logs={p.name:dict(sha256=io.sha(p),text=p.read_text(encoding='utf-8').strip()) for p in [HERE/'harness_review_validation/pytest_final_review.log']}
    result=dict(status='implementation_supported_pending_prior_acceptance' if not issues else 'review_findings',issues=issues,checked_unix=time.time(),runtime=runtime,
        input_sha256=baseline,post_worker_dependency_change_only=True,experiments_adapter_sha256=adapter_hash,source_count=41,local_candidate_lock_sha256=io.sha(HERE/'CANDIDATE_LOCK.json'),parent_lock_sha256=io.sha(study.SHARED/'CANDIDATE_LOCK.json'),
        prior_status=prior_status,prior_terminal_quantitative_results_not_interpreted=True,actual_pending_gate_reads=gate_reads,upstream_original_failed_and_correction_chain=upstream,
        matrix=matrix,config=study.configuration(),freeze_closure=dict(inherited=len(inherited),current=len(current),old_dropped=0,existing_hashed=hashed,missing=[p.relative_to(ROOT).as_posix() for p in missing],hash_read_errors=bad,input_sha256=hashes,maximum_available_future_archive_length=future[0][0],copy_read_probes=probes),
        predecessor_full_real_gate_checked=False,execution_clearance=False,test_logs=logs,static_design_reused_sha256=io.sha(HERE/'DESIGN_REVIEW.md'),script_sha256=io.sha(Path(__file__)),
        no_real_preflight_freeze_fit_search_or_scoring=True)
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['status','issues','prior_status','source_count','runtime']}))
    print(json.dumps({k:result['freeze_closure'][k] for k in ['inherited','current','old_dropped','existing_hashed','missing','maximum_available_future_archive_length','copy_read_probes']},indent=2));print('receipt_sha256',io.sha(out))

if __name__=='__main__':main()
