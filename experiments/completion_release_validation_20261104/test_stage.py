"""Only synthetic receipts/arrays; no formal task generation, fit or scoring."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pytest

from experiments.completion_release_validation_20261104 import study,gate,worker,evaluate,run_bounded
from experiments.compositional_completion_20261031 import harness
from experiments.completion_development_verified_20261031 import study as prior_study


def put(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj),encoding='utf-8')


def update(path,**changes):
    obj=harness.read(path);obj.update(changes);put(path,obj)


def extended(path):return Path('\\\\?\\'+str(path.resolve())) if os.name=='nt' else path


@pytest.fixture
def prior_chain(tmp_path,monkeypatch):
    root=extended(tmp_path);here=root/'experiments/release';prior=root/'experiments/development';shared=root/'experiments/shared'
    for n,v in (('ROOT',root),('HERE',here),('PRIOR',prior),('SHARED',shared)):monkeypatch.setattr(study,n,v)
    config={'fixture':True};monkeypatch.setattr(study,'configuration',lambda:deepcopy(config))
    put(root/'src/stub.py',{})
    sources={'src/stub.py':harness.sha(root/'src/stub.py')}
    put(shared/'CANDIDATE_LOCK.json',dict(source_files=sources,expected_source_count=1,shared_configuration=config))
    put(here/'CANDIDATE_LOCK.json',dict(source_files=sources,parent_candidate_lock_sha256=harness.sha(shared/'CANDIDATE_LOCK.json')))
    jobs=[dict(case='fixture'+str(i//2),seed=1,method=('base','completion')[i%2],seconds=60) for i in range(32)]
    budget=dict(start_unix=1000.,limit_seconds=2880,verification_reserve_seconds=600,system_cap_seconds=60,fit_seconds=52,sequential_workers=1,threads=1)
    put(prior/'BUDGET.json',budget)
    put(prior/'FREEZE.json',dict(frozen_unix=1001.,jobs=jobs,config=config,budget=budget,files=sources,candidate_lock_sha256=harness.sha(shared/'CANDIDATE_LOCK.json')))
    for name in sources:
        dest=prior/'frozen_inputs'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/name,dest)
    put(prior/'TRAINING_STARTED.json',dict(start_unix=1002.))
    files={}
    for i,job in enumerate(jobs):
        folder=prior/'runs'/study.name(job)
        put(folder/'job.json',job);put(folder/'resources.json',dict(status='complete',returncode=0,wall_seconds=1.))
        cell_files={n:harness.sha(folder/n) for n in ('job.json','resources.json')}
        put(folder/'CELL_FROZEN.json',dict(frozen_unix=1003.+i,files=cell_files))
        for n in ('job.json','resources.json','CELL_FROZEN.json'):files[(folder/n).relative_to(prior).as_posix()]=harness.sha(folder/n)
    put(prior/'ALL_MODELS_FROZEN.json',dict(frozen_unix=1040.,files=files,freeze_sha256=harness.sha(prior/'FREEZE.json')))
    put(prior/'SCORES.json',dict(scored_unix=1041.,rows=[dict(job=j,status='complete') for j in jobs]))
    put(prior/'VERIFICATION.json',dict(status='passed',budget_met=True,expected_systems=32,verified_systems=32,
        verified_unix=1042.,elapsed_seconds=42.,rows=[dict(job=j,verified=True) for j in jobs]))
    put(prior/'SUMMARY.json',dict(expected_systems=32,technical_pass=True,adoption_pass=False,full_denominator=True))
    put(prior/'EXECUTION.json',dict(returncode=0,termination_reason=None,verification_passed=True,wall_seconds=45.))
    put(prior/'FROZEN_INPUT_ARCHIVE.json',dict(status='archived_original_bytes',file_count=1,files=sources,freeze_sha256=harness.sha(prior/'FREEZE.json')))
    put(prior/'POST_RUN_REVIEW.md',{'review':'fixture'})
    put(prior/'POST_REVIEW_ACCEPTANCE.json',dict(status='accepted_for_release_validation',sha256={},candidate_lock_sha256=harness.sha(shared/'CANDIDATE_LOCK.json')))
    def reseal():
        update(prior/'SCORES.json',all_models_sha256=harness.sha(prior/'ALL_MODELS_FROZEN.json'))
        update(prior/'VERIFICATION.json',freeze_sha256=harness.sha(prior/'FREEZE.json'),models_sha256=harness.sha(prior/'ALL_MODELS_FROZEN.json'),scores_sha256=harness.sha(prior/'SCORES.json'))
        update(prior/'POST_REVIEW_ACCEPTANCE.json',sha256={n:harness.sha(prior/n) for n in gate.RECEIPT_FILES})
    reseal()
    monkeypatch.setattr(gate.development_gate,'check',lambda s:dict(status='passed'))
    return dict(root=root,here=here,prior=prior,shared=shared,reseal=reseal)


@pytest.mark.parametrize('adoption',[False,True])
def test_predecessor_science_verdict_is_preserved_not_forced(prior_chain,adoption):
    c=prior_chain;update(c['prior']/'SUMMARY.json',adoption_pass=adoption);c['reseal']()
    result=study.check_stage_gate()
    assert result['status']=='passed' and result['prior_adoption_pass'] is adoption


@pytest.mark.parametrize('fault',['pending','extra_key','hash','technical_type','science_type','verified_count','verified_type','budget_type',
    'execution_type','termination','deadline','extra_attempt','cell_time','archive','source','parent_lock','config'])
def test_predecessor_gate_fails_closed(prior_chain,monkeypatch,fault):
    c=prior_chain;p=c['prior'];reseal=True
    if fault=='pending':(p/'POST_REVIEW_ACCEPTANCE.json').unlink();reseal=False
    elif fault=='extra_key':
        r=harness.read(p/'POST_REVIEW_ACCEPTANCE.json');r['sha256']['extra']='0'*64;put(p/'POST_REVIEW_ACCEPTANCE.json',r);reseal=False
    elif fault=='hash':update(p/'SUMMARY.json',technical_pass=False);reseal=False
    elif fault=='technical_type':update(p/'SUMMARY.json',technical_pass=1)
    elif fault=='science_type':update(p/'SUMMARY.json',adoption_pass=0)
    elif fault=='verified_count':update(p/'VERIFICATION.json',verified_systems=31)
    elif fault=='verified_type':
        r=harness.read(p/'VERIFICATION.json');r['rows'][0]['verified']=1;put(p/'VERIFICATION.json',r)
    elif fault=='budget_type':update(p/'VERIFICATION.json',budget_met=1)
    elif fault=='execution_type':update(p/'EXECUTION.json',returncode=False)
    elif fault=='termination':update(p/'EXECUTION.json',termination_reason='deadline')
    elif fault=='deadline':update(p/'VERIFICATION.json',verified_unix=3881.)
    elif fault=='extra_attempt':(p/'runs/retry').mkdir()
    elif fault=='cell_time':
        folder=next((p/'runs').iterdir());update(folder/'CELL_FROZEN.json',frozen_unix=900.)
    elif fault=='archive':put(p/'frozen_inputs/src/stub.py',{'changed':True})
    elif fault=='source':put(c['root']/'src/stub.py',{'changed':True})
    elif fault=='parent_lock':update(c['here']/'CANDIDATE_LOCK.json',parent_candidate_lock_sha256='0'*64)
    else:monkeypatch.setattr(study,'configuration',lambda:{'changed':True})
    if reseal:c['reseal']()
    with pytest.raises((ValueError,RuntimeError,OSError,KeyError,TypeError)):study.check_stage_gate()
    if fault=='pending':assert gate.status(study)['status']=='pending_missing_prior'


def test_new_cases_and_verifiers_are_bound_without_old_data():
    v=evaluate.verifier
    assert len(study.matrix())==study.EXPECTED==32 and {j['seed'] for j in study.matrix()}=={10241,10242}
    assert study.cases is not prior_study.cases and study.cases.HERE==study.HERE
    assert v.base.cases is v.legacy.cases is study.cases and v.base.HERE==v.legacy.HERE==study.HERE
    assert v.base.study is v.legacy.study is study
    assert v.base.verify_cell.__globals__['cases'] is study.cases
    assert worker.worker_run is harness.worker_run and run_bounded.bounded is harness.bounded
    assert study.supervisor.HERE==study.HERE and study.supervisor.budget is study.budget
    with pytest.raises(RuntimeError,match='seal'):study.cases.generate(study.cases.CASES[0],'test',10241)
    with pytest.raises(RuntimeError,match='No archived'):v.control_identity({}, {})


@pytest.fixture
def summary_data(tmp_path,monkeypatch):
    root=extended(tmp_path);monkeypatch.setattr(study,'HERE',root)
    v=evaluate.NewTaskVerifier(study);jobs=study.matrix()
    candidate=dict(system={},constants=[],errors=[1e-8]*2,complexity=1.,rademacher=0.,rademacher_complexity=0.)
    rows=[dict(job=j,status='complete',both_accepted=True,clean_nrmse=[1e-8]*2,observed_nrmse=[1e-8]*2,mean_clean_nrmse=1e-8,
        structure=dict(shared_graph_size=12,independent_parameters=4,expanded_total_nodes=20,maximum_output_depth=5)) for j in jobs]
    verified=[dict(job=j,verified=True,closure=dict(new_evaluations=1,valid=1,protected=1,validation_retained=1),completion={}) for j in jobs]
    models={study.name(j):dict(candidates=[deepcopy(candidate)],best=dict(errors=[1e-8]*2),validation_nrmse=[1e-8]*2,
        diagnostics=dict(sharing_closure=dict(base_candidate_count=1,base_candidate_sha256=v.legacy.digest([candidate]),
            base_evaluations=3,base_search_fingerprint='cache'))) for j in jobs}
    for name,model in models.items():put(root/'runs'/name/'model.json',model)
    return v,rows,verified,models,root


def test_summary_reports_only_real_same_batch_identity(summary_data):
    v,rows,verified,models,root=summary_data;result=v.summarize(rows,verified)
    assert result['technical_pass'] and result['adoption_pass'] and result['paired_base_identity_pass']
    assert result['protected_outputs']==32 and result['candidate_all_accepted']
    assert 'archived_control_identity' not in json.dumps(result)
    assert result['historical_control_comparison']=='not_applicable_no_same_task_archive'


@pytest.mark.parametrize('fault',['missing','duplicate','wrong_case','wrong_seconds','nan','negative','bool_error','float_size','bool_verified','negative_counter',
    'candidate_15','protection','macro','mean_graph','foundation'])
def test_new_summary_keeps_full_denominators_and_all_gates(summary_data,fault):
    v,rows,verified,models,root=summary_data
    target=next(r for r in rows if r['job']['method']=='completion')
    if fault=='missing':rows.pop()
    elif fault=='duplicate':rows[-1]=rows[0]
    elif fault=='wrong_case':rows[-1]['job']=dict(rows[-1]['job'],case='wrong')
    elif fault=='wrong_seconds':rows[-1]['job']=dict(rows[-1]['job'],seconds=61)
    elif fault=='nan':target['clean_nrmse'][0]=float('nan')
    elif fault=='negative':target['mean_clean_nrmse']=-1.
    elif fault=='bool_error':target['clean_nrmse'][0]=False
    elif fault=='float_size':target['structure']['shared_graph_size']=12.
    elif fault=='bool_verified':verified[-1]['verified']=1
    elif fault=='negative_counter':verified[-1]['closure']['valid']=-1
    elif fault=='candidate_15':
        target['both_accepted']=False
        next(r for r in rows if r['job']['method']=='base' and r['job']['case']==target['job']['case'] and r['job']['seed']==target['job']['seed'])['both_accepted']=False
    elif fault=='protection':
        target['clean_nrmse'][0]=3e-8
        for r in rows:
            if r is not target and r['job']['method']=='completion':r['clean_nrmse']=[0.,0.]
    elif fault=='macro':target['clean_nrmse']=[1.1e-8]*2
    elif fault=='mean_graph':target['structure']['shared_graph_size']=13
    else:
        model=models[study.name(target['job'])];model['diagnostics']['sharing_closure']['base_evaluations']=4
        put(root/'runs'/study.name(target['job'])/'model.json',model)
    result=v.summarize(rows,verified)
    assert not result['adoption_pass'] and result['expected_systems']==32 and result['expected_pairs']==16 and result['expected_outputs']==32
    assert 'archived_control_identity' not in json.dumps(result)
    json.dumps(result,allow_nan=False)
    if fault=='protection':assert result['macro_clean_error_nonincrease'] and result['protected_outputs']==31
    if fault=='candidate_15':assert result['technical_pass'] and result['control_acceptance_preserved']


def test_verify_never_calls_archived_control_identity(summary_data,monkeypatch):
    v,rows,verified,models,root=summary_data
    monkeypatch.setattr(v,'guards',lambda:(dict(jobs=study.matrix(),budget=dict(start_unix=evaluate.time.time())),dict(frozen_unix=1)))
    monkeypatch.setattr(study,'budget',lambda:evaluate.time.time()+2880)
    monkeypatch.setattr(v,'control_identity',lambda *a:pytest.fail('No archived control comparison for new tasks.'))
    monkeypatch.setattr(v.base,'verify_cell',lambda j,r,f:dict(job=j,verified=True))
    monkeypatch.setattr(v,'closure_evidence',lambda *a:(dict(new_evaluations=1,valid=1,protected=1,validation_retained=1),{}))
    put(root/'FREEZE.json',{});put(root/'ALL_MODELS_FROZEN.json',{})
    put(root/'SCORES.json',dict(rows=rows,scored_unix=2,all_models_sha256=harness.sha(root/'ALL_MODELS_FROZEN.json')))
    v.verify();receipt=harness.read(root/'VERIFICATION.json');summary=harness.read(root/'SUMMARY.json')
    assert receipt['verified_systems']==32 and summary['technical_pass']
    assert 'archived_control_identity' not in json.dumps(receipt)+json.dumps(summary)


def test_preflight_reads_only_fake_training_validation_and_binds_tasks(tmp_path,monkeypatch):
    root=extended(tmp_path);monkeypatch.setattr(study,'HERE',root)
    monkeypatch.setattr(study,'check_candidate_lock',lambda:None);monkeypatch.setattr(study,'check_stage_gate',lambda:None)
    monkeypatch.setattr(gate,'validate_task_sources',lambda s:None)
    for n in ('CANDIDATE_LOCK.json','PLAN.md','cases.py','DESIGN_CHECKS.json'):put(root/n,{})
    splits=[]
    def data(case,split,seed):
        assert split in ('train','validation');splits.append(split)
        n=96 if split=='train' else 48
        return np.ones((n,2)),np.arange(n*2,dtype=float).reshape(n,2)
    monkeypatch.setattr(study.cases,'training_data',data)
    study.preflight();r=harness.read(root/'PREFLIGHT.json')
    assert splits==['train','validation']*16 and r['test_labels_accessed'] is False and len(r['rows'])==16
    assert r['jobs']==study.matrix() and r['local_candidate_lock_sha256']==harness.sha(root/'CANDIDATE_LOCK.json')
    assert not (root/'ALL_MODELS_FROZEN.json').exists()


def test_freeze_keeps_old_dependencies_registries_and_new_sources(prior_chain,monkeypatch):
    c=prior_chain;sentinel=c['root']/'old_frozen_dependency.py'
    monkeypatch.setattr(gate.development_gate,'freeze_inputs',lambda s:[sentinel])
    put(c['here']/'DESIGN_CHECKS.json',dict(novelty=dict(source_freeze_sha256={'experiments/registry/FREEZE.json':'fixture'})))
    put(c['here']/'history_sources/raw.py',{})
    put(c['here']/'harness_validation/pytest.log',{})
    put(c['here']/'harness_validation/tmp/fixture.json',{})
    files=study.freeze_inputs()
    assert sentinel in files and c['root']/'experiments/registry/FREEZE.json' in files
    assert c['here']/'history_sources/raw.py' in files and c['prior']/'POST_REVIEW_ACCEPTANCE.json' in files
    assert c['here']/'harness_validation/pytest.log' in files
    assert c['here']/'harness_validation/tmp/fixture.json' not in files


def test_pending_gate_blocks_bounded_entry_before_process(prior_chain,monkeypatch):
    c=prior_chain;(c['prior']/'POST_REVIEW_ACCEPTANCE.json').unlink()
    monkeypatch.setattr(harness.subprocess,'Popen',lambda *a,**k:pytest.fail('Pending prerequisite must not start a stage process.'))
    with pytest.raises(gate.PendingPrior,match='pending_missing_prior'):run_bounded.main()
    assert not (c['here']/'supervisor.log').exists() and not (c['here']/'EXECUTION.json').exists()


@pytest.mark.parametrize('fault',[None,'early','late','missing_member','binding','test_flag'])
def test_current_cell_seals_and_preflight_binding_precede_test_labels(summary_data,monkeypatch,fault):
    v,rows,verified,models,root=summary_data
    frozen=dict(jobs=study.matrix(),candidate_lock_sha256=harness.sha(study.SHARED/'CANDIDATE_LOCK.json'))
    sealed=dict(frozen_unix=50.,files={})
    put(root/'TRAINING_STARTED.json',dict(start_unix=10.))
    for i,job in enumerate(frozen['jobs']):
        folder=root/'runs'/study.name(job)
        stamp=12.+i
        if i==31:stamp={'early':9.,'late':51.}.get(fault,stamp)
        put(folder/'CELL_FROZEN.json',dict(frozen_unix=stamp,files={'model.json':harness.sha(folder/'model.json')}))
        for n in ('CELL_FROZEN.json','model.json'):sealed['files'][(folder/n).relative_to(root).as_posix()]=harness.sha(folder/n)
    if fault=='missing_member':sealed['files'].pop((folder/'CELL_FROZEN.json').relative_to(root).as_posix())
    put(root/'PREFLIGHT.json',dict(test_labels_accessed=0 if fault=='test_flag' else False,fixture='wrong' if fault=='binding' else 'bound'))
    monkeypatch.setattr(v.legacy,'_base_guards',lambda:(frozen,sealed))
    monkeypatch.setattr(study,'check_candidate_lock',lambda:None);monkeypatch.setattr(study,'check_stage_gate',lambda:None)
    monkeypatch.setattr(study,'task_binding',lambda:dict(fixture='bound'))
    monkeypatch.setattr(study.cases,'generate',lambda *a:pytest.fail('No test labels before valid current seals.'))
    if fault:
        with pytest.raises(ValueError):v.evaluate()
    else:assert v.guards()==(frozen,sealed)


def test_local_long_archive_io_is_idempotent_and_keeps_logical_keys(tmp_path):
    io=study.archive_io
    source=tmp_path/('a'*110)/('b'*110)/'original.bin'
    io.disk_path(source).parent.mkdir(parents=True,exist_ok=True);io.disk_path(source).write_bytes(b'original fixture bytes')
    expected=io.sha(source)
    assert io.disk_path(io.disk_path(source))==io.disk_path(source)
    assert io.sha(io.disk_path(source))==expected and len(str(source))>260
    logical=source.relative_to(tmp_path).as_posix();io.verify_files({logical:expected},tmp_path)
    destination=tmp_path/'archive'/logical
    io.copy_original(source,destination,expected);io.copy_original(source,destination,expected)
    assert io.sha(destination)==expected and not logical.startswith('//?/')
    with pytest.raises(ValueError):io.verify_files({'../outside.bin':expected},tmp_path)
    io.disk_path(destination).write_bytes(b'changed')
    with pytest.raises(RuntimeError,match='drift'):io.copy_original(source,destination,expected)


def test_freeze_adapts_long_io_without_patching_shared_harness(tmp_path,monkeypatch):
    root=tmp_path;here=root/'current';prior=root/'prior';shared=root/'shared';io=study.archive_io
    for n,v in (('ROOT',root),('HERE',here),('PRIOR',prior),('SHARED',shared)):monkeypatch.setattr(study,n,v)
    long_input=root/('input'*24)/('piece'*24)/'source.bin'
    io.disk_path(long_input).parent.mkdir(parents=True,exist_ok=True);io.disk_path(long_input).write_bytes(b'long archived dependency')
    logical=long_input.relative_to(root).as_posix();expected=io.sha(long_input)
    io.copy_original(long_input,prior/'frozen_inputs'/logical,expected)
    put(prior/'FREEZE.json',dict(files={logical:expected}))
    put(prior/'runs/stub.json',{})
    put(prior/'ALL_MODELS_FROZEN.json',dict(files={'runs/stub.json':io.sha(prior/'runs/stub.json')}))
    put(root/'src/stub.py',{});put(shared/'CANDIDATE_LOCK.json',{})
    put(here/'PREFLIGHT.json',dict(status='passed',test_labels_accessed=False,fixture='bound'))
    monkeypatch.setattr(study,'check_candidate_lock',lambda:None);monkeypatch.setattr(study,'check_stage_gate',lambda:None)
    monkeypatch.setattr(gate,'validate_task_sources',lambda s:None);monkeypatch.setattr(study,'task_binding',lambda:dict(fixture='bound'))
    monkeypatch.setattr(study,'freeze_inputs',lambda:[long_input,prior/'frozen_inputs'/logical,here/'PREFLIGHT.json'])
    monkeypatch.setattr(study,'configuration',lambda:{})
    monkeypatch.setattr(study.cases,'CASES',['fixture']);monkeypatch.setattr(study.cases,'seeds',lambda c:[1]);monkeypatch.setattr(study.cases,'design',lambda c:{})
    monkeypatch.setattr(study.cases,'training_data',lambda *a:(np.ones((2,2)),np.ones((2,2))))
    monkeypatch.setattr(harness.subprocess,'check_output',lambda *a,**k:b'fixture')
    shared_sha=harness.sha;shared_copy=harness.shutil.copyfile
    study.freeze();frozen=harness.read(here/'FREEZE.json')
    assert logical in frozen['files'] and all(not n.startswith('//?/') for n in frozen['files'])
    assert frozen['files'][logical]==expected and (here/'source/src/stub.py').read_bytes()==(root/'src/stub.py').read_bytes()
    study.verify_files(frozen['files'])
    assert harness.sha is shared_sha and harness.shutil.copyfile is shared_copy


@pytest.mark.parametrize('corruption',[None,'archive','registry'])
def test_historical_audit_binds_archived_version_not_mutable_live_plan(tmp_path,monkeypatch,corruption):
    root=extended(tmp_path);here=root/'release';monkeypatch.setattr(study,'ROOT',root);monkeypatch.setattr(study,'HERE',here)
    key='experiments/unfrozen_development/PLAN.md'
    put(here/'history_sources'/key,{'historical_snapshot':True})
    put(root/key,{'current_plan_after_engineering_review':True})
    put(root/'registry/FREEZE.json',{'frozen_registry':True});put(here/'cases.py',{'case_source':True})
    put(here/'NOVELTY_AUDIT.json',dict(source_files={key:study.archive_io.sha(here/'history_sources'/key)}))
    put(here/'DESIGN_CHECKS.json',dict(input_sha256={'cases.py':study.archive_io.sha(here/'cases.py')},
        novelty=dict(source_freeze_sha256={'registry/FREEZE.json':study.archive_io.sha(root/'registry/FREEZE.json')})))
    if corruption=='archive':put(here/'history_sources'/key,{'changed':True})
    elif corruption=='registry':put(root/'registry/FREEZE.json',{'changed':True})
    if corruption:
        with pytest.raises(ValueError):gate.validate_task_sources(study)
    else:gate.validate_task_sources(study)
