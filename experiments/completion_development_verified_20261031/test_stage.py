"""Non-training fixtures for the continuation chain and its immutable predecessors."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import types

import pytest

from experiments.completion_development_verified_20261031 import gate,study,worker,evaluate,run_bounded
from experiments.compositional_completion_20261031 import harness
from experiments.completion_development_20261031 import study as prepared
from experiments.completion_verification_repair_20261031.verify_sealed import SealedVerifier


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


def update(path,**values):
    obj=harness.read(path);obj.update(values);put(path,obj)


def copy_inputs(root,folder,files):
    for name in files:
        dest=folder/'frozen_inputs'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(root/name,dest)


@pytest.fixture
def chain(tmp_path,monkeypatch):
    # Nested archive fixtures exceed Win32 MAX_PATH under the owned basetemp.
    root=Path('\\\\?\\'+str(tmp_path.resolve())) if os.name=='nt' else tmp_path
    here=root/'experiments/completion_development_verified_20261031'
    old=here.parent/'completion_historical_20261031';fixed=here.parent/'completion_verification_repair_20261031'
    original_development=here.parent/'completion_development_20261031';shared=here.parent/'compositional_completion_20261031'
    for name,value in [('ROOT',root),('HERE',here),('SHARED',shared)]:monkeypatch.setattr(study,name,value)
    config={'fixed_configuration':True}
    monkeypatch.setattr(study,'configuration',lambda:deepcopy(config))
    put(root/'src/stub.py',{})
    put(original_development/'PLAN.md',{'original':'development'})
    put(old/'PLAN.md',{'original':'historical'})
    put(shared/'CANDIDATE_LOCK.json',dict(expected_source_count=1,source_files={'src/stub.py':harness.sha(root/'src/stub.py')},
        shared_configuration=config,stage_plan_sha256={'completion_historical_20261031':harness.sha(old/'PLAN.md'),
            'completion_development_20261031':harness.sha(original_development/'PLAN.md')}))
    budget=dict(start_unix=1000.,limit_seconds=2880)
    put(old/'BUDGET.json',budget)
    inputs=[root/'src/stub.py',shared/'CANDIDATE_LOCK.json',old/'PLAN.md',original_development/'PLAN.md',old/'BUDGET.json']
    files={p.relative_to(root).as_posix():harness.sha(p) for p in inputs}
    jobs=[dict(case='fixture'+str(i//2),seed=1,method=('base','completion')[i%2],seconds=60) for i in range(34)]
    put(old/'FREEZE.json',dict(frozen_unix=1001.,files=files,jobs=jobs,config=config,budget=budget,candidate_lock_sha256=harness.sha(shared/'CANDIDATE_LOCK.json')))
    copy_inputs(root,old,files)
    put(old/'runs/model.json',{'fixture_model':True})
    run_files={'runs/model.json':harness.sha(old/'runs/model.json')}
    put(old/'ALL_MODELS_FROZEN.json',dict(files=run_files,freeze_sha256=harness.sha(old/'FREEZE.json'),frozen_unix=1100.))
    put(old/'SCORES.json',dict(all_models_sha256=harness.sha(old/'ALL_MODELS_FROZEN.json'),scored_unix=1110.))
    put(old/'VERIFICATION.json',dict(status='failed',verified_unix=1120.,freeze_sha256=harness.sha(old/'FREEZE.json'),
        models_sha256=harness.sha(old/'ALL_MODELS_FROZEN.json'),scores_sha256=harness.sha(old/'SCORES.json')))
    put(old/'SUMMARY.json',dict(technical_pass=False,adoption_pass=False))
    put(old/'EXECUTION.json',dict(verification_passed=False))
    put(old/'FROZEN_INPUT_ARCHIVE.json',dict(status='archived_original_bytes',freeze_sha256=harness.sha(old/'FREEZE.json'),
        files=files,file_count=len(files),runs_unchanged=True,sealed_run_file_count=len(run_files)))
    put(old/'POST_RUN_REVIEW.md',{'review':'original failure preserved'})
    put(old/'post_review_checks.json',{})
    put(old/'POST_REVIEW_RECEIVED.json',dict(status='received_consistent_with_original_failure',original_technical_pass=False,
        original_adoption_pass=False,development_authorized=False,sha256={n:harness.sha(old/n) for n in ('POST_RUN_REVIEW.md','post_review_checks.json')}))
    put(fixed/'verify_sealed.py',{'fixture_code':True});put(fixed/'PLAN.md',{'fixture':'sealed correction'})
    corrected_inputs=[old/n for n in gate.ARTIFACTS]+[fixed/'verify_sealed.py',fixed/'PLAN.md']
    corrected_files={p.relative_to(root).as_posix():harness.sha(p) for p in corrected_inputs}
    put(fixed/'FREEZE.json',dict(frozen_unix=1200.,deadline_unix=1800.,verification_cap_seconds=600,original_deadline_unix=3880.,
        original_sha256={n:harness.sha(old/n) for n in gate.ARTIFACTS},files=corrected_files,no_fit_search_or_rescoring=True,original_verdict='failed'))
    put(fixed/'VERIFICATION.json',dict(status='passed',expected_systems=34,verified_systems=34,rows=[dict(job=j,verified=True) for j in jobs],
        budget_met=True,inputs_unchanged=True,verified_unix=1250.,original_verdict='failed'))
    put(fixed/'SUMMARY.json',dict(expected_systems=34,expected_pairs=17,expected_outputs=34,protected_outputs=34,full_denominator=True,
        technical_pass=True,adoption_pass=True,original_verdict='failed',verification_of_sealed_results=True))
    put(fixed/'EXECUTION.json',dict(returncode=0,termination_reason=None,wall_seconds=50.,verification_passed=True,no_training_performed=True))
    put(fixed/'POST_RUN_REVIEW.md',{'independent':'fixture correction review'})
    put(fixed/'POST_REVIEW_ACCEPTANCE.json',dict(status='accepted_for_sealed_historical_correction',accepted_unix=1260.,sha256={}))

    def reseal():
        f=harness.read(fixed/'FREEZE.json')
        f['original_sha256']={n:harness.sha(old/n) for n in gate.ARTIFACTS}
        f['files']={n:harness.sha(root/n) for n in f['files']}
        put(fixed/'FREEZE.json',f)
        update(fixed/'VERIFICATION.json',freeze_sha256=harness.sha(fixed/'FREEZE.json'),original_sha256=f['original_sha256'])
        copy_inputs(root,fixed,f['files'])
        put(fixed/'FROZEN_INPUT_ARCHIVE.json',dict(status='archived_original_bytes',file_count=len(f['files']),files=f['files'],
            freeze_sha256=harness.sha(fixed/'FREEZE.json'),original_historical_inputs_and_runs_unchanged=True))
        receipt=harness.read(fixed/'POST_REVIEW_ACCEPTANCE.json')
        receipt['sha256']={n:harness.sha(fixed/n) for n in gate.RECEIPT_FILES if n!='CANDIDATE_LOCK.json'}
        receipt['sha256']['CANDIDATE_LOCK.json']=harness.sha(shared/'CANDIDATE_LOCK.json')
        put(fixed/'POST_REVIEW_ACCEPTANCE.json',receipt)
    reseal()
    return types.SimpleNamespace(root=root,here=here,old=old,fixed=fixed,prepared=original_development,shared=shared,reseal=reseal)


def test_valid_new_chain_passes_while_original_failure_stays(chain):
    result=study.check_stage_gate()
    assert result['original_inputs']==5 and result['original_runtime_files']==1 and result['correction_inputs']==12
    assert harness.read(chain.old/'SUMMARY.json')['technical_pass'] is False
    assert not (chain.old/'POST_REVIEW_ACCEPTANCE.json').exists()


@pytest.mark.parametrize('fault',['missing_acceptance','bad_acceptance','missing_key','extra_key','sha_drift','original_technical',
    'original_adoption_type','original_verdict','original_execution','original_received','verify_budget_type','verify_input_type',
    'verify_status','verify_count_float','verify_count_short','verify_row_short','verify_row_type','verify_row_wrong',
    'summary_technical_type','summary_adoption_type','summary_full_type','summary_sealed_type','summary_original',
    'summary_outputs','execution_code_bool','execution_training_type','execution_training_false','execution_termination',
    'execution_wall_negative','execution_wall_inf','execution_wall_over','freeze_training_type','freeze_cap_float',
    'deadline_original','deadline_own','verified_time_early','verified_time_late','verified_time_type',
    'old_development_started','source_drift','plan_drift','config_drift','correction_input','correction_archive'])
def test_invalid_or_unaccepted_chain_fails_closed(chain,monkeypatch,fault):
    c=chain;v=c.fixed/'VERIFICATION.json';summary=c.fixed/'SUMMARY.json';execution=c.fixed/'EXECUTION.json';freeze=c.fixed/'FREEZE.json'
    reseal=True
    if fault=='missing_acceptance':(c.fixed/'POST_REVIEW_ACCEPTANCE.json').unlink();reseal=False
    elif fault=='bad_acceptance':update(c.fixed/'POST_REVIEW_ACCEPTANCE.json',status='accepted');reseal=False
    elif fault in ('missing_key','extra_key'):
        obj=harness.read(c.fixed/'POST_REVIEW_ACCEPTANCE.json')
        if fault=='missing_key':obj['sha256'].pop('SUMMARY.json')
        else:obj['sha256']['EXTRA.json']='0'*64
        put(c.fixed/'POST_REVIEW_ACCEPTANCE.json',obj);reseal=False
    elif fault=='sha_drift':update(summary,technical_pass=False);reseal=False
    elif fault=='original_technical':update(c.old/'SUMMARY.json',technical_pass=True)
    elif fault=='original_adoption_type':update(c.old/'SUMMARY.json',adoption_pass=0)
    elif fault=='original_verdict':update(c.old/'VERIFICATION.json',status='passed')
    elif fault=='original_execution':update(c.old/'EXECUTION.json',verification_passed=True)
    elif fault=='original_received':update(c.old/'POST_REVIEW_RECEIVED.json',development_authorized=True)
    elif fault=='verify_budget_type':update(v,budget_met=1)
    elif fault=='verify_input_type':update(v,inputs_unchanged=1)
    elif fault=='verify_status':update(v,status='failed')
    elif fault=='verify_count_float':update(v,verified_systems=34.)
    elif fault=='verify_count_short':update(v,verified_systems=33)
    elif fault in ('verify_row_short','verify_row_type','verify_row_wrong'):
        obj=harness.read(v)
        if fault=='verify_row_short':obj['rows'].pop()
        elif fault=='verify_row_type':obj['rows'][0]['verified']=1
        else:obj['rows'][0]['job']['case']='different'
        put(v,obj)
    elif fault=='summary_technical_type':update(summary,technical_pass=1)
    elif fault=='summary_adoption_type':update(summary,adoption_pass=1)
    elif fault=='summary_full_type':update(summary,full_denominator=1)
    elif fault=='summary_sealed_type':update(summary,verification_of_sealed_results=1)
    elif fault=='summary_original':update(summary,original_verdict='passed')
    elif fault=='summary_outputs':update(summary,protected_outputs=33)
    elif fault=='execution_code_bool':update(execution,returncode=False)
    elif fault=='execution_training_type':update(execution,no_training_performed=1)
    elif fault=='execution_training_false':update(execution,no_training_performed=False)
    elif fault=='execution_termination':update(execution,termination_reason='deadline')
    elif fault=='execution_wall_negative':update(execution,wall_seconds=-1.)
    elif fault=='execution_wall_inf':update(execution,wall_seconds=float('inf'))
    elif fault=='execution_wall_over':update(execution,wall_seconds=601.)
    elif fault=='freeze_training_type':update(freeze,no_fit_search_or_rescoring=1)
    elif fault=='freeze_cap_float':update(freeze,verification_cap_seconds=600.)
    elif fault=='deadline_original':update(freeze,deadline_unix=3881.)
    elif fault=='deadline_own':update(freeze,deadline_unix=1801.)
    elif fault=='verified_time_early':update(v,verified_unix=1199.)
    elif fault=='verified_time_late':update(v,verified_unix=1801.)
    elif fault=='verified_time_type':update(v,verified_unix=True)
    elif fault=='old_development_started':put(c.prepared/'BUDGET.json',{})
    elif fault=='source_drift':put(c.root/'src/stub.py',{'changed':True})
    elif fault=='plan_drift':put(c.prepared/'PLAN.md',{'changed':True})
    elif fault=='config_drift':monkeypatch.setattr(study,'configuration',lambda:{'different':True})
    elif fault=='correction_input':put(c.fixed/'verify_sealed.py',{'changed':True});reseal=False
    elif fault=='correction_archive':
        p=next((c.fixed/'frozen_inputs').rglob('verify_sealed.py'));put(p,{'changed':True});reseal=False
    if reseal:c.reseal()
    with pytest.raises((ValueError,OSError,KeyError,TypeError)):
        study.check_stage_gate()


def test_new_thin_entries_and_isolated_bindings():
    assert study.EXPECTED==32 and study.LIMIT==2880 and study.matrix()==prepared.matrix()
    assert study.configuration()==prepared.configuration()
    assert study.cases is not prepared.cases and study.cases.HERE==study.HERE and prepared.cases.HERE==prepared.HERE
    assert study.supervisor.HERE==study.HERE and study.supervisor.budget is study.budget
    assert isinstance(evaluate.verifier,SealedVerifier)
    assert evaluate.verifier.legacy.HERE==evaluate.verifier.base.HERE==study.HERE
    assert evaluate.verifier.legacy.cases is evaluate.verifier.base.cases is study.cases
    assert evaluate.verifier.legacy.study is evaluate.verifier.base.study is study
    assert evaluate.verifier.base.evaluate.__globals__['guards'].__self__ is evaluate.verifier
    assert worker.worker_run is harness.worker_run and run_bounded.bounded is harness.bounded
    with pytest.raises(RuntimeError,match='seal'):
        study.cases.generate(study.cases.CASES[0],'test',9131)


def test_original_stage_remains_failed_and_cannot_start():
    original=prepared.HERE.parent/'completion_historical_20261031'
    assert harness.read(original/'SUMMARY.json')['technical_pass'] is False
    assert harness.read(original/'SUMMARY.json')['adoption_pass'] is False
    assert not (original/'POST_REVIEW_ACCEPTANCE.json').exists()
    with pytest.raises((ValueError,OSError)):
        prepared.check_stage_gate()


def test_freeze_retains_all_inherited_paths_and_replaces_only_obsolete_receipt(chain,monkeypatch):
    a=chain.root/'inherited_a.py';b=chain.root/'inherited_failure.log';obsolete=chain.old/'POST_REVIEW_ACCEPTANCE.json'
    monkeypatch.setattr(harness,'freeze_inputs',lambda s:[a,b,obsolete])
    paths=study.freeze_inputs()
    assert a in paths and b in paths and obsolete not in paths
    for p in (chain.fixed/'verify_sealed.py',chain.fixed/'FREEZE.json',chain.fixed/'POST_REVIEW_ACCEPTANCE.json',
        chain.fixed/'FROZEN_INPUT_ARCHIVE.json',chain.old/'SUMMARY.json',chain.old/'VERIFICATION.json',
        chain.old/'runs/model.json',chain.old/'frozen_inputs/src/stub.py',chain.fixed/'frozen_inputs/experiments/completion_historical_20261031/SUMMARY.json'):
        assert p in paths


def test_worker_gate_precedes_all_data_and_fit(chain,monkeypatch):
    (chain.fixed/'POST_REVIEW_ACCEPTANCE.json').unlink()
    monkeypatch.setattr(study.cases,'training_data',lambda *a:pytest.fail('No data before accepted gate.'))
    with pytest.raises(OSError):worker.run(chain.here,{'method':'base'},study.configuration())
    assert not (chain.here/'result.json').exists()


def test_new_scratch_fixtures_stay_local_while_prior_inputs_and_logs_are_frozen(chain,monkeypatch):
    obsolete=chain.old/'POST_REVIEW_ACCEPTANCE.json'
    local_log=chain.here/'harness_validation/pytest_03.log'
    local_script=chain.here/'harness_review_validation/probe_paths.py'
    scratch=chain.here/'harness_validation/p2/case/fixture.json'
    review_scratch=chain.here/'harness_review_validation/pytest_temp/case/fixture.json'
    old_scratch=chain.old/'harness_validation/pytest_temp/case/fixture.json'
    monkeypatch.setattr(harness,'freeze_inputs',lambda s:[obsolete,local_log,local_script,scratch,review_scratch,old_scratch])
    paths=study.freeze_inputs()
    assert local_log in paths and local_script in paths and old_scratch in paths
    assert scratch not in paths and review_scratch not in paths and obsolete not in paths
