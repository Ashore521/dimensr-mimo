from dataclasses import asdict
import inspect
import json

import numpy as np
import pytest

from dimensr_mimo import SearchConfig
from experiments.new_task_validation_20261030 import study, worker, evaluate, run_bounded
from experiments.precision_repair_20261028 import evaluate as historical


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


def test_worker_only_changes_flag_and_cannot_read_test(monkeypatch,tmp_path):
    captured,splits = [],[]
    class StopBeforeFit:
        def __init__(self,cfg,*args):
            captured.append(asdict(cfg))
        def fit(self,*args,**kwargs):
            raise RuntimeError('Construction fixture; no fitting.')
    monkeypatch.setattr(worker,'DimenSRMIMORegressor',StopBeforeFit)
    monkeypatch.setattr(study,'check_candidate_lock',lambda:None)
    def data(case,split,seed):
        splits.append(split)
        assert split in ('train','validation')
        return np.ones((4,2)),np.ones((4,2))
    monkeypatch.setattr(worker.cases,'training_data',data)
    for method in study.METHODS:
        path=tmp_path/method;path.mkdir()
        worker.run(path,dict(case=study.cases.CASES[0],seed=9131,method=method),study.configuration())
    assert [row.pop('precision_repair') for row in captured]==[False,True]
    assert captured[0]==captured[1]
    assert captured[0]['initialization']=='compound' and captured[0]['sharing_strategy']=='closure'
    assert captured[0]['wall_seconds']==52 and captured[0]['max_evaluations']==1200
    assert splits==['train','validation']*2
    assert 'truth_graph' not in inspect.getsource(worker.run)
    with pytest.raises(ValueError):
        worker.run(tmp_path,dict(method='other'),{})
    jobs=study.matrix()
    assert len(jobs)==len({study.name(j) for j in jobs})==32 and study.LIMIT==2880
    assert len({(j['case'],j['seed']) for j in jobs})==16
    assert all(j['seconds']==60 for j in jobs)
    for case in study.cases.CASES:
        SearchConfig(**dict(study.configuration(),final_selection='precision',acceptance=study.cases.design(case)['acceptance'])).validate()


def test_reused_verifiers_have_isolated_directory_and_case_bindings():
    assert evaluate.legacy is not historical and evaluate.base is not historical.base
    assert evaluate.legacy.HERE==evaluate.base.HERE==study.HERE
    assert evaluate.legacy.cases is evaluate.base.cases is study.cases
    assert evaluate.legacy.study is evaluate.base.study is study
    assert historical.HERE!=study.HERE and historical.base.cases is not study.cases
    assert evaluate.evaluate.__globals__['guards'] is evaluate.guards
    assert evaluate.legacy._base_guards.__globals__['cases'] is study.cases
    assert evaluate.closure_evidence.__globals__['cases'] is study.cases
    assert evaluate.base.verify_cell.__globals__['cases'] is study.cases
    assert study.supervisor.HERE==study.HERE and study.supervisor.ROOT==study.ROOT
    assert study.supervisor.budget is study.budget


def test_frozen_configuration_cannot_drift(monkeypatch,tmp_path):
    monkeypatch.setattr(study,'PRIOR',tmp_path)
    save(tmp_path/'FREEZE.json',dict(config={'fixture':'original'}))
    save(tmp_path/'ALL_MODELS_FROZEN.json',dict(freeze_sha256=study.sha(tmp_path/'FREEZE.json')))
    assert study.configuration()=={'fixture':'original'}
    save(tmp_path/'FREEZE.json',dict(config={'fixture':'changed'}))
    with pytest.raises(RuntimeError,match='configuration drift'):
        study.configuration()


def test_entry_and_worker_source_drift_block_before_execution(monkeypatch,tmp_path):
    def reject():
        raise RuntimeError('locked source drift')
    monkeypatch.setattr(study,'check_candidate_lock',reject)
    monkeypatch.setattr(run_bounded,'HERE',tmp_path)
    monkeypatch.setattr(worker.cases,'training_data',lambda *a:pytest.fail('No arrays with source drift.'))
    with pytest.raises(RuntimeError,match='source drift'):
        run_bounded.main()
    with pytest.raises(RuntimeError,match='source drift'):
        worker.run(tmp_path,dict(method='repair'),{})
    assert not list(tmp_path.iterdir())


@pytest.fixture
def summary_inputs(monkeypatch):
    jobs=study.matrix()
    scores=[dict(job=j,status='complete',both_accepted=True,mean_clean_nrmse=1e-8,clean_nrmse=[1e-8]*2,
        observed_nrmse=[1e-8]*2,structure=dict(shared_graph_size=12,independent_parameters=2,
            expanded_total_nodes=20,maximum_output_depth=5)) for j in jobs]
    verified=[dict(job=j,verified=True,closure=dict(new_evaluations=1,valid=1,protected=1,validation_retained=1),
        repair=dict(triggered=0,new_evaluations=0,valid=0,protected=0,validation_retained=0,
            linear_subproblems=0,retained_roots=0,implicit_new_systems=0) if j['method']=='repair' else {}) for j in jobs]
    models={study.name(j):dict(candidates=[{'fixture':'candidate'}],best={'fixture':'best'},validation_nrmse=[1e-8]*2,
        search_fingerprint='cache',diagnostics=dict(sharing_closure=dict(base_candidate_count=1,
            base_candidate_sha256='base',base_evaluations=3,base_search_fingerprint='base-cache',
            **(dict(precision_repair=dict(triggered=False)) if j['method']=='repair' else {})))) for j in jobs}
    monkeypatch.setattr(evaluate,'read',lambda path:models[path.parent.name])
    return scores,verified,models


def test_zero_trigger_is_technical_success_without_discovery_claim(summary_inputs):
    scores,verified,_=summary_inputs
    result=evaluate.summarize(scores,verified)
    assert result['technical_pass'] and result['adoption_pass']
    assert result['expected_systems']==32 and result['expected_pairs']==16 and result['protected_outputs']==32
    assert result['candidate_all_accepted']
    evidence=result['repair_discovery_evidence']
    assert evidence['triggered']==0 and not evidence['actual_implicit_evaluation']
    assert not evidence['validation_retained_discovery'] and not evidence['acceptance_gain_observed']
    assert set(result['methods']['repair']['strata'])==set(study.cases.FAMILIES)|{'noise0','noise1'}


@pytest.mark.parametrize('kind',['missing_score','duplicate_score','missing_verification','duplicate_verification','timeout','nonfinite','missing_metric','nonfinite_mean','negative_error','negative_mean'])
def test_full_matrix_and_invalid_metrics_fail_closed(summary_inputs,kind):
    scores,verified,_=summary_inputs
    if kind=='missing_score':scores.pop()
    elif kind=='duplicate_score':scores[-1]=scores[0]
    elif kind=='missing_verification':verified.pop()
    elif kind=='duplicate_verification':verified[-1]=verified[0]
    elif kind=='timeout':scores[-1]={'job':scores[-1]['job'],'status':'system_timeout'}
    elif kind=='nonfinite':scores[-1]['clean_nrmse'][0]=float('nan')
    elif kind=='missing_metric':scores[-1].pop('observed_nrmse')
    elif kind=='negative_error':scores[-1]['observed_nrmse'][0]=-1.
    elif kind=='negative_mean':scores[-1]['mean_clean_nrmse']=-1.
    else:scores[-1]['mean_clean_nrmse']=float('inf')
    result=evaluate.summarize(scores,verified)
    assert not result['technical_pass'] and not result['adoption_pass']
    assert result['expected_systems']==32 and result['expected_outputs']==32 and len(result['pairs'])==16
    assert result['methods']['base']['expected']==result['methods']['repair']['expected']==16
    if kind not in ('missing_verification','duplicate_verification'):
        assert any(m['mean_clean_nrmse'] is None for m in result['methods'].values())
    json.dumps(result,allow_nan=False)


@pytest.mark.parametrize('gate',['candidate_all','protection','retention','macro_error','mean_graph','identity','skip_pool','skip_choice'])
def test_scientific_and_identity_gates_cannot_be_offset_by_means(summary_inputs,gate):
    scores,verified,models=summary_inputs
    target=next(r for r in scores if r['job']['method']=='repair')
    key=study.name(target['job'])
    if gate=='candidate_all':
        target['both_accepted']=False
        next(r for r in scores if r['job']['method']=='base' and r['job']['case']==target['job']['case'] and r['job']['seed']==target['job']['seed'])['both_accepted']=False
    elif gate=='protection':
        target['clean_nrmse'][0]=3e-8
        for r in scores:
            if r is not target and r['job']['method']=='repair':r['clean_nrmse']=[0.,0.]
    elif gate=='retention':target['both_accepted']=False
    elif gate=='macro_error':target['clean_nrmse']=[1.1e-8]*2
    elif gate=='mean_graph':target['structure']['shared_graph_size']=13
    elif gate=='identity':models[key]['diagnostics']['sharing_closure']['base_evaluations']=4
    elif gate=='skip_pool':models[key]['search_fingerprint']='changed'
    else:models[key]['best']={'fixture':'different'}
    result=evaluate.summarize(scores,verified)
    assert not result['adoption_pass']
    if gate=='candidate_all':
        assert not result['candidate_all_accepted'] and result['control_acceptance_preserved']
    if gate=='macro_error':
        assert result['protected_outputs']==32 and not result['macro_clean_error_nonincrease']
    if gate=='mean_graph':assert not result['mean_graph_nonincrease']
    if gate=='protection':
        assert result['macro_clean_error_nonincrease'] and result['mean_graph_nonincrease']
        assert result['protected_outputs']==31


@pytest.mark.parametrize('method',['base','repair'])
@pytest.mark.parametrize('field',['new_evaluations','valid','protected','validation_retained'])
def test_matching_actual_evidence_required_in_each_arm(summary_inputs,method,field):
    scores,verified,_=summary_inputs
    for row in verified:
        if row['job']['method']==method:row['closure'][field]=0
    assert not evaluate.summarize(scores,verified)['technical_pass']


def test_trigger_without_retention_is_reported_without_technical_failure(summary_inputs):
    scores,verified,models=summary_inputs
    row=next(r for r in verified if r['job']['method']=='repair')
    row['repair'].update(triggered=1,new_evaluations=2,valid=1,linear_subproblems=5,retained_roots=1,implicit_new_systems=2)
    models[study.name(row['job'])]['diagnostics']['sharing_closure']['precision_repair']['triggered']=True
    result=evaluate.summarize(scores,verified)
    assert result['technical_pass'] and result['adoption_pass']
    evidence=result['repair_discovery_evidence']
    assert evidence['new_evaluations']==2 and evidence['valid']==1 and evidence['protected']==0
    assert evidence['actual_implicit_evaluation'] and not evidence['validation_retained_discovery']
    assert not evidence['acceptance_gain_observed']
    assert evidence['by_family'][study.cases.FAMILIES[0]]['triggered']==1
    assert evidence['families_with_validation_retained']==0


@pytest.mark.parametrize('fault',['changed','missing','extra'])
def test_candidate_source_lock_blocks_drift(monkeypatch,tmp_path,fault):
    here=tmp_path/'experiments/new';prior=tmp_path/'experiments/old'
    monkeypatch.setattr(study,'ROOT',tmp_path);monkeypatch.setattr(study,'HERE',here);monkeypatch.setattr(study,'PRIOR',prior)
    source={}
    for i in range(40):
        path=tmp_path/f'src/module{i}.py';path.parent.mkdir(exist_ok=True);path.write_text('# locked')
        source[path.relative_to(tmp_path).as_posix()]=study.sha(path)
    save(prior/'SUMMARY.json',{})
    save(here/'CANDIDATE_LOCK.json',dict(source_files=source,source_summary_sha256=study.sha(prior/'SUMMARY.json')))
    study.check_candidate_lock()
    if fault=='changed':path.write_text('# changed')
    elif fault=='missing':path.unlink()
    else:(path.parent/'extra.py').write_text('# extra')
    with pytest.raises(RuntimeError,match='drift'):
        study.check_candidate_lock()
    with pytest.raises(RuntimeError,match='drift'):
        study.freeze()
    assert not (here/'BUDGET.json').exists()


def test_train_checks_lock_before_every_cell(monkeypatch,tmp_path):
    checks=[];calls=[]
    def check():
        checks.append(True)
        if calls:raise RuntimeError('source drift after first cell')
    monkeypatch.setattr(study,'HERE',tmp_path)
    monkeypatch.setattr(study,'check_candidate_lock',check)
    monkeypatch.setattr(study,'verify_files',lambda *a:None)
    monkeypatch.setattr(study,'budget',lambda:study.time.time()+2880)
    jobs=study.matrix()[:2]
    save(tmp_path/'FREEZE.json',dict(files={},jobs=jobs,config={}))
    def run_job(job,*args):
        calls.append(job)
        save(tmp_path/'runs'/study.name(job)/'resources.json',dict(status='fixture'))
    monkeypatch.setattr(study.supervisor,'run_job',run_job)
    with pytest.raises(RuntimeError,match='source drift'):
        study.train()
    assert len(calls)==1 and len(checks)==3
    assert (tmp_path/'runs'/study.name(jobs[0])/'CELL_FROZEN.json').exists()
    assert not (tmp_path/'ALL_MODELS_FROZEN.json').exists()


def test_freeze_input_closure_and_source_copies(monkeypatch,tmp_path):
    here=tmp_path/'experiments/new';prior=tmp_path/'experiments/old'
    monkeypatch.setattr(study,'HERE',here);monkeypatch.setattr(study,'ROOT',tmp_path);monkeypatch.setattr(study,'PRIOR',prior)
    monkeypatch.setattr(study,'check_candidate_lock',lambda:None)
    deps=['experiments/precision_delivery_20261026/selector.py','experiments/guarded_selection_20261026/selector.py',
        'experiments/phase_budget_20261027/source/src/dimensr_mimo/grammar.py',
        'experiments/native_api_20261026/runs/dimensional_frequency_normalized/model.json']
    for dep in deps:save(tmp_path/dep,{})
    save(prior/'FREEZE.json',dict(files={n:'previous' for n in deps},config={'fixture':'locked'}))
    save(prior/'ALL_MODELS_FROZEN.json',dict(files={'runs/previous/model.json':'previous'},freeze_sha256=study.sha(prior/'FREEZE.json')))
    save(prior/'runs/previous/model.json',{})
    for filename in ('SUMMARY.json','SCORES.json','VERIFICATION.json','POST_RUN_REVIEW.md','HARNESS_REVIEW.md',
                     'FROZEN_INPUT_ARCHIVE.json','post_review_checks.json'):
        save(prior/filename,{})
    for filename in ('PLAN.md','PROTOCOL.md','PRE_RUN_REVIEW.md','TEST_VALIDATION.json','CANDIDATE_LOCK.json',
                     'NOVELTY_AUDIT.json','ORACLE_AUDIT.json','DESIGN_AUDIT.json','validation/own.log',
                     'history_sources/original.py','extra_review.md'):
        save(here/filename,{})
    save(here/'PREFLIGHT.json',dict(status='passed'))
    save(tmp_path/'src/stub.py',{})
    save(tmp_path/'tests/test_fixture.py',{})
    save(tmp_path/'pyproject.toml',{})
    monkeypatch.setattr(study.cases,'CASES',['fixture'])
    monkeypatch.setattr(study.cases,'seeds',lambda c:[1])
    monkeypatch.setattr(study.cases,'design',lambda c:{})
    monkeypatch.setattr(study.cases,'training_data',lambda *a:(np.ones((2,2)),np.ones((2,2))))
    monkeypatch.setattr(study.subprocess,'check_output',lambda *a,**k:b'fixture')
    study.freeze()
    frozen=study.read(here/'FREEZE.json')
    for filename in deps+['experiments/old/FREEZE.json','experiments/old/runs/previous/model.json',
                         'experiments/new/CANDIDATE_LOCK.json','experiments/new/PRE_RUN_REVIEW.md',
                         'experiments/new/history_sources/original.py','experiments/new/extra_review.md',
                         'experiments/new/TEST_VALIDATION.json','experiments/new/validation/own.log',
                         'experiments/new/source/src/stub.py','tests/test_fixture.py']:
        assert filename in frozen['files']
    assert (here/'source/src/stub.py').read_bytes()==(tmp_path/'src/stub.py').read_bytes()
    study.verify_files(frozen['files'])
    (here/'source/src/stub.py').write_text('changed')
    with pytest.raises(RuntimeError,match='drift'):
        study.verify_files(frozen['files'])


def test_scoring_invokes_original_cell_guards_before_labels(monkeypatch):
    calls=[]
    monkeypatch.setattr(study,'check_candidate_lock',lambda:calls.append('lock'))
    def reject():
        calls.append('cell_guards')
        raise RuntimeError('fixture missing all-model seal')
    monkeypatch.setattr(evaluate.legacy,'_base_guards',reject)
    monkeypatch.setattr(study.cases,'generate',lambda *a:pytest.fail('No labels before guards.'))
    with pytest.raises(RuntimeError,match='seal'):
        evaluate.evaluate()
    assert calls==['lock','cell_guards']
    assert 'check_candidate_lock' in inspect.getsource(run_bounded.main)


def test_real_scoring_path_retains_all_nonfinite_cell_failures(monkeypatch,tmp_path):
    jobs=study.matrix()
    monkeypatch.setattr(evaluate.base,'HERE',tmp_path)
    monkeypatch.setattr(evaluate.base,'guards',lambda:(dict(jobs=jobs),{}))
    monkeypatch.setattr(study,'budget',lambda:study.time.time()+2880)
    monkeypatch.setattr(evaluate.base,'read',lambda p:dict(status='complete') if p.name=='resources.json' else {})
    monkeypatch.setattr(evaluate.base,'models',lambda *a:[object()])
    monkeypatch.setattr(evaluate.base,'prediction',lambda *a:np.full((4,2),float('nan')))
    monkeypatch.setattr(study.cases,'generate',lambda *a:(np.ones((4,2)),np.ones((4,2))))
    monkeypatch.setattr(study.cases,'latent',lambda *a:np.ones((4,2)))
    # Use nonzero observed-training SD; all arrays here are synthetic fixtures.
    monkeypatch.setattr(study.cases,'training_data',lambda *a:(np.ones((4,2)),np.arange(8.).reshape(4,2)))
    (tmp_path/'ALL_MODELS_FROZEN.json').write_text('{}')
    evaluate.evaluate()
    scores=study.read(tmp_path/'SCORES.json')['rows']
    assert len(scores)==32 and all(r['status']=='scoring_failed' for r in scores)
    assert all('Nonfinite' in r['error'] for r in scores)
    result=evaluate.summarize(scores,[dict(job=j,verified=False) for j in jobs])
    assert not result['adoption_pass'] and not result['technical_pass']
    assert result['methods']['repair']['mean_clean_nrmse'] is None
    json.dumps(result,allow_nan=False)


@pytest.mark.parametrize('fault',['empty_rows','missing_file','invalid_json','malformed_rows'])
def test_verify_keeps_32_rows_when_all_scores_are_missing(monkeypatch,tmp_path,fault):
    monkeypatch.setattr(evaluate,'HERE',tmp_path)
    monkeypatch.setattr(evaluate,'guards',lambda:(dict(jobs=study.matrix(),budget=dict(start_unix=study.time.time())),dict(frozen_unix=1)))
    monkeypatch.setattr(study,'budget',lambda:study.time.time()+2880)
    save(tmp_path/'FREEZE.json',{})
    save(tmp_path/'ALL_MODELS_FROZEN.json',{})
    if fault!='missing_file':
        save(tmp_path/'SCORES.json',dict(rows=[{}] if fault=='malformed_rows' else [],scored_unix=2,
            all_models_sha256=study.sha(tmp_path/'ALL_MODELS_FROZEN.json')))
    if fault=='invalid_json':(tmp_path/'SCORES.json').write_text('{')
    evaluate.verify()
    result=study.read(tmp_path/'VERIFICATION.json')
    summary=study.read(tmp_path/'SUMMARY.json')
    assert result['expected_systems']==len(result['rows'])==32 and result['verified_systems']==0
    assert result['status']=='failed' and not summary['technical_pass'] and not summary['adoption_pass']
    assert summary['methods']['repair']['mean_clean_nrmse'] is None
    assert (result['preparation_error'] is None)==(fault=='empty_rows')


@pytest.mark.parametrize('fault',['budget','resource'])
def test_fatal_scoring_error_retains_full_failure_receipt_without_retry(monkeypatch,tmp_path,fault):
    monkeypatch.setattr(evaluate,'HERE',tmp_path)
    jobs=study.matrix()
    monkeypatch.setattr(evaluate,'guards',lambda:(dict(jobs=jobs,budget=dict(start_unix=study.time.time())),dict(frozen_unix=1)))
    monkeypatch.setattr(study,'budget',lambda:study.time.time()+(2880 if fault=='resource' else -1))
    calls=[]
    def score():
        calls.append(True)
        raise FileNotFoundError('resources.json') if fault=='resource' else ValueError('Scoring budget expired.')
    monkeypatch.setattr(evaluate,'evaluate',score)
    save(tmp_path/'FREEZE.json',{});save(tmp_path/'ALL_MODELS_FROZEN.json',{})
    evaluate.run()
    receipt=study.read(tmp_path/'VERIFICATION.json');summary=study.read(tmp_path/'SUMMARY.json')
    assert len(calls)==1 and len(receipt['rows'])==32 and receipt['verified_systems']==0
    assert receipt['status']=='failed' and not summary['technical_pass'] and not summary['adoption_pass']
    assert summary['expected_systems']==32 and summary['expected_pairs']==16 and summary['expected_outputs']==32
    assert summary['methods']['repair']['mean_clean_nrmse'] is None
    assert (tmp_path/'SCORING_FAILURE.json').exists() and not (tmp_path/'SCORES.json').exists()
    with pytest.raises(RuntimeError,match='Keep existing'):
        evaluate.run()
    assert len(calls)==1
