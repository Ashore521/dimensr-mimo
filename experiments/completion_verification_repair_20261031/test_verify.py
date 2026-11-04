from copy import deepcopy
import json
from types import SimpleNamespace
import pytest
import verify_sealed as repair


def model():
    return dict(config=dict(epochs=1, gp_generations=2, population_size=24, wall_seconds=52),
        diagnostics=dict(deadline_reached=False, cache='fixed', sharing_closure={'precision_repair':{'triggered':True}}),
        best={'errors':[.01,.02]}, history=[
            dict(phase='phase_budget', step=0, epoch_allowance_seconds=8., neural_until_seconds=5., gp_until_seconds=8., exploration_until_seconds=39.),
            dict(phase='neural', step=0, elapsed_seconds=2., proposals=24, evaluations=24),
            dict(phase='gp', step=0, elapsed_seconds=3.), dict(phase='gp', step=1, elapsed_seconds=4.)])


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def test_only_three_nonbinding_paths_normalized():
    a=model();b=deepcopy(a)
    b['history'][0].update(epoch_allowance_seconds=7., neural_until_seconds=6., gp_until_seconds=9.)
    assert repair.normalized_model(a)==repair.normalized_model(b)
    for value in (a,b):
        value['history'][1]['gp_until_seconds']=10.
    b['history'][1]['gp_until_seconds']=11.
    assert repair.normalized_model(a)!=repair.normalized_model(b)


@pytest.mark.parametrize('fault', ['infinite','negative','string','bool','bad_order','missing_epoch','duplicate_gp',
    'missing_gp','short_neural','neural_deadline','gp_deadline','global_deadline','wrong_phase'])
def test_unsafe_phase_evidence_rejected(fault):
    m=model();budget=m['history'][0]
    if fault in ('infinite','negative','string','bool'):
        budget['epoch_allowance_seconds']={'infinite':float('inf'),'negative':-1.,'string':'8','bool':True}[fault]
    elif fault=='bad_order':budget['gp_until_seconds']=4.
    elif fault=='missing_epoch':m['config']['epochs']=2
    elif fault=='duplicate_gp':m['history'][3]['step']=0
    elif fault=='missing_gp':m['history'].pop()
    elif fault=='short_neural':m['history'][1]['proposals']=23
    elif fault=='neural_deadline':m['history'][1]['elapsed_seconds']=5.
    elif fault=='gp_deadline':m['history'][3]['elapsed_seconds']=8.
    elif fault=='global_deadline':m['diagnostics']['deadline_reached']=True
    elif fault=='wrong_phase':budget['phase']='unrelated'
    with pytest.raises((ValueError,KeyError)):repair.normalized_model(m)


@pytest.mark.parametrize('drift',[None,'errors','cache','history','config','result','delivery','pool','score',
    'old_table_duplicate','old_table_missing','old_table_extra','old_table_incomplete'])
def test_control_comparison_keeps_all_non_timing_evidence(tmp_path,drift):
    current=tmp_path/'current';prior=tmp_path/'prior';job=dict(case='fixture',seed=1,method='base',seconds=60)
    name=lambda j:j['case']+'_s'+str(j['seed'])+'_'+j['method']
    s=SimpleNamespace(HERE=current,PRIOR=prior,name=name,EXPECTED=2)
    v=repair.SealedVerifier.__new__(repair.SealedVerifier);v.study=s
    a=model();b=deepcopy(a);b['history'][0].update(epoch_allowance_seconds=7.,neural_until_seconds=6.,gp_until_seconds=9.)
    score=dict(job=job,status='complete',clean_nrmse=[.01,.02],observed_nrmse=[.01,.02],mean_clean_nrmse=.015,
        both_accepted=True,structure=dict(shared_graph_size=4,independent_parameters=2,expanded_total_nodes=5,maximum_output_depth=2))
    old_score=deepcopy(score);old_score['job']['method']='repair'
    if drift=='errors':a['best']['errors'][0]=.02
    elif drift=='cache':a['diagnostics']['cache']='changed'
    elif drift=='history':a['history'][1]['evaluations']=25
    elif drift=='config':a['config']['new_algorithm_setting']=True
    elif drift=='score':old_score['clean_nrmse'][0]=.015
    for filename in ('model.json','result.json','delivery.json','pool.json'):
        x,y=(a,b) if filename=='model.json' else ({'unchanged':True},{'unchanged':True})
        if filename==str(drift)+'.json':x={'changed':True}
        save(current/'runs'/name(job)/filename,x)
        save(prior/'runs'/name(old_score['job'])/filename,y)
    old_rows=[deepcopy(score),old_score]
    if drift=='old_table_duplicate':old_rows.append(deepcopy(old_rows[0]))
    elif drift=='old_table_missing':old_rows.pop(0)
    elif drift=='old_table_extra':
        extra=deepcopy(score);extra['job']['case']='extra';old_rows.append(extra)
    elif drift=='old_table_incomplete':old_rows[0]['status']='scoring_failed'
    save(prior/'SCORES.json',{'rows':old_rows})
    if drift:
        with pytest.raises(ValueError):v.control_identity(job,score)
    else:assert v.control_identity(job,score)


def test_input_drift_stops_before_verifier(monkeypatch,tmp_path):
    monkeypatch.setattr(repair,'HERE',tmp_path);monkeypatch.setattr(repair,'ROOT',tmp_path)
    save(tmp_path/'input.json',{'changed':True})
    save(tmp_path/'FREEZE.json',{'files':{'input.json':'wrong'}})
    monkeypatch.setattr(repair,'SealedVerifier',lambda *a:pytest.fail('No verification after drift.'))
    with pytest.raises(ValueError,match='drift'):repair.verify()


def test_missing_prereview_does_not_start(monkeypatch,tmp_path):
    monkeypatch.setattr(repair,'HERE',tmp_path)
    for name in ('PLAN.md','TEST_VALIDATION.json'):save(tmp_path/name,{})
    with pytest.raises(ValueError,match='Missing prerequisite'):repair.run()
    assert not (tmp_path/'FREEZE.json').exists()
