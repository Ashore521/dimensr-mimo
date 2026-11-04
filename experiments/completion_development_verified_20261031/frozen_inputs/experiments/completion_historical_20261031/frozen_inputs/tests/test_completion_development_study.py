from copy import deepcopy
import json
import numpy as np
import pytest

from experiments.compositional_completion_20261031 import harness,verification
from experiments.completion_development_20261031 import study,evaluate
from test_completion_historical_study import save,stage,summary_fixture,completion_fixture


@pytest.mark.parametrize('fault',[None,'missing','unaccepted','failed_science','changed_summary','changed_lock','changed_config',
    'technical_string','adoption_string','technical_integer','adoption_integer'])
def test_development_requires_accepted_historical_chain(monkeypatch,tmp_path,fault):
    here=tmp_path/'experiments/completion_development_20261031';prior=here.parent/'completion_historical_20261031';shared=here.parent/'shared'
    monkeypatch.setattr(study,'HERE',here);monkeypatch.setattr(study,'SHARED',shared)
    monkeypatch.setattr(harness,'configuration',lambda s:{'fixed':True})
    save(shared/'CANDIDATE_LOCK.json',{})
    for n in harness.GATE_FILES:save(prior/n,{})
    save(prior/'FREEZE.json',dict(config={'fixed':True},candidate_lock_sha256=harness.sha(shared/'CANDIDATE_LOCK.json')))
    save(prior/'SUMMARY.json',dict(technical_pass=True,adoption_pass=True))
    save(prior/'ALL_MODELS_FROZEN.json',dict(freeze_sha256=harness.sha(prior/'FREEZE.json')))
    save(prior/'SCORES.json',dict(all_models_sha256=harness.sha(prior/'ALL_MODELS_FROZEN.json')))
    save(prior/'VERIFICATION.json',dict(status='passed',freeze_sha256=harness.sha(prior/'FREEZE.json'),
        models_sha256=harness.sha(prior/'ALL_MODELS_FROZEN.json'),scores_sha256=harness.sha(prior/'SCORES.json')))
    hashes={n:harness.sha(prior/n) for n in harness.GATE_FILES};hashes['CANDIDATE_LOCK.json']=harness.sha(shared/'CANDIDATE_LOCK.json')
    save(prior/'POST_REVIEW_ACCEPTANCE.json',dict(status='accepted',sha256=hashes))
    if fault=='missing':(prior/'POST_REVIEW_ACCEPTANCE.json').unlink()
    elif fault=='unaccepted':save(prior/'POST_REVIEW_ACCEPTANCE.json',dict(status='pending',sha256=hashes))
    elif fault=='failed_science':
        save(prior/'SUMMARY.json',dict(technical_pass=True,adoption_pass=False));hashes['SUMMARY.json']=harness.sha(prior/'SUMMARY.json')
        save(prior/'POST_REVIEW_ACCEPTANCE.json',dict(status='accepted',sha256=hashes))
    elif fault in ('technical_string','adoption_string','technical_integer','adoption_integer'):
        field,kind=fault.split('_');summary=dict(technical_pass=True,adoption_pass=True)
        summary[field+'_pass']='false' if kind=='string' else 1
        save(prior/'SUMMARY.json',summary);hashes['SUMMARY.json']=harness.sha(prior/'SUMMARY.json')
        save(prior/'POST_REVIEW_ACCEPTANCE.json',dict(status='accepted',sha256=hashes))
    elif fault=='changed_summary':save(prior/'SUMMARY.json',dict(technical_pass=False,adoption_pass=False))
    elif fault=='changed_lock':save(shared/'CANDIDATE_LOCK.json',{'new':'source'})
    elif fault=='changed_config':monkeypatch.setattr(harness,'configuration',lambda s:{'changed':True})
    if fault:
        with pytest.raises((ValueError,OSError)):study.check_stage_gate()
    else:study.check_stage_gate()


def test_development_candidate_all_gate_even_when_control_equal(summary_fixture):
    s,v,scores,rows,_=summary_fixture
    if s.STAGE!='development':return
    pair=scores[:2]
    for r in pair:r['both_accepted']=False
    result=v.summarize(scores,rows)
    assert result['technical_pass'] and result['control_acceptance_preserved']
    assert result['methods']['completion']['both_accepted']==result['methods']['base']['both_accepted']==15
    assert not result['candidate_all_accepted'] and not result['adoption_pass']


def test_completion_missing_root_copies_parent_without_cycling(completion_fixture):
    v,model,payload,X,Y,cfg,known,r=completion_fixture
    r['initialization_diagnostics'][1]['returned_sources']=[]
    r['initialization_diagnostics'][1]['compositional_retained_indices']=[]
    r['compositional_returned_roots']=1
    source=r['events'][0]['root_sources'][0]
    r['events'][0]['root_sources']=[source,{'kind':'parent'}]
    from dimensr_mimo.graph import Builder,System
    from dimensr_mimo._single.expression import Node
    from dimensr_mimo.genetics import clone_subgraph
    b=Builder();b.roots.append(b.tree(Node.from_dict(source['tree']),source['constants']))
    parent=System.from_dict(payload[0]['system']);b.roots.append(clone_subgraph(b,parent,[],parent.roots[1]))
    graph,warm=b.finish();event=r['events'][0]
    event.update(graph=graph.to_dict(),warm=warm.tolist(),compositional_outputs=[0])
    event['child'].update(system=graph.to_dict(),errors=[event['child']['errors'][0],payload[0]['errors'][1]],complexity=float(graph.size))
    r['missing_roots']=1
    assert v.completion_evidence(model,payload,X,Y,cfg,known)[1]['missing_roots']==1
    event['graph']['nodes'][-1]['token']='x0'
    with pytest.raises(ValueError,match='graph'):v.completion_evidence(model,payload,X,Y,cfg,known)


def test_freeze_closure_contains_live_and_archived_dependencies(monkeypatch,tmp_path):
    s=study;here=tmp_path/'experiments/development';prior=tmp_path/'experiments/old';shared=tmp_path/'experiments/shared'
    monkeypatch.setattr(s,'ROOT',tmp_path);monkeypatch.setattr(s,'HERE',here);monkeypatch.setattr(s,'PRIOR',prior);monkeypatch.setattr(s,'SHARED',shared)
    deps=['src/stub.py','experiments/precision_delivery_20261026/selector.py','experiments/guarded_selection_20261026/selector.py',
          'experiments/native_api_20261026/runs/dimensional_frequency_normalized/model.json']
    for dep in deps:save(tmp_path/dep,{});save(prior/'frozen_inputs'/dep,{'old':True})
    save(prior/'FREEZE.json',dict(files={n:harness.sha(prior/'frozen_inputs'/n) for n in deps}))
    save(prior/'runs/source/model.json',{})
    save(prior/'ALL_MODELS_FROZEN.json',dict(files={'runs/source/model.json':harness.sha(prior/'runs/source/model.json')}))
    for n in ('FROZEN_INPUT_ARCHIVE.json','SCORES.json','VERIFICATION.json','SUMMARY.json','POST_RUN_REVIEW.md'):save(prior/n,{})
    for n in ('PLAN.md','PROTOCOL.md','HARNESS_REVIEW.md','TEST_VALIDATION.json','PREFLIGHT.json','study.py',
              'harness_validation/fixture.log','harness_review_validation/review.log'):save(here/n,{})
    for n in ('PLAN.md','DESIGN_REVIEW.md','BASE_SOURCE_LOCK.json','CANDIDATE_LOCK.json','CORE_REVIEW.md','harness.py','verification.py'):save(shared/n,{})
    gate=here.parent/'completion_historical_20261031'
    for n in harness.GATE_FILES+('POST_REVIEW_ACCEPTANCE.json',):save(gate/n,{})
    for n in ('study.py','worker.py','evaluate.py','run_bounded.py','PLAN.md','PROTOCOL.md'):save(gate/n,{})
    save(tmp_path/'tests/test_fixture.py',{});save(tmp_path/'pyproject.toml',{})
    save(tmp_path/'docs/COMPOSITIONAL_COMPLETION.zh-CN.md',{})
    paths=s.freeze_inputs()
    assert all(p.is_file() for p in paths)
    assert tmp_path/deps[0] in paths and prior/'frozen_inputs'/deps[0] in paths
    assert prior/'runs/source/model.json' in paths and shared/'verification.py' in paths
    assert here/'harness_validation/fixture.log' in paths and gate/'POST_REVIEW_ACCEPTANCE.json' in paths
    assert here/'harness_review_validation/review.log' in paths
    assert all(gate/n in paths for n in ('study.py','worker.py','evaluate.py','run_bounded.py','PLAN.md','PROTOCOL.md'))
    monkeypatch.setattr(s,'check_candidate_lock',lambda:None);monkeypatch.setattr(s,'check_stage_gate',lambda:None)
    monkeypatch.setattr(s,'configuration',lambda:{})
    monkeypatch.setattr(s.cases,'CASES',['fixture']);monkeypatch.setattr(s.cases,'seeds',lambda c:[1]);monkeypatch.setattr(s.cases,'design',lambda c:{})
    monkeypatch.setattr(s.cases,'training_data',lambda *a:(np.ones((2,2)),np.ones((2,2))))
    monkeypatch.setattr(harness.subprocess,'check_output',lambda *a,**k:b'fixture')
    save(here/'PREFLIGHT.json',dict(status='passed'))
    s.freeze()
    frozen=harness.read(here/'FREEZE.json')
    assert 'experiments/development/source/src/stub.py' in frozen['files']
    assert (here/'source/src/stub.py').read_bytes()==(tmp_path/'src/stub.py').read_bytes()
    s.verify_files(frozen['files'])
    (here/'source/src/stub.py').write_text('changed')
    with pytest.raises(ValueError,match='drift'):s.verify_files(frozen['files'])


def test_nonfinite_scoring_keeps_all_cells(monkeypatch,tmp_path):
    s=study;v=verification.Verifier(s)
    monkeypatch.setattr(v.base,'HERE',tmp_path)
    monkeypatch.setattr(v.base,'guards',lambda:(dict(jobs=s.matrix()),{}))
    monkeypatch.setattr(s,'budget',lambda:verification.time.time()+2880)
    monkeypatch.setattr(v.base,'read',lambda p:dict(status='complete') if p.name=='resources.json' else {})
    monkeypatch.setattr(v.base,'models',lambda *a:[object()])
    monkeypatch.setattr(v.base,'prediction',lambda *a:np.full((4,2),float('inf')))
    monkeypatch.setattr(s.cases,'generate',lambda *a:(np.ones((4,2)),np.ones((4,2))))
    monkeypatch.setattr(s.cases,'latent',lambda *a:np.ones((4,2)))
    monkeypatch.setattr(s.cases,'training_data',lambda *a:(np.ones((4,2)),np.arange(8.).reshape(4,2)))
    save(tmp_path/'ALL_MODELS_FROZEN.json',{})
    v.evaluate();rows=harness.read(tmp_path/'SCORES.json')['rows']
    assert len(rows)==32 and all(r['status']=='scoring_failed' for r in rows)
    assert all('Invalid normalized' in r['error'] for r in rows)


def test_train_does_not_rerun_unsealed_jobs(monkeypatch,tmp_path):
    s=study;monkeypatch.setattr(s,'HERE',tmp_path);monkeypatch.setattr(s,'check_candidate_lock',lambda:None)
    monkeypatch.setattr(s,'check_stage_gate',lambda:None);monkeypatch.setattr(s,'verify_files',lambda *a:None)
    jobs=s.matrix();save(tmp_path/'FREEZE.json',dict(files={},jobs=jobs))
    save(tmp_path/'runs'/s.name(jobs[0])/'job.json',jobs[0])
    monkeypatch.setattr(s.supervisor,'run_job',lambda *a:pytest.fail('No retry.'))
    with pytest.raises(ValueError,match='incident'):s.train()
    assert not (tmp_path/'ALL_MODELS_FROZEN.json').exists()
