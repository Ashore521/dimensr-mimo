"""Extra constructed checks of the new verifier fork; no real data or fitting."""
import json
import importlib.util
from types import SimpleNamespace
import pytest
from experiments.completion_release_validation_20261104 import test_stage as fixtures
from experiments.completion_release_validation_20261104 import evaluate,study


def test_stage_preseal_adapter_runs_real_guards_and_restores_paths(tmp_path):
    from experiments.completion_development_verified_20261031 import study as development,evaluate as dev_evaluate
    path=study.ROOT/'experiments/conftest.py'
    spec=importlib.util.spec_from_file_location('_independent_experiment_adapter',path)
    adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
    function=adapter.isolated_stage_preseal_directory.__wrapped__
    for stage,verifier,label in [(development,dev_evaluate.verifier,'test_new_thin_entries_and_isolated_bindings'),
                                (study,evaluate.verifier,'test_new_cases_and_verifiers_are_bound_without_old_data')]:
        owners=(stage,stage.cases,stage.supervisor,verifier.legacy,verifier.base)
        before=[o.HERE for o in owners];generate=stage.cases.generate
        directory=tmp_path/stage.STAGE;directory.mkdir()
        with pytest.MonkeyPatch.context() as patch:
            class Request:
                node=SimpleNamespace(originalname=label)
                module=SimpleNamespace(study=stage,evaluate=SimpleNamespace(verifier=verifier))
                def getfixturevalue(self,name):return {'monkeypatch':patch,'tmp_path':directory}[name]
            function(Request())
            assert all(o.HERE==directory for o in owners) and stage.cases.generate is generate
            patch.setattr(fixtures.np.random,'default_rng',lambda *a,**kw:pytest.fail('Actual guard must reject before RNG.'))
            with pytest.raises(RuntimeError,match='seal'):
                stage.cases.generate(stage.cases.CASES[0],'test',stage.cases.seeds(stage.cases.CASES[0])[0])
        assert [o.HERE for o in owners]==before
    class OtherRequest:
        node=SimpleNamespace(originalname='unrelated_test')
        def getfixturevalue(self,name):pytest.fail('Unrelated tests must not resolve adapter fixtures.')
    function(OtherRequest())


@pytest.mark.parametrize('fault',['payload','fingerprint','prefix_hash','zero_count','bool_count'])
def test_each_foundation_identity_component_is_required(tmp_path,monkeypatch,fault):
    v,scores,verified,models,root=fixtures.summary_data.__wrapped__(tmp_path,monkeypatch)
    target=next(j for j in study.matrix() if j['method']=='completion')
    model=models[study.name(target)];record=model['diagnostics']['sharing_closure']
    if fault=='payload':
        model['candidates'][0]['errors'][0]=2e-8
        record['base_candidate_sha256']=v.legacy.digest(v.legacy.training_payload(model))
    elif fault=='fingerprint':record['base_search_fingerprint']='different-effective-cache'
    elif fault=='prefix_hash':record['base_candidate_sha256']='wrong-payload-hash'
    elif fault=='zero_count':record['base_candidate_count']=0;record['base_candidate_sha256']=v.legacy.digest([])
    else:record['base_candidate_count']=True
    fixtures.put(root/'runs'/study.name(target)/'model.json',model)
    result=v.summarize(scores,verified)
    assert not result['paired_base_identity_pass'] and not result['technical_pass'] and not result['adoption_pass']
    assert 'archived_control_identity' not in json.dumps(result)


@pytest.mark.parametrize('fault',['missing_scores','malformed_scores','broken_guard','scoring_failure'])
def test_new_verifier_fork_keeps_all_32_failure_rows(tmp_path,monkeypatch,fault):
    v,scores,verified,models,root=fixtures.summary_data.__wrapped__(tmp_path,monkeypatch)
    now=evaluate.time.time()
    frozen=dict(jobs=study.matrix(),budget=dict(start_unix=now))
    def guards():
        if fault=='broken_guard':raise ValueError('Constructed broken seal.')
        return frozen,dict(frozen_unix=1.)
    monkeypatch.setattr(v,'guards',guards)
    monkeypatch.setattr(study,'budget',lambda:now+2880)
    monkeypatch.setattr(v.base,'verify_cell',lambda *a:pytest.fail('Preparation failure must not enter native verification.'))
    if fault=='malformed_scores':(root/'SCORES.json').write_text('{invalid',encoding='utf-8')
    if fault=='scoring_failure':fixtures.put(root/'SCORING_FAILURE.json',dict(status='failed'))
    v.verify()
    receipt=fixtures.harness.read(root/'VERIFICATION.json');summary=fixtures.harness.read(root/'SUMMARY.json')
    assert receipt['status']=='failed' and receipt['verified_systems']==0 and len(receipt['rows'])==32
    assert all(row['verified'] is False for row in receipt['rows'])
    assert not summary['technical_pass'] and not summary['adoption_pass']
    assert summary['methods']['base']['mean_clean_nrmse'] is None
    assert summary['methods']['completion']['mean_clean_nrmse'] is None
    assert len(summary['completion_failures'])==16
    assert 'archived_control_identity' not in json.dumps(receipt)+json.dumps(summary)
