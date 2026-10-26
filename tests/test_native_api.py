from dataclasses import replace
import json
import numpy as np
import pytest
from dimensr_mimo import DimenSRMIMORegressor,SearchConfig
from dimensr_mimo._single.units import ZERO,unit
from dimensr_mimo import regressor
from experiments.unit_composition_20261026.worker import intervention
from experiments.guarded_selection_20261026.selector import derive


def config():
    return SearchConfig(population_size=6,epochs=1,gp_generations=1,selection_generations=2,max_evaluations=100,
        min_candidates=3,polish_candidates=0,constant_restarts=1,constant_max_nfev=5,
        operators=('add','sub','mul','div','square','sin','cos','sqrt','log'))


def limits():return dict(nrmse=[1e-4,1e-4],shared_graph_size=24,independent_parameters=6,expanded_total_nodes=30,maximum_output_depth=6)


def test_native_dimensional_search_matches_frozen_experimental_helpers(tmp_path):
    from experiments.typed_coverage_20261026.observe import Observation
    L=unit([1,0,0,0,0,0,0]);K=unit([-1,0,0,0,0,0,0])
    spec=dict(features=[L,L],targets=[L,L],constants=[ZERO,K,L],bounds=[[-.8,.8]]*2)
    X=np.random.default_rng(11).uniform(-.8,.8,(32,2));Y=np.column_stack((X[:,0]+X[:,1],X[:,0]-X[:,1]))
    cfg=config()
    def fit(c,callback=None):return DimenSRMIMORegressor(c,spec['features'],spec['targets'],spec['constants']).fit(X[:24],Y[:24],X_val=X[24:],Y_val=Y[24:],input_bounds=spec['bounds'],callback=callback)
    with intervention('normalized',spec),Observation(tmp_path/'trace.jsonl',cfg,spec,True) as observed:
        original=fit(cfg,observed.event)
    expected=derive(original.to_dict(),'guarded')
    native=fit(replace(cfg,initialization='dimensional',final_selection='guarded'))
    assert native.search_fingerprint_==original.search_fingerprint_
    assert native.diagnostics_['frozen_candidate_sha256']==original.diagnostics_['frozen_candidate_sha256']
    assert native.to_dict()['best']==expected['best']
    assert native.post_selection_['candidate_index']==expected['post_selection']['candidate_index']


def test_precision_limits_validate_before_search(monkeypatch):
    def forbidden(*a,**k):raise AssertionError('Search should not start.')
    monkeypatch.setattr(regressor.Evaluator,'__init__',forbidden)
    cfg=replace(config(),final_selection='precision',acceptance=limits())
    with pytest.raises(ValueError,match='match outputs'):
        DimenSRMIMORegressor(cfg).fit(np.ones((10,2)),np.ones((10,1)))
    for wrong in (dict(initialization='guess'),dict(final_selection='precision'),dict(acceptance=limits()),dict(final_selection='precision',acceptance=limits(),use_units=False)):
        with pytest.raises(ValueError):replace(config(),**wrong).validate()


def test_precision_not_met_persists_and_configuration_is_frozen(tmp_path):
    threshold=dict(limits(),shared_graph_size=0)
    cfg=replace(config(),initialization='dimensional',final_selection='precision',acceptance=threshold,structured_seeds=False)
    X=np.random.default_rng(12).uniform(-.8,.8,(28,2));Y=np.column_stack((X[:,0],X[:,1]))
    m=DimenSRMIMORegressor(cfg).fit(X,Y,X_val=X,Y_val=Y,input_bounds=[[-.8,.8]]*2)
    assert m.post_selection_['status']=='not_met'
    threshold['nrmse'][0]=100
    assert m.config_.acceptance['nrmse'][0]==1e-4
    m.save(tmp_path/'model.json');loaded=DimenSRMIMORegressor.load(tmp_path/'model.json')
    loaded.save(tmp_path/'again.json')
    assert json.loads((tmp_path/'again.json').read_text())==json.loads((tmp_path/'model.json').read_text())
    np.testing.assert_array_equal(m.predict(X),loaded.predict(X))
    tampered=json.loads((tmp_path/'model.json').read_text());tampered['post_selection']['status']='accepted'
    (tmp_path/'tampered.json').write_text(json.dumps(tampered),encoding='utf-8')
    with pytest.raises(ValueError,match='selection or acceptance'):DimenSRMIMORegressor.load(tmp_path/'tampered.json')
