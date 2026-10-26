from dataclasses import asdict
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dimensr_mimo import DimenSRMIMORegressor,SearchConfig
from experiments.shared_discovery_20261026 import cases,study,trace,worker


def fixture():
    X=np.random.default_rng(919).uniform(-.5,.5,(24,2))
    Y=np.column_stack((X[:,0]+X[:,1],X[:,0]-X[:,1]))
    cfg=SearchConfig(population_size=6,epochs=1,gp_generations=1,selection_generations=2,
        max_evaluations=90,min_candidates=3,polish_candidates=0,structured_seeds=False,
        constant_restarts=1,constant_max_nfev=5,max_nodes=9,max_depth=5,random_state=7)
    return X,Y,cfg


def test_matrix_and_test_firewall():
    jobs=study.matrix()
    assert len(jobs)==40 and sum(j['seconds'] for j in jobs)==2400
    assert len({(j['case'],j['seed']) for j in jobs})==20
    assert set(cases.SEEDS).isdisjoint({0,1,17,101,102,103,104,105,201,202,203,204,205})
    assert study.legacy.HERE==ROOT/'experiments/shared_discovery_20261026'
    with pytest.raises(ValueError,match='cannot request test'):
        cases.training_data('shared_parameter','test',301)


def test_trace_does_not_change_control_search(tmp_path):
    X,Y,cfg=fixture()
    base=DimenSRMIMORegressor(cfg).fit(X,Y,X_val=X,Y_val=Y,input_bounds=[[-.5,.5]]*2)
    base.save(tmp_path/'base.json')
    restore=trace.install(tmp_path/'trace.jsonl',False)
    try:
        observed=DimenSRMIMORegressor(cfg).fit(X,Y,X_val=X,Y_val=Y,input_bounds=[[-.5,.5]]*2)
    finally:restore()
    observed.save(tmp_path/'observed.json')
    a,b=[worker.read(tmp_path/n) for n in ('base.json','observed.json')]
    assert a['search_fingerprint']==b['search_fingerprint'] and a['candidates']==b['candidates']
    assert trace.summarize(tmp_path/'trace.jsonl',b)['seeds'].get('added',0)==0


@pytest.mark.parametrize('method',['control','trig_seeds'])
def test_worker_export_and_trace_reconstruction(tmp_path,monkeypatch,method):
    from experiments.shared_discovery_20261026 import verify_report as verify
    from dimensr_mimo.graph import System
    X,Y,cfg=fixture()
    # Artificial data only: this smoke never generates formal-study data.
    monkeypatch.setattr(cases,'training_data',lambda *args:(X,Y))
    monkeypatch.setattr(worker.base_worker,'generate',cases.training_data)
    job=dict(case='shared_parameter',seed=7,method=method,seconds=60)
    worker.write(tmp_path/'job.json',job)
    worker.run(tmp_path,job,asdict(cfg))
    result=worker.read(tmp_path/'result.json');assert result['status']=='complete',result
    data=worker.read(tmp_path/'model.json')
    discovery=verify.source_and_trace(tmp_path,data,dict(System=System),[(X,Y)]*3)
    assert (discovery['seeds'].get('added',0)>0)==(method=='trig_seeds')
    record=verify.guarded.selection_check(data,X,Y,data['scales'],False,0,dict(System=System))
    assert record['rule']==data['post_selection']
    assert result['reload_predictions_equal'] and result['reload_audits_equal']
