import numpy as np
import pytest

from dimensr_mimo import SearchConfig,DimenSRMIMORegressor,AffineCoordinates,PhysicalModel
from experiments.physical_pool_20261027 import data,worker


def test_pool_matrix_and_test_data_gate(tmp_path,monkeypatch):
    jobs=data.jobs()
    assert len(jobs)==24 and len({data.name(j) for j in jobs})==24
    assert [sum(j['kind']==kind for j in jobs) for kind in ('synthetic','slump','affine')]==[17,6,1]
    monkeypatch.setattr(data,'HERE',tmp_path)
    for job in jobs:
        with pytest.raises(RuntimeError,match='Freeze all'):
            data.arrays(job,'test')


def test_pool_worker_never_refits_and_exports_unchanged_source(tmp_path,monkeypatch):
    cfg=SearchConfig(population_size=6,epochs=1,gp_generations=1,selection_generations=2,max_evaluations=60,
        min_candidates=2,polish_candidates=0,constant_restarts=1,constant_max_nfev=5,
        structured_seeds=False,operators=('add','mul'),max_nodes=7,max_depth=4)
    X=np.random.default_rng(42).uniform(.2,.8,(20,2));Y=(X[:,0]+X[:,1])[:,None]
    m=DimenSRMIMORegressor(cfg).fit(X,Y,X_val=X,Y_val=Y,input_bounds=[[0.,1.]]*2)
    source=tmp_path/'source.json';m.save(source);before=data.sha(source)
    bounds=[[0.,1.]]*2
    spec=dict(coordinates=AffineCoordinates.identity(2,1).to_dict(),bounds=bounds,
        limits=dict(nrmse=[2.],shared_graph_size=30,independent_parameters=10,expanded_total_nodes=40,maximum_output_depth=8))
    monkeypatch.setattr(worker,'ROOT',tmp_path)
    monkeypatch.setattr(data,'spec',lambda job:spec)
    def arrays(job,split):
        assert split in ('train','validation')
        return X,Y
    monkeypatch.setattr(data,'arrays',arrays)
    monkeypatch.setattr(DimenSRMIMORegressor,'fit',DimenSRMIMORegressor.fit)
    folder=tmp_path/'run';folder.mkdir()
    worker.run(folder,dict(case='fixture',seed=42,method='pool',source='source.json'),{})
    r=data.read(folder/'result.json')
    assert r['status']=='complete',r
    assert r['searches']==0 and not r['coefficients_refitted'] and r['source_model_unchanged']
    assert before==data.sha(source)==r['source_sha256']
    selected=PhysicalModel.load(folder/'selected.json')
    report=data.read(folder/'selection.json')
    assert selected.source['constants']==m.candidates_[report['candidate_index']]['constants']
