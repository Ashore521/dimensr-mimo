from dataclasses import asdict
from pathlib import Path
import sys
import json
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.public_transfer_20261026 import data,study,worker


def test_frozen_matrix_and_test_cache_firewall(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'HERE',tmp_path)
    data.write(tmp_path/'ALGORITHM.json',{'trig_seeds':False})
    jobs=study.matrix()
    assert len(jobs)==18 and sum(j['seconds']==60 for j in jobs)==12
    assert sum(j['seconds'] for j in jobs)==810
    with pytest.raises(ValueError,match='cannot access test'):
        data.training_data('energy','test',401)


def test_public_manifest_partitions_and_normalization():
    if not (data.HERE/'DATA_MANIFEST.json').exists():pytest.skip('Local public-data preparation is not bundled in sdist.')
    for name,meta in data.manifest().items():
        ids=[set(meta['row_ids'][split]) for split in ('train','validation','test')]
        assert len(set.union(*ids))==meta['rows']==sum(map(len,ids))
        assert 'cnt' not in meta['feature_names'] and not set(meta['target_names']) & set(meta['feature_names'])
        if not (ROOT/meta['development']).exists():continue
        X,Y=data.training_data(name,'train',401)
        np.testing.assert_allclose(X.mean(axis=0),0,atol=1e-12)
        np.testing.assert_allclose(Y.std(axis=0),meta['target_scale'],rtol=0,atol=0)
    if 'energy' in data.manifest():
        groups=[set(g) for g in data.manifest()['energy']['groups_by_split'].values()]
        assert sum(map(len,groups))==len(set.union(*groups))==12


@pytest.mark.parametrize('trig_seeds',[False,True])
def test_all_worker_paths_scoring_and_verification_on_artificial_data(tmp_path,monkeypatch,trig_seeds):
    from dimensr_mimo import SearchConfig,DimenSRMIMORegressor
    from dimensr_mimo._single import DimenSRRegressor
    from dimensr_mimo.graph import System
    from dimensr_mimo._single.expression import Node
    from experiments.public_transfer_20261026 import verify_report as verify
    X=np.random.default_rng(919).uniform(-.5,.5,(24,2));Y=np.column_stack((X[:,0]+X[:,1],X[:,0]-X[:,1]))
    cfg=asdict(SearchConfig(population_size=6,epochs=1,gp_generations=1,selection_generations=2,max_evaluations=90,
        min_candidates=3,polish_candidates=0,structured_seeds=False,constant_restarts=1,constant_max_nfev=5,max_nodes=9,max_depth=5))
    cfg=json.loads(json.dumps(cfg))
    for module in (worker,study,verify,verify.core):monkeypatch.setattr(module,'HERE',tmp_path)
    data.write(tmp_path/'ALGORITHM.json',{'trig_seeds':trig_seeds})
    zero=[0]*7
    spec=dict(bounds=[[-.5,.5]]*2,features=[zero]*2,targets=[zero]*2,constants=[zero])
    monkeypatch.setattr(data,'training_data',lambda *args:(X,Y));monkeypatch.setattr(data,'generate',lambda *args:(X,Y))
    monkeypatch.setattr(data,'design',lambda *args:spec);monkeypatch.setattr(worker.base_worker,'generate',data.training_data)
    monkeypatch.setattr(worker.base_worker,'design',data.design)
    classes=dict(joint=DimenSRMIMORegressor,single=DimenSRRegressor,System=System,Node=Node,design=data.design)
    for method in study.METHODS:
        job=dict(case='energy',seed=7,method=method,seconds=15 if method=='candidate' and not trig_seeds else 60)
        folder=tmp_path/'runs'/f'energy_s7_{method}';folder.mkdir(parents=True)
        data.write(folder/'job.json',job)
        worker.run(folder,job,cfg);result=data.read(folder/'result.json')
        assert result['status']=='complete',result
        resource=dict(status='complete',job=job,returncode=0,wall_seconds=.1)
        data.write(folder/'resources.json',resource);study.checkpoint(folder)
        row=dict(job=job,resources=resource,mechanism=study.mechanism(folder),**study.score_cell(folder,job))
        assert row['score_status']=='complete'
        verified=verify.core.verify_run(job,row,{'configs':{method:cfg}},[(X,Y)]*3,classes)
        assert verified['verified'] and verified['domain_pass']
