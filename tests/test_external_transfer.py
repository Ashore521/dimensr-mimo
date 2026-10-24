import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

directory=Path(__file__).resolve().parents[1]/'experiments/external_transfer_v1'
def module(name):
    spec=importlib.util.spec_from_file_location('external_test_'+name,directory/(name+'.py'))
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result

common=module('common');old=sys.modules.get('common');sys.modules['common']=common
try:data,study,evaluate,analysis=(module(n) for n in ('data','study','evaluate','analyze'))
finally:
    if old is None:sys.modules.pop('common',None)
    else:sys.modules['common']=old

@pytest.mark.parametrize('dataset,gap',[('water',7),('tetouan',1)])
def test_chronological_partitions_have_full_calendar_gaps(dataset,gap):
    days=np.repeat(np.arange(1000,1200),3);p=data.partition(dataset,days)
    assert days[p['train']].max()<days[p['val']].min()-gap
    assert days[p['val']].max()<days[p['test']].min()-gap
    assert len(set(np.concatenate(list(p.values()))))==sum(map(len,p.values()))

def test_video_ids_never_cross_partitions():
    groups=np.repeat(np.arange(100).astype(str),5);p=data.partition('video',groups)
    assert [len(np.unique(groups[p[k]])) for k in ('train','val','test')]==[60,20,20]
    assert not set(groups[p['train']]) & set(groups[p['test']])
    for k,v in data.partition('video',groups).items():np.testing.assert_array_equal(v,p[k])

def test_water_uses_inlet_predictors_and_declared_outlet_targets():
    row=['D-1/3/90']+[str(i) for i in range(1,39)]
    result=data.parse('water',','.join(row)+'\n\n')
    np.testing.assert_array_equal(result['X'][0],np.arange(1,10))
    np.testing.assert_array_equal(result['Y'][0],[24,25,26])
    row[5]='?';result=data.parse('water',','.join(row)+'\n')
    assert result['counts']['missing_or_nonfinite']==1 and len(result['X'])==0

def test_preprocessing_does_not_use_test_values(tmp_path,monkeypatch):
    rng=np.random.default_rng(7);X=rng.normal(size=(200,2));Y=rng.normal(size=(200,2));groups=np.repeat(np.arange(100).astype(str),2)
    raw=dict(X=X,Y=Y,ids=np.arange(200),groups=groups,feature_names=['a','b'],target_names=['u','v'],counts={})
    monkeypatch.setattr(data,'source_bytes',lambda key:'unused');monkeypatch.setattr(data,'parse',lambda *args:raw)
    results=[]
    for index in range(2):
        monkeypatch.setattr(data,'HERE',tmp_path/str(index));monkeypatch.setattr(data,'SEALED',tmp_path/str(index)/'sealed')
        results.append(data.prepare_one('video'))
        holdout=data.partition('video',groups)['test'];X[holdout]=1e8;Y[holdout]=-1e7
    for key in ('input_mean','input_scale','target_mean','target_scale','bounds'):
        assert results[0][key]==results[1][key]
    assert results[0]['sealed_sha256']!=results[1]['sealed_sha256']

def test_all_methods_fit_without_test_data_and_changed_model_blocks_opening(tmp_path,monkeypatch):
    rng=np.random.default_rng(8);X=rng.uniform(-1,1,(80,2));Y=np.column_stack((X[:,0]+X[:,1]**2,2*X[:,0]))
    split=dict(X_train=X[:60],Y_train=Y[:60],X_val=X[60:],Y_val=Y[60:])
    meta=dict(bounds=[[-1.,1.]]*2,target_scale=Y[:60].std(axis=0).tolist(),target_mean=Y[:60].mean(axis=0).tolist(),
        feature_names=['a','b'],target_names=['u','v'])
    monkeypatch.setattr(study,'check_freeze',lambda:None);monkeypatch.setattr(study,'development',lambda d:(split,meta))
    for k,v in (('BASE_SECONDS',.06),('CONTROL_SECONDS',.08),('EXTRA_SECONDS',.02)):monkeypatch.setattr(study,k,v)
    out=tmp_path/'runs/toy_s0';out.mkdir(parents=True)
    result=study.fit_job(('toy',0),out);assert result['status']=='ok' and set(result['choices'])==set(common.METHODS)
    common.save(out/'result.json',result);(tmp_path/'FREEZE.json').write_text('{}')
    receipt=dict(freeze_sha256=common.sha(tmp_path/'FREEZE.json'),records={'toy_s0':dict(result_sha256=common.sha(out/'result.json'),
        choices_sha256=result['choices_sha256'],models={k:c['model_sha256'] for k,c in result['choices'].items()})})
    common.save(tmp_path/'MODELS_FROZEN.json',receipt)
    monkeypatch.setattr(evaluate,'HERE',tmp_path);monkeypatch.setattr(evaluate,'check_freeze',lambda:None)
    monkeypatch.setattr(evaluate,'jobs',lambda:[('toy',0)])
    assert len(evaluate.verify_models())==1
    model=out/'semantic25.json';model.write_bytes(model.read_bytes()+b' ')
    monkeypatch.setattr(evaluate.np,'load',lambda *a,**k:pytest.fail('Test arrays opened before integrity gate'))
    with pytest.raises(ValueError,match='Model changed'):evaluate.main()
    assert not (tmp_path/'evaluation').exists()

def test_bootstrap_uses_counts_and_preserves_fixed_model_pairs():
    left=np.array([[[1.],[27.]],[[1.],[27.]]]);counts=np.array([1,3])
    result=analysis.paired_bootstrap(left,4*left,counts)
    assert result['difference']==pytest.approx(-np.sqrt(7))
    assert result['lower']==pytest.approx(-3.) and result['upper']==pytest.approx(-1.)
    assert result==analysis.paired_bootstrap(left,4*left,counts)
    np.testing.assert_array_equal(evaluate.group_errors(np.array([[1.],[4.],[9.]]),[0,1,1],2),[[1.],[13.]])

@pytest.mark.parametrize('error,k,passed',[(.99,44.,True),(1.005,36.,True),(.991,44.,False),(1.006,36.,False)])
def test_gate_boundaries_and_output_guard(error,k,passed):
    def group(e,size):return dict(all=dict(complete=True,nrmse={'mean':e},complexity={'mean':size}))
    groups=dict(semantic25=group(error,k),independent25=group(1.,40.),independent20=group(1.02,40.))
    rows=[dict(dataset=d,method=m,test={'nrmse':[e]}) for d in common.DATASETS for s in range(5)
          for m,e in (('semantic25',error),('independent25',1.))]
    assert analysis.gates(groups,rows)['eligible'] is passed
    for r in rows:
        if r['dataset']=='water' and r['method']=='semantic25':r['test']['nrmse']=[1.03]
    assert not analysis.gates(groups,rows)['eligible']
