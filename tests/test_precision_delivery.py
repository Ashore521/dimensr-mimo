import pytest
from dimensr_mimo import SearchConfig
from dimensr_mimo.graph import System,Vertex
from dataclasses import asdict
from experiments.guarded_selection_20261026.selector import FIELDS,digest
from experiments.precision_delivery_20261026.selector import select


def source():
    simple=System((Vertex('x0'),),(0,0))
    large=System((Vertex('x0'),Vertex('c0',parameter=0),Vertex('mul',(0,1))),(2,2))
    pool=[]
    for s,error in [(large,1e-8),(simple,5e-5)]:
        pool.append(dict(system=s.to_dict(),constants=[1.] if s.n_parameters else [],errors=[error]*2,validation_errors=[error]*2,
            complexity=float(s.size),rademacher=0.,rademacher_complexity=0.,violation=0.,score=error))
    return dict(n_outputs=2,config=asdict(SearchConfig()),candidates=pool,best={k:pool[0][k] for k in FIELDS},
        feature_units=[[0]*7],target_units=[[0]*7]*2,constant_units=[[0]*7],reference_train=[1.,1.],
        diagnostics=dict(frozen_candidate_sha256=digest([{k:c[k] for k in FIELDS} for c in pool])))


def limits():return dict(nrmse=[1e-4]*2,shared_graph_size=24,independent_parameters=6,expanded_total_nodes=30,maximum_output_depth=6)


def rehash(s):s['diagnostics']['frozen_candidate_sha256']=digest([{k:c[k] for k in FIELDS} for c in s['candidates']]);s['best']={k:s['candidates'][0][k] for k in FIELDS}


def test_absolute_precision_does_not_relax_relative_protection():
    s=source();out=select(s,limits())
    assert out['post_selection']['status']=='accepted' and out['post_selection']['candidate_index']==0
    assert out['post_selection']['feasible_count']==1
    assert 'post_selection' not in s


def test_smallest_feasible_graph_and_explicit_failure():
    s=source()
    for c in s['candidates']:c['errors']=c['validation_errors']=[5e-5,5e-5]
    # Keep the larger weighted anchor strictly preferred, before post-selection.
    s['candidates'][1]['score']=1.;rehash(s)
    out=select(s,limits());assert out['post_selection']['candidate_index']==1
    assert out['post_selection']['status']=='accepted'
    strict=dict(limits(),nrmse=[1e-9]*2);failure=select(s,strict)
    assert failure['post_selection']['status']=='not_met' and failure['post_selection']['fallback_method']=='guarded'
    assert failure['post_selection']['feasible_count']==0
    assert failure['diagnostics']['application_acceptance']['status']=='not_met'


def test_invalid_threshold_and_frozen_pool_tampering_rejected():
    with pytest.raises(ValueError):select(source(),dict(limits(),nrmse=[float('nan')]*2))
    s=source();s['candidates'][1]['errors'][0]=0.1
    with pytest.raises(ValueError,match='pool'):select(s,limits())


def test_unmet_acceptance_survives_estimator_resave(tmp_path):
    import json
    from dimensr_mimo import DimenSRMIMORegressor
    s=source();s.update(format='dimensr-mimo-dag',schema_version=1,package_version='1.0.0',n_features=1,
        feature_names=['distance'],scales=[1.,1.],validation_nrmse=[1e-8]*2,reference_validation=[1.,1.],history=[],search_fingerprint='fixture')
    selected=select(s,dict(limits(),nrmse=[0.,0.]))
    p=tmp_path/'selected.json';p.write_text(json.dumps(selected),encoding='utf-8')
    model=DimenSRMIMORegressor.load(p);model.save(tmp_path/'resaved.json')
    reloaded=DimenSRMIMORegressor.load(tmp_path/'resaved.json')
    assert reloaded.diagnostics_['application_acceptance']==selected['post_selection']
    assert reloaded.diagnostics_['application_acceptance']['status']=='not_met'
