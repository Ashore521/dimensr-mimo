import json
import numpy as np
import pytest
from dimensr_mimo import AffineCoordinates,PhysicalModel,DimenSRMIMORegressor,SearchConfig
from dimensr_mimo.graph import System,Vertex
from dimensr_mimo.evaluation import Candidate
from dimensr_mimo._single.units import ZERO,unit
from dimensr_mimo.physical import restore_graph


def model():
    L=unit([1,0,0,0,0,0,0]);V=unit([1,0,-1,0,0,0,0]);K=unit([-1,0,0,0,0,0,0])
    s=System((Vertex('c0',units=K,parameter=0),Vertex('x0',units=L),Vertex('mul',(0,1)),Vertex('sin',(2,)),
        Vertex('c1',units=L,parameter=1),Vertex('mul',(4,3),L),Vertex('x1',units=L),Vertex('mul',(0,6)),
        Vertex('cos',(7,)),Vertex('c2',units=V,parameter=2),Vertex('mul',(9,8),V)),(5,10))
    m=DimenSRMIMORegressor();m.config_=SearchConfig();m.n_features_in_=2;m.n_outputs_=2
    m.best_=Candidate(s,np.array([1.7,.8,.6]),np.zeros(2),11.,0.,0.)
    m.feature_units_=(L,L);m.target_units_=(L,V);m.constant_units_=(K,L,V);m.feature_names_=['distance0','distance1']
    m.scales_=np.ones(2);m.reference_train_=m.reference_validation_=np.ones(2);m.validation_nrmse_=np.zeros(2)
    m.diagnostics_=dict(domain=dict(input_bounds=[[-.8,.8]]*2));m.search_fingerprint_='unit-test-fixture'
    return m


def limits():return dict(nrmse=[1e-4]*2,shared_graph_size=24,independent_parameters=6,expanded_total_nodes=30,maximum_output_depth=6)


def test_physical_affine_restore_units_binding_and_complexity(tmp_path):
    m=model();coordinates=AffineCoordinates([10.,-3.],[2.,5.],[.3,-2.],[.01,1000.])
    bounds=[[8.4,11.6],[-7.,1.]]
    p=PhysicalModel.from_estimator(m,coordinates,bounds,limits())
    X=np.random.default_rng(13).uniform(np.asarray(bounds)[:,0],np.asarray(bounds)[:,1],(31,2))
    expected=coordinates.restore_outputs(m.predict(coordinates.inputs(X)))
    np.testing.assert_array_equal(p.predict(X),expected)
    record=p.to_dict();assert record['fitted_parameters']==3 and record['fixed_coordinate_parameters']==8
    frequency=next(x for x in record['parameters'] if x.get('source_parameter')==0)
    assert frequency['outputs']==[0,1] and frequency['units']==['-1','0','0','0','0','0','0']
    assert record['physical_acceptance']['status']=='not_met' # Added fixed coordinates must not disappear from complexity.
    assert record['physical_structure']['independent_parameters']==11
    relaxed=dict(limits(),shared_graph_size=80,independent_parameters=20,expanded_total_nodes=100,maximum_output_depth=12)
    assert PhysicalModel.from_estimator(m,coordinates,bounds,relaxed).acceptance['status']=='not_assessed'
    checked=PhysicalModel.from_estimator(m,coordinates,bounds,relaxed,training_data=(X,expected),validation_data=(X,expected))
    assert checked.acceptance['status']=='accepted'
    bad=PhysicalModel.from_estimator(m,coordinates,bounds,relaxed,training_data=(X,expected+1.),validation_data=(X,expected+1.))
    assert bad.acceptance['status']=='not_met'
    p.save(tmp_path/'physical.json');loaded=PhysicalModel.load(tmp_path/'physical.json')
    np.testing.assert_array_equal(p.predict(X),loaded.predict(X))
    with pytest.raises(ValueError,match='outside'):p.predict([[20.,0.]])
    record['constants'][0]+=1;(tmp_path/'tampered.json').write_text(json.dumps(record),encoding='utf-8')
    with pytest.raises(ValueError,match='mismatch'):PhysicalModel.load(tmp_path/'tampered.json')


def test_identity_delivery_preserves_structure_and_acceptance():
    m=model();p=PhysicalModel.from_estimator(m,physical_limits=limits())
    assert p.system==m.best_.system
    assert p.acceptance['status']=='accepted'
    assert p.to_dict()['fixed_coordinate_parameters']==0
    assert PhysicalModel.from_estimator(m).acceptance['status']=='not_assessed'


def test_training_only_coordinates_and_invalid_metadata():
    X=np.array([[1.,2.],[3.,2.],[5.,2.]]);Y=np.column_stack((X[:,0],X[:,0]*100))
    a=AffineCoordinates.from_training(X,Y)
    np.testing.assert_allclose(a.inputs(X).mean(axis=0),0.,atol=1e-15)
    assert a.x_offset==(3.,2.) and a.x_scale[1]==1.
    assert a.outputs([[1000.,100000.]]).shape==(1,2)
    assert a.x_offset==(3.,2.)
    with pytest.raises(ValueError):AffineCoordinates([0.],[0.],[0.],[1.])
    with pytest.raises(ValueError):a.input_bounds([[1.,1.],[0.,3.]])
    with pytest.raises(ValueError):PhysicalModel.from_estimator(model(),AffineCoordinates.identity(3,2))


def test_invalid_intermediate_stays_invalid_after_physical_restore():
    s=System((Vertex('x0'),Vertex('div',(0,0)),Vertex('c0',parameter=0),Vertex('mul',(1,2))),(3,))
    coords=AffineCoordinates([10.],[2.],[3.],[5.])
    restored,c,_,_=restore_graph(s,[0.],[ZERO],[ZERO],[ZERO],coords)
    assert np.isnan(restored.evaluate([[10.]],c)).all()
    np.testing.assert_array_equal(restored.evaluate([[12.]],c),[[3.]])
