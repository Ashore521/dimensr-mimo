from dataclasses import replace
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dimensr_mimo import SearchConfig
from dimensr_mimo.graph import System,Vertex,Builder
from dimensr_mimo.generation import SystemGrammar
from dimensr_mimo.evaluation import Evaluator,Candidate
from dimensr_mimo.genetics import offspring
from dimensr_mimo._single.units import ZERO,unit
from experiments.shared_discovery_20261026.seeds import typed_seeds as appended
from experiments.typed_coverage_20261026 import schedule,audit


def config():
    return SearchConfig(population_size=24,max_evaluations=1200,max_nodes=15,max_depth=6,
        operators=('add','sub','mul','div','square','sin','cos','sqrt','log'),constant_restarts=1,constant_max_nfev=25)


@pytest.mark.parametrize('features',[8,11])
def test_high_dimensional_seed_coverage_with_same_budget(features):
    cfg=config();g=SystemGrammar((ZERO,)*features,(ZERO,)*2,(ZERO,),cfg).plain[0]
    old=list(appended(g,np.random.default_rng(401)));new=list(schedule.generator(cfg,2)(g,np.random.default_rng(401)))
    assert len(new)<=schedule.typed_limit(cfg,2)==168
    assert not any(schedule.category(t)[0] in ('sin','cos') for t in old[:180])
    assert set(new).issubset(old) and len(new)==len(set(new))
    for family in ('sin','cos','square'):
        assert {i for tree in new if schedule.category(tree)[0]==family for i in schedule.category(tree)[1]}==set(range(features))
    for tree in new:g.trace(tree)


def test_small_limits_unique_deterministic_and_structure_reserve():
    cfg=config();g=SystemGrammar((ZERO,)*2,(ZERO,)*2,(ZERO,),cfg).plain[0]
    trees=list(appended(g,np.random.default_rng(4)))
    assert schedule.balanced_prefix(trees,0)==[]
    assert len(schedule.balanced_prefix(trees+trees,10000))==len(set(trees))
    assert schedule.balanced_prefix(trees,20)==schedule.balanced_prefix(trees,20)
    assert schedule.typed_limit(replace(cfg,structured_seeds=False),2)==180
    with pytest.raises(ValueError):schedule.balanced_prefix(trees,-1)


def dimensional_system():
    L=unit([1,0,0,0,0,0,0]);V=unit([1,0,-1,0,0,0,0]);K=unit([-1,0,0,0,0,0,0])
    # Same inverse-length parameter feeds distinct trigonometric roots; their
    # amplitudes have distinct physical units and must stay independent.
    s=System((Vertex('c0',units=K,parameter=0),Vertex('x0',units=L),Vertex('mul',(0,1)),Vertex('sin',(2,)),
        Vertex('c1',units=L,parameter=1),Vertex('mul',(4,3),L),Vertex('x1',units=L),Vertex('mul',(0,6)),
        Vertex('cos',(7,)),Vertex('c2',units=V,parameter=2),Vertex('mul',(9,8),V)),(5,10))
    return s,(L,L),(L,V),(K,L,V)


def test_dimensioned_shared_frequency_fit_replay_and_variation():
    s,features,targets,constants=dimensional_system();cfg=config()
    X=np.random.default_rng(6).uniform(-.8,.8,(64,2));Y=s.evaluate(X,[1.7,.8,.6])
    evaluator=Evaluator(X,Y,cfg,input_bounds=[[-.8,.8]]*2);candidate=evaluator(s,np.ones(3))
    assert candidate is not None and max(candidate.errors)<1e-7
    np.testing.assert_allclose(candidate.constants,[1.7,.8,.6],atol=1e-6)
    restored=System.from_dict(s.to_dict());np.testing.assert_array_equal(restored.evaluate(X,candidate.constants),s.evaluate(X,candidate.constants))
    result=audit.units(restored,features,targets,constants)
    assert result['parameter_units'][0]==['-1','0','0','0','0','0','0']
    g=SystemGrammar(features,targets,constants,cfg);g.trace(s)
    for operator in ('mutation','crossover','share','unshare','bind','unbind','shared_mutation'):
        rng=np.random.default_rng(9)
        for _ in range(8):
            child,_,_=offspring(candidate,candidate,g,rng,operator)
            assert audit.units(child,features,targets,constants)['status']=='passed'


def test_unit_audit_rejects_changed_metadata_and_cross_unit_binding():
    s,features,targets,constants=dimensional_system()
    with pytest.raises(ValueError,match='Input dimension'):audit.units(s,(ZERO,ZERO),targets,constants)
    with pytest.raises(ValueError,match='Output dimension'):audit.units(s,features,(ZERO,ZERO),constants)
    b=Builder(s,[1.7,.8,.6]);b.nodes[9]=replace(b.nodes[9],parameter=1)
    with pytest.raises(ValueError,match='identical units'):b.finish()


def test_shared_and_expanded_complexity_are_distinct():
    s=System((Vertex('c0',parameter=0),Vertex('x0'),Vertex('mul',(0,1)),Vertex('sin',(2,)),Vertex('cos',(2,))),(3,4))
    report=audit.complexity(s)
    assert report['shared_dag_vertices']==5 and report['shared_dag_operations']==3
    assert report['expanded_total_nodes']==8 and report['expanded_total_operations']==4
    assert report['maximum_output_depth']==3 and report['independent_parameters']==1
    assert [r['independent_parameters'] for r in report['outputs']]==[1,1]


def test_observer_proves_actual_evaluation_without_changing_control(tmp_path,monkeypatch):
    from dimensr_mimo import DimenSRMIMORegressor,regressor
    from experiments.typed_coverage_20261026.observe import Observation
    cfg=replace(config(),population_size=6,epochs=1,gp_generations=1,selection_generations=2,
        max_evaluations=160,min_candidates=3,structured_seeds=False,polish_candidates=0,constant_max_nfev=5)
    X=np.random.default_rng(991).uniform(-.5,.5,(28,8));Y=np.column_stack((X[:,0]+X[:,7],X[:,1]-X[:,6]))
    spec=dict(features=[ZERO]*8,targets=[ZERO]*2,constants=[ZERO],bounds=[[-.5,.5]]*8)
    monkeypatch.setattr(regressor,'typed_seeds',appended)
    original=DimenSRMIMORegressor(cfg).fit(X,Y,X_val=X,Y_val=Y,input_bounds=spec['bounds'])
    for balanced in (False,True):
        with Observation(tmp_path/f'{balanced}.jsonl',cfg,spec,balanced) as observed:
            model=DimenSRMIMORegressor(cfg).fit(X,Y,X_val=X,Y_val=Y,input_bounds=spec['bounds'],callback=observed.event)
        report=observed.summary()
        assert report['unit_checks']['evaluator_proposals']==model.diagnostics_['proposals']
        if not balanced:assert original.search_fingerprint_==model.search_fingerprint_
        for family in ('sin','cos'):
            actual=report['coverage'].get('output0/typed/'+family,{}).get('new_evaluated',0)
            assert (actual>0)==balanced


def test_physical_delivery_feasibility_and_test_access_boundary():
    from experiments.typed_coverage_20261026 import cases,delivery
    s,features,targets,constants=dimensional_system()
    spec=cases.design('dimensional_frequency')
    # The constructed graph's terminal catalogue is deliberately ordered differently.
    spec.update(features=features,targets=targets,constants=constants)
    X,Y=cases.training_data('dimensional_frequency','train',601)
    V,Z=cases.training_data('dimensional_frequency','validation',601)
    result=delivery.record(s,[1.7,.8,.6],spec,X,Y,V,Z)
    assert result['development_accepted']
    assert result['parameters'][0]['units']==['-1','0','0','0','0','0','0']
    assert result['input_transform']=='identity in declared SI units'
    assert not delivery.accepted([1e-5,1.01e-4],result['structure'],spec['acceptance'])
    assert not delivery.accepted([0,0],dict(result['structure'],independent_parameters=7),spec['acceptance'])
    assert not delivery.accepted([float('nan'),0],result['structure'],spec['acceptance'])
    with pytest.raises(ValueError,match='test'):cases.training_data('products','test',601)
    for case in cases.CASES:
        metadata=cases.design(case);A,B=cases.training_data(case,'train',601)
        assert A.shape==(96,len(metadata['features'])) and B.shape==(96,2)


@pytest.mark.parametrize('method',['prefix','balanced','independent'])
def test_dimensional_worker_and_independent_verifier_on_artificial_data(tmp_path,monkeypatch,method):
    from dataclasses import asdict
    from experiments.typed_coverage_20261026 import cases,worker,study,verify,delivery
    cfg=replace(config(),population_size=6,epochs=1,gp_generations=1,selection_generations=2,
        max_evaluations=80,min_candidates=3,structured_seeds=False,polish_candidates=0,constant_max_nfev=4)
    spec=cases.design('products');spec.update(features=[ZERO,ZERO],targets=[ZERO,ZERO],constants=[ZERO],bounds=[[-.8,.8]]*2)
    def data(name,split,seed):
        rng=np.random.default_rng({'train':1,'validation':2,'test':3}[split]);X=rng.uniform(-.8,.8,(20,2))
        return X,np.column_stack((X[:,0]+X[:,1],X[:,0]-X[:,1]))
    monkeypatch.setattr(cases,'design',lambda name:spec)
    monkeypatch.setattr(cases,'training_data',data);monkeypatch.setattr(cases,'generate',data)
    monkeypatch.setattr(worker.base,'generate',data)
    monkeypatch.setattr(worker.base,'design',lambda name:spec)
    monkeypatch.setattr(verify,'HERE',tmp_path)
    job=dict(case='products',seed=601,method=method,seconds=60)
    folder=tmp_path/'runs'/study.name(job);folder.mkdir(parents=True)
    worker.write(folder/'job.json',job);worker.run(folder,job,asdict(cfg))
    result=worker.read(folder/'result.json');assert result['status']=='complete',result
    bundle=worker.read(folder/'delivery.json');T,U=data('', 'test', 0);_,Y=data('', 'train',0)
    system=System.from_dict(bundle['system']);values=[p['value'] for p in bundle['parameters']]
    e=np.sqrt(np.mean((system.evaluate(T,values)-U)**2,axis=0))/Y.std(axis=0)
    score=dict(structure=bundle['structure'],nrmse=e.tolist(),test_accepted=delivery.accepted(e,bundle['structure'],spec['acceptance']),mechanism=study.mechanism(folder))
    assert verify.verify_cell(job,score)['verified']
