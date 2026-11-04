from dataclasses import replace
import numpy as np
import pytest
from dimensr_mimo import SearchConfig
from dimensr_mimo.generation import SystemGrammar
from dimensr_mimo._single.units import ZERO,unit
from experiments.typed_coverage_20261026 import cases,schedule,audit
from experiments.shared_discovery_20261026.seeds import typed_seeds
from experiments.unit_composition_20261026 import aligned,structure


def setup(case):
    spec=cases.design(case)
    cfg=SearchConfig(population_size=24,max_evaluations=1200,max_nodes=15,max_depth=6,
        operators=('add','sub','mul','div','square','sin','cos','sqrt','log'))
    g=SystemGrammar(tuple(map(unit,spec['features'])),tuple(map(unit,spec['targets'])),tuple(map(unit,spec['constants'])),cfg)
    return spec,cfg,g


def test_aligned_families_preserve_vocabulary_rng_and_budget():
    spec,cfg,g=setup('dimensional_frequency');rng=np.random.default_rng(13)
    old=[list(typed_seeds(x,rng)) for x in g.plain];state=rng.bit_generator.state
    rng=np.random.default_rng(13);generate=aligned.generator(cfg,spec)
    rows=[list(generate(x,rng)) for x in g.plain]
    assert rng.bit_generator.state==state
    assert len(rows[0])==len(rows[1])<=schedule.typed_limit(cfg,2)
    for j in range(2):assert set(rows[j]).issubset(set(old[j]))
    for i in range(len(rows[0])):
        fam=[schedule.category(r[i])[0] for r in rows]
        if fam[0] in ('sin','cos'):assert fam[0]==fam[1]
    for j in range(2):
        assert {k for t in rows[j] if schedule.category(t)[0]=='sin' for k in schedule.category(t)[1]}=={0,1}


def test_missing_typed_catalog_keeps_structured_fallback():
    _,_,g=setup('products');rows=[list(typed_seeds(x,np.random.default_rng(3))) for x in g.plain]
    assert aligned.align(rows,12)==[schedule.balanced_prefix(r,12) for r in rows]


def test_unit_compositions_are_legal_and_include_parameterized_nonlinearity():
    # Inspect candidate syntax using artificial responses, never formal test data.
    for case in ('dimensional_frequency','partial','boundary'):
        spec,cfg,g=setup(case);bounds=np.asarray(spec['bounds']);X=np.random.default_rng(8).uniform(bounds[:,0],bounds[:,1],(24,len(bounds)))
        for j,grammar in enumerate(g.plain):
            y=np.sin(X[:,0])+X[:,1]**2
            seeds,stats=structure.structured_seeds(grammar,X,y,12)
            assert stats['retained_features']<=512 and len(seeds)<=12
            for tree,c in seeds:
                grammar.trace(tree)
                assert tree.units==unit(spec['targets'][j]) and np.isfinite(tree.evaluate(X,c)).all()
                assert tree.size<=cfg.max_nodes and tree.depth<=cfg.max_depth
            if case=='dimensional_frequency':
                assert any(any(n.token in ('sin','cos') for _,n in t.walk()) for t,c in seeds)


@pytest.mark.parametrize('method',['balanced','aligned','normalized','combined'])
def test_intervention_worker_and_verifier_preserve_protocol(tmp_path,monkeypatch,method):
    from dataclasses import asdict
    from dimensr_mimo.graph import System
    from experiments.unit_composition_20261026 import worker,study,verify
    from experiments.typed_coverage_20261026 import delivery
    spec,cfg,_=setup('dimensional_frequency')
    cfg=replace(cfg,population_size=6,epochs=1,gp_generations=1,selection_generations=2,
        max_evaluations=80,min_candidates=3,polish_candidates=0,constant_max_nfev=4,constant_restarts=1)
    def data(name,split,seed):
        X=np.random.default_rng({'train':1,'validation':2,'test':3}[split]).uniform(-.8,.8,(20,2))
        return X,np.column_stack((X[:,0]+X[:,1],X[:,0]-X[:,1]))
    monkeypatch.setattr(cases,'design',lambda name:spec)
    monkeypatch.setattr(cases,'training_data',data);monkeypatch.setattr(cases,'generate',data)
    monkeypatch.setattr(verify.prior,'HERE',tmp_path)
    job=dict(case='dimensional_frequency',seed=601,method=method,seconds=60)
    folder=tmp_path/'runs'/study.name(job);folder.mkdir(parents=True)
    worker.write(folder/'job.json',job);worker.run(folder,job,asdict(cfg))
    result=worker.read(folder/'result.json');assert result['status']=='complete',result
    assert result['job']==job
    bundle=worker.read(folder/'delivery.json');T,U=data('','test',0);_,Y=data('','train',0)
    graph=System.from_dict(bundle['system']);e=np.sqrt(np.mean((graph.evaluate(T,[p['value'] for p in bundle['parameters']])-U)**2,axis=0))/Y.std(axis=0)
    score=dict(structure=bundle['structure'],nrmse=e.tolist(),test_accepted=delivery.accepted(e,bundle['structure'],spec['acceptance']),mechanism=study.mechanism(folder))
    assert verify.prior.verify_cell(job,score)['verified']
    assert len(study.matrix())==48 and study.LIMIT==3480
