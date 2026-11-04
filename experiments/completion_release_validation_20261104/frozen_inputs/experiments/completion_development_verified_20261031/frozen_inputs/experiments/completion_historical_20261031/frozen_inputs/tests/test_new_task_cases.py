import numpy as np
import pytest

from dimensr_mimo import SearchConfig,audit
from dimensr_mimo.generation import SystemGrammar
from dimensr_mimo.graph import in_bounds
from dimensr_mimo._single.units import unit
from experiments.new_task_validation_20261030 import cases
from experiments.precision_repair_20261028.study import configuration


@pytest.mark.parametrize('family',cases.FAMILIES)
def test_new_truth_pairs_are_unit_legal_representable_and_safe(family):
    name=family+'_n0';spec=cases.design(name);graph,constants=cases.truth_graph(name)
    cfg=SearchConfig(**dict(configuration(),final_selection='precision',acceptance=spec['acceptance']))
    audit.units(graph,spec['features'],spec['targets'],spec['constants'])
    assert in_bounds(graph,cfg)
    shape=audit.complexity(graph)
    assert all(shape[k]<=spec['acceptance'][k] for k in ('shared_graph_size','independent_parameters','expanded_total_nodes','maximum_output_depth'))
    grammar=SystemGrammar(*[tuple(unit(u) for u in spec[k]) for k in ('features','targets','constants')],cfg)
    for j,r in enumerate(graph.roots):
        tree,_=graph.expanded(r,constants);grammar.plain[j].trace(tree)
    X=np.array([[.2,.2],[.2,1.4],[1.4,.2],[1.4,1.4],[.7,.8]])
    np.testing.assert_allclose(graph.evaluate(X,constants),cases.latent(name,X),rtol=1e-14,atol=1e-14)
    assert all(row['status']=='enclosed' for row in graph.audit(constants,spec['bounds'],128,None))


def test_new_matrix_and_test_label_isolation(monkeypatch,tmp_path):
    monkeypatch.setattr(cases,'HERE',tmp_path)
    assert len(cases.CASES)==8 and len(cases.SEEDS)==2
    for name in cases.CASES:
        spec=cases.design(name)
        assert spec['acceptance']['nrmse']==([1e-4]*2 if name.endswith('n0') else [.03]*2)
        assert spec['clean_test_nrmse']==([1e-4]*2 if name.endswith('n0') else [.02]*2)
        for seed in cases.SEEDS:
            with pytest.raises(ValueError,match='Training'):
                cases.training_data(name,'test',seed)
            with pytest.raises(RuntimeError,match='seal'):
                cases.generate(name,'test',seed)
    with pytest.raises(ValueError,match='seeds'):
        cases.inputs(cases.CASES[0],'train',701)


def test_new_training_generation_is_deterministic_with_separate_splits():
    name='bilinear_saturation_n1';seed=cases.SEEDS[0]
    X,Y=cases.training_data(name,'train',seed)
    A,B=cases.training_data(name,'validation',seed)
    repeat=cases.training_data(name,'train',seed)
    np.testing.assert_array_equal(X,repeat[0]);np.testing.assert_array_equal(Y,repeat[1])
    assert X.shape==Y.shape==(96,2) and A.shape==B.shape==(48,2)
    assert np.isfinite(Y).all() and np.all(Y.std(axis=0)>0)
    assert not np.array_equal(X[:48],A)
    assert not np.array_equal(X,cases.inputs(name,'train',cases.SEEDS[1]))
    assert not np.array_equal(Y,cases.latent(name,X))
