from dataclasses import asdict
import numpy as np
import pytest

from dimensr_mimo import SearchConfig, PhysicalModel
from experiments.compound_search_20261027 import cases, study, worker


def test_compound_matrix_data_isolation_and_denominators():
    jobs = study.matrix()
    assert len(jobs) == 38 and len({study.name(j) for j in jobs}) == 38
    assert sum(j['method'] == 'atoms' for j in jobs) == 4
    assert sum(j['method'] == 'dimensional' for j in jobs) == 17
    assert sum(j['method'] == 'compound' for j in jobs) == 17
    for case in cases.CASES:
        for seed in cases.seeds(case):
            with pytest.raises(ValueError, match='test data'):
                cases.training_data(case, 'test', seed)
    cfg = study.configuration()
    assert cfg['reference_budget'] == 'expanded' and cfg['time_allocation'] == 'phased'
    assert study.LIMIT == 3120


def test_compound_worker_delivery_with_small_training_only_case(tmp_path):
    cfg = SearchConfig(population_size=6, epochs=1, gp_generations=1, selection_generations=2,
        max_evaluations=80, min_candidates=2, polish_candidates=0, constant_max_nfev=5,
        constant_restarts=1, max_nodes=9, max_depth=5, operators=('add', 'mul', 'square'))
    job = dict(case='products', seed=601, method='compound', seconds=60)
    worker.run(tmp_path, job, asdict(cfg))
    result = worker.read(tmp_path/'result.json')
    assert result['status'] == 'complete', result
    assert result['effective_config']['initialization'] == 'compound'
    assert result['test_scores_computed'] is False
    assert all(row['retained_features'] <= 512 for row in result['diagnostics'][0]['initialization_diagnostics'])
    physical = PhysicalModel.load(tmp_path/'physical.json')
    model = worker.DimenSRMIMORegressor.load(tmp_path/'model.json')
    X, _ = cases.training_data('products', 'validation', 601)
    np.testing.assert_array_equal(physical.predict(X), model.predict(X))
