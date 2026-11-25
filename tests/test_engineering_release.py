import csv
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from dimensr_mimo import PhysicalModel
from dimensr_mimo.engineering_cli import main
from dimensr_mimo.ranked_flow import RankedFlowSearch, shortlist


@pytest.mark.parametrize('count, expected', [(12, 12), (81, 50), (201, 61)])
def test_training_shortlist(count, expected):
    search = RankedFlowSearch.__new__(RankedFlowSearch)
    search.evaluator = SimpleNamespace(scales=np.ones(2))
    search.variance = np.ones(2)
    archive = {i: SimpleNamespace(errors=np.array([0, .4 + i / 1000]), error=(.4 + i / 1000) / 2)
               for i in range(count)}
    # Better on the weakest output even though its mean error is larger.
    balanced = SimpleNamespace(errors=np.array([.39, .39]), error=.39)
    archive[count-1] = balanced
    result = shortlist(archive, search.training_rank, .30, 50)
    assert len(result) == expected and result[0] is balanced
    assert result[-1] is archive[expected-2]


def test_engineering_csv_delivery(tmp_path, monkeypatch):
    x = np.linspace(-1, 1, 32)
    y = np.column_stack((2*x, -3*x))
    path = tmp_path/'train.csv'
    np.savetxt(path, np.column_stack((x, y)), delimiter=',', header='x,left,right', comments='')
    options = dict(max_evaluations=45, parallel_mode='serial', n_cpus=1,
                   population_size=4, structured_seeds=False, max_nodes=12, max_depth=4,
                   polish_candidates=1, sharing_attempts=4,
                   structure_limits=dict(parameters=16, nonlinear_parameters=16, nodes=32, depth=8),
                   stage_windows=dict(nsga2=1, residual=1, sharing=1),
                   stagnation_settings=dict(initialization_evaluations=4, initialization_seed_limit=2,
                       exploration_warmup=1, exploration_interval=1, exploration_patience=1, max_restarts=0))
    config = tmp_path/'config.json'
    config.write_text(json.dumps(options), encoding='utf-8')
    folder = tmp_path/'fit'
    monkeypatch.setattr(sys, 'argv', ['dimensr-mimo-engineering', '--data', str(path), '--features', 'x',
                        '--targets', 'left,right', '--output-dir', str(folder), '--config', str(config)])
    main()
    report = json.loads((folder/'results.json').read_text(encoding='utf-8'))
    physical = PhysicalModel.load(folder/'model.json')
    predictions = physical.predict(x[:, None])
    np.testing.assert_allclose(report['training_r2'], 1 - ((predictions - y)**2).mean(0)/y.var(0), atol=1e-12)
    pngs = list(folder.rglob('*.png'))
    assert pngs == [folder/'pareto'/'pareto.png']
    assert pngs[0].read_bytes().startswith(b'\x89PNG')
    with (folder/'pareto'/'candidates.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    delivered, = [row for row in rows if row['delivered'] == 'True']
    assert float(delivered['complexity']) == report['complexity']
    assert sum(row['in_selection_pool'] == 'True' for row in rows) == report['selection_before_compression']['candidates']
    log = (folder/'progress.log').read_text(encoding='utf-8')
    blocks = log.strip().split('\n\n')
    assert all(len(block.splitlines()) == 2 for block in blocks)
    assert all(block.count('混合复杂度') == 1 for block in blocks if '训练 R²' in block)
    assert f"混合复杂度 {report['complexity']:.2f}" in blocks[-1]
