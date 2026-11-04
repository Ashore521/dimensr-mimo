"""Bounded observer equivalence and restoration checks before the formal freeze."""
from dataclasses import replace
import io
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from dimensr_mimo import DimenSRMIMORegressor, SearchConfig, regressor
from dimensr_mimo.evaluation import Evaluator
from experiments.native_api_20261026.run import write, sha
from experiments.search_lineage_20261028.trace import Trace


def main():
    assert not (HERE/'SELFCHECK.json').exists(), 'Preserve the selfcheck.'
    torch.set_num_threads(1)
    X = np.linspace(.2, 1.2, 30)[:, None]
    Y = np.column_stack((np.sin(1.3*X[:, 0]), np.cos(1.3*X[:, 0])))
    cfg = SearchConfig(population_size=6, epochs=2, gp_generations=1, selection_generations=2,
        max_evaluations=100, min_candidates=5, max_nodes=9, constant_max_nfev=8,
        constant_restarts=1, domain_restarts=1, polish_candidates=1, polish_max_nfev=12, random_state=1601)
    original = Evaluator.__call__, regressor.offspring
    checked = []
    for method in ('random', 'matched'):
        config = replace(cfg, sharing_strategy=method)
        plain = DimenSRMIMORegressor(config).fit(X, Y, X_val=X[::3], Y_val=Y[::3], input_bounds=[[.2, 1.2]])
        stream = io.StringIO()
        with Trace(stream) as trace:
            observed = DimenSRMIMORegressor(config).fit(X, Y, X_val=X[::3], Y_val=Y[::3], input_bounds=[[.2, 1.2]], callback=trace.phase)
            trace.finish(observed)
        assert (Evaluator.__call__, regressor.offspring) == original
        assert plain.search_fingerprint_ == observed.search_fingerprint_
        assert plain.candidates_ == observed.candidates_
        assert plain.best_.to_dict() == observed.best_.to_dict()
        rows = [json.loads(line) for line in stream.getvalue().splitlines()]
        assert [r['serial'] for r in rows] == list(range(1, len(rows)+1))
        assert len([r for r in rows if r['kind'] == 'variation']) == sum(observed.diagnostics_['operations'].values())
        assert len([r for r in rows if r['kind'] == 'evaluation']) == observed.diagnostics_['proposals']
        checked.append(dict(method=method, fingerprint=plain.search_fingerprint_, trace_rows=len(rows),
            evaluations=plain.diagnostics_['evaluations'], candidates=len(plain.candidates_)))
    try:
        with Trace(io.StringIO()):
            raise RuntimeError('deliberate restoration check')
    except RuntimeError:
        pass
    assert (Evaluator.__call__, regressor.offspring) == original
    write(HERE/'SELFCHECK.json', dict(status='passed', rows=checked, exception_restores=True,
        test_data_accessed=False, formal_matrix_runs=0, files={p.name:sha(p) for p in (HERE/'trace.py', HERE/'selfcheck.py')}))
    print('Observer equivalence passed for random/matched; patches restore after success and exception.')


if __name__ == '__main__':
    main()
