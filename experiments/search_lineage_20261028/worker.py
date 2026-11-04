"""Train with lineage observation; no test or latent-label path."""
from dataclasses import asdict
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from dimensr_mimo import DimenSRMIMORegressor, SearchConfig, PhysicalModel
from experiments.compound_search_20261027 import cases
from experiments.native_api_20261026.run import read, write, arrays_hash
from experiments.typed_coverage_20261026 import delivery
from experiments.search_lineage_20261028.trace import Trace


def run(folder, job, config):
    torch.set_num_threads(1)
    started = time.monotonic()
    result = dict(job=job, status='failed', test_data_accessed=False)
    try:
        X, Y = cases.training_data(job['case'], 'train', job['seed'])
        V, Z = cases.training_data(job['case'], 'validation', job['seed'])
        spec = cases.design(job['case'])
        cfg = SearchConfig(**dict(config, random_state=job['seed'], sharing_strategy=job['method'],
            final_selection='precision', acceptance=spec['acceptance']))
        with (folder/'trace.jsonl').open('x', encoding='utf-8') as log, (folder/'events.jsonl').open('x', encoding='utf-8') as events:
            with Trace(log, events) as trace:
                m = DimenSRMIMORegressor(cfg, spec['features'], spec['targets'], spec['constants'], spec['feature_names']).fit(
                    X, Y, X_val=V, Y_val=Z, input_bounds=spec['bounds'], callback=trace.phase)
                trace.finish(m)
        m.save(folder/'model.json')
        PhysicalModel.from_estimator(m, input_bounds=spec['bounds'], physical_limits=spec['acceptance'],
            training_data=(X, Y), validation_data=(V, Z)).save(folder/'physical.json')
        write(folder/'pool.json', delivery.pool(read(folder/'model.json'), spec))
        result.update(status='complete', effective_config=asdict(cfg), diagnostics=[m.diagnostics_],
            data_sha256=arrays_hash(X, Y, V, Z), application_acceptance=m.post_selection_)
    except Exception as error:
        result['error'] = repr(error)
        (folder/'traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
    result['worker_wall_seconds'] = time.monotonic()-started
    write(folder/'result.json', result)


if __name__ == '__main__':
    run(Path(sys.argv[1]), read(sys.argv[2]), read(sys.argv[3]))
