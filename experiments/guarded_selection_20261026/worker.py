"""One bounded search or validation-only derivation; never requests test data."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from dimensr_mimo import DimenSRMIMORegressor
from dimensr_mimo._single import DimenSRRegressor
from dimensr_mimo.graph import System
from experiments.guarded_selection_20261026 import cases, selector
from experiments.second_round_20261026.worker import selection_record

base_worker = cases.previous('worker', cases=cases)
base_worker.generate = cases.training_data
write = base_worker.write
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def predictions(model, X, V):
    return {name: hashlib.sha256(model.predict(A).astype('<f8').tobytes()).hexdigest()
            for name, A in [('train', X), ('validation', V)]}


def run(folder, job, config):
    torch.set_num_threads(1)
    if (folder / 'result.json').exists():
        raise RuntimeError('Refusing to overwrite an existing result.')
    started = time.monotonic()
    result = dict(job=job, status='failed', models=[], diagnostics=[], prediction_hashes=[], audits=[])
    method = job['method']
    try:
        if method not in ('original', 'no_penalty', 'guarded', 'independent'):
            raise ValueError('Unknown method.')
        X, _ = cases.training_data(job['case'], 'train', job['seed'])
        V, _ = cases.training_data(job['case'], 'validation', job['seed'])
        if method in ('original', 'independent'):
            base_worker.run(folder, job, config)
            result = read(folder / 'result.json')
            if result['status'] != 'complete':
                return
            result['search_reused'] = False
        else:
            source = folder.parent / f"{job['case']}_s{job['seed']}_original"
            result['search_reused'] = True
            result['source_search'] = source.name
            if read(source / 'resources.json')['status'] != 'complete':
                result['status'] = 'source_search_failed'
                write(folder / 'result.json', result)
                return
            checkpoint = read(source / 'CELL_FROZEN.json')
            for name, expected in checkpoint['files'].items():
                if sha(source / name) != expected:
                    raise RuntimeError('Source search checkpoint drift.')
            original = read(source / 'model.json')
            data = selector.derive(original, method)
            graph = System.from_dict(data['best']['system'])
            before = {name: hashlib.sha256(graph.evaluate(A, data['best']['constants']).astype('<f8').tobytes()).hexdigest()
                      for name, A in [('train', X), ('validation', V)]}
            write(folder / 'model.json', data)
            result.update(status='complete', models=['model.json'], prediction_hashes=[before],
                          diagnostics=[data['diagnostics']], source_model_sha256=sha(source / 'model.json'),
                          fit_wall_seconds=0., fit_cpu_seconds=0.)
            write(folder / 'selection.json', data['post_selection'])
            event = dict(phase='post_selection', method=method, source_search=source.name,
                         candidate_index=data['post_selection']['candidate_index'], new_evaluations=0,
                         output=None, worker_elapsed=time.monotonic() - started, monotonic=time.monotonic())
            (folder / 'events.jsonl').write_text(json.dumps(event) + '\n', encoding='utf-8')
        model_type = DimenSRRegressor if method == 'independent' else DimenSRMIMORegressor
        result['selections'], reload_hashes, audits = [], [], []
        for filename in result['models']:
            data = read(folder / filename)
            record = selection_record(data)
            if method != 'independent':
                record['rule'] = data.get('post_selection', selector.choose(data['candidates'], 'original'))
            result['selections'].append(dict(model=filename, **record))
            model = model_type.load(folder / filename)
            reload_hashes.append(predictions(model, X, V))
            audits.append(model.audit_domain(input_bounds=cases.design(job['case'])['bounds']))
        if method in ('no_penalty', 'guarded'):
            result['audits'] = audits
        result['reload_predictions_equal'] = reload_hashes == result['prediction_hashes']
        result['reload_audits_equal'] = audits == result['audits']
        if not result['reload_predictions_equal'] or not result['reload_audits_equal']:
            raise RuntimeError('Save/load verification failed.')
        result['worker_total_seconds'] = time.monotonic() - started
    except Exception as error:
        result.update(status='failed', error=repr(error))
        (folder / 'traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
    write(folder / 'result.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder'); parser.add_argument('job'); parser.add_argument('config')
    args = parser.parse_args()
    run(Path(args.folder), read(Path(args.job)), read(Path(args.config)))
