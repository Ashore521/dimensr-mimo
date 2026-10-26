"""Reuse the frozen fitting paths, then pin selections and reload evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.second_round_20261026 import cases

previous_worker = cases.previous('worker', cases=cases)
previous_worker.generate = cases.training_data
write = previous_worker.write


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def selection_record(data):
    candidates = data.get('candidates', data.get('pareto_front'))
    selected = data['best']
    matches = [i for i, candidate in enumerate(candidates)
               if all(candidate.get(key) == value for key, value in selected.items())]
    if len(matches) != 1:
        raise RuntimeError('Selected model is not uniquely identified in its frozen candidate pool.')
    return dict(candidate_index=matches[0], candidate_sha256=digest(candidates[matches[0]]),
                best_sha256=digest(selected), pool_sha256=digest(candidates))


def run(folder, job, config):
    if job['method'] not in ('original', 'proposed', 'independent'):
        raise ValueError('Unknown second-round method.')
    if (folder / 'result.json').exists():
        raise RuntimeError('Refusing to overwrite an existing fit.')
    previous_worker.run(folder, job, config)
    result = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
    if result['status'] != 'complete':
        return
    try:
        X, _ = cases.training_data(job['case'], 'train', job['seed'])
        V, _ = cases.training_data(job['case'], 'validation', job['seed'])
        model_class = previous_worker.DimenSRRegressor if job['method'] == 'independent' else previous_worker.DimenSRMIMORegressor
        result['selections'], reloaded_hashes, reloaded_audits = [], [], []
        for filename in result['models']:
            path = folder / filename
            data = json.loads(path.read_text(encoding='utf-8'))
            result['selections'].append(dict(model=filename, **selection_record(data)))
            model = model_class.load(path)
            reloaded_hashes.append({name: hashlib.sha256(model.predict(A).astype('<f8').tobytes()).hexdigest()
                                    for name, A in [('train', X), ('validation', V)]})
            reloaded_audits.append(model.audit_domain(input_bounds=cases.design(job['case'])['bounds']))
        result['reload_predictions_equal'] = reloaded_hashes == result['prediction_hashes']
        result['reload_audits_equal'] = reloaded_audits == result['audits']
        if not result['reload_predictions_equal'] or not result['reload_audits_equal']:
            raise RuntimeError('Save/reload verification failed.')
    except Exception as error:
        result.update(status='failed_reload_verification', error=repr(error))
        (folder / 'reload_traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
    write(folder / 'result.json', result)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('folder'); p.add_argument('job'); p.add_argument('config')
    a = p.parse_args()
    run(Path(a.folder), json.loads(Path(a.job).read_text(encoding='utf-8-sig')),
        json.loads(Path(a.config).read_text(encoding='utf-8-sig')))
