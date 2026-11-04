"""Read sealed JSON only; never fit, score data, or replace original verdicts."""
import hashlib
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from experiments.compositional_completion_20261031.verification import normalized


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def differences(a, b, path=''):
    if type(a) is not type(b):
        return [dict(path=path, kind='type', current=a, original=b)]
    if isinstance(a, dict):
        result = []
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                result.append(dict(path=path+'/'+key, kind='missing', current=a.get(key), original=b.get(key)))
            else:
                result.extend(differences(a[key], b[key], path+'/'+key))
        return result
    if isinstance(a, list):
        if len(a) != len(b):
            return [dict(path=path, kind='length', current=len(a), original=len(b))]
        return [row for i, (x, y) in enumerate(zip(a, b)) for row in differences(x, y, path+'/'+str(i))]
    return [] if a == b else [dict(path=path, kind='value', current=a, original=b)]


def timing_path(path):
    parts = path.split('/')
    return (len(parts) == 4 and parts[1] == 'history' and parts[2].isdigit()
            and parts[3] in {'epoch_allowance_seconds', 'neural_until_seconds', 'gp_until_seconds'})


def run():
    path = HERE/'DIAGNOSIS.json'
    if path.exists():
        raise RuntimeError('Keep existing diagnosis.')
    current = ROOT/'experiments/completion_historical_20261031'
    original = ROOT/'experiments/precision_repair_20261028'
    freeze, seal = read(current/'FREEZE.json'), read(current/'ALL_MODELS_FROZEN.json')
    assert seal['freeze_sha256'] == sha(current/'FREEZE.json')
    for name, digest in freeze['files'].items():
        assert sha(ROOT/name) == sha(current/'frozen_inputs'/name) == digest, name
    for name, digest in seal['files'].items():
        assert sha(current/name) == digest, name
    old_seal = read(original/'ALL_MODELS_FROZEN.json')
    for name, digest in old_seal['files'].items():
        assert sha(original/name) == digest, name
    scores, old_scores = read(current/'SCORES.json')['rows'], read(original/'SCORES.json')['rows']
    rows = []
    for job in freeze['jobs']:
        if job['method'] != 'base':
            continue
        prefix = job['case']+'_s'+str(job['seed'])
        changes = {}
        for filename in ('model.json', 'result.json', 'delivery.json', 'pool.json'):
            a, b = current/'runs'/(prefix+'_base')/filename, original/'runs'/(prefix+'_repair')/filename
            changes[filename] = differences(normalized(read(a)), normalized(read(b)))
        def score_for(items, method):
            found = [r for r in items if r['job']['case'] == job['case'] and r['job']['seed'] == job['seed'] and r['job']['method'] == method]
            assert len(found) == 1
            return normalized({k:v for k,v in found[0].items() if k not in {'job', 'resources'}})
        changes['score'] = differences(score_for(scores, 'base'), score_for(old_scores, 'repair'))
        remaining = {name:[r for r in items if name != 'model.json' or not timing_path(r['path'])]
                     for name, items in changes.items()}
        rows.append(dict(job=job, differences=changes, differences_outside_three_history_timing_fields=remaining,
                         only_declared_history_timing_differs=not any(remaining.values())))
    assert len(rows) == 17
    record = dict(status='readonly_diagnosis_complete', checked_unix=time.time(), rows=rows,
                  expected_pairs=17, pairs_only_history_timing=sum(r['only_declared_history_timing_differs'] for r in rows),
                  original_verification=read(current/'VERIFICATION.json')['status'],
                  original_technical_pass=read(current/'SUMMARY.json')['technical_pass'],
                  original_adoption_pass=read(current/'SUMMARY.json')['adoption_pass'],
                  source_sha256={n:sha(current/n) for n in ('FREEZE.json', 'ALL_MODELS_FROZEN.json', 'SCORES.json', 'VERIFICATION.json', 'SUMMARY.json', 'EXECUTION.json')},
                  diagnostic_sha256=sha(Path(__file__)), no_fit_search_or_rescoring=True,
                  permits_development=False)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in record.items() if k not in {'rows', 'source_sha256'}}, ensure_ascii=False))


if __name__ == '__main__':
    run()
