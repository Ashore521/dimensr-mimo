"""Reuse the raw-graph verifier and report every matched/random paired output."""
import importlib.util
from pathlib import Path
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from experiments.matched_sharing_20261028 import study
from experiments.compound_search_20261027 import cases
from experiments.native_api_20261026.run import read, write, sha
from experiments.native_api_20261026.verify import require

spec = importlib.util.spec_from_file_location('_matched_raw_verifier', HERE.parent/'compound_search_20261027/evaluate.py')
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
base.HERE, base.study = HERE, study
evaluate = base.evaluate


def summarize(scores, verified):
    index = {(r['job']['case'], r['job']['seed'], r['job']['method']):r for r in scores}
    methods, pairs = {}, []
    for method in study.METHODS:
        rows = [r for r in scores if r['job']['method'] == method]
        valid = [r for r in rows if r['status'] == 'complete']
        strata = {}
        for group, predicate in [('original_n0', lambda c:c.endswith('_n0')), ('original_n1', lambda c:c.endswith('_n1')),
                                 ('variants', lambda c:c in cases.VARIANTS), ('regressions', lambda c:c in cases.REGRESSIONS)]:
            group_rows = [r for r in rows if predicate(r['job']['case'])]
            group_valid = [r for r in valid if predicate(r['job']['case'])]
            strata[group] = dict(expected=len(group_rows), complete=len(group_valid),
                both_accepted=sum(r['both_accepted'] for r in group_valid))
        methods[method] = dict(expected=len(rows), complete=len(valid), both_accepted=sum(r['both_accepted'] for r in valid),
            mean_clean_nrmse=float(np.mean([r['mean_clean_nrmse'] for r in valid])) if len(valid) == len(rows) else None,
            mean_shared_graph_size=float(np.mean([r['structure']['shared_graph_size'] for r in valid])) if len(valid) == len(rows) else None,
            strata=strata)
    for case in cases.CASES:
        for seed in cases.seeds(case):
            a, b = index[case, seed, 'matched'], index[case, seed, 'random']
            pair = dict(case=case, seed=seed, available=a['status'] == b['status'] == 'complete')
            if pair['available']:
                pair.update(protected=(np.asarray(a['clean_nrmse']) <= np.asarray(b['clean_nrmse'])*1.1 + 1e-8).tolist(),
                    difference=(np.asarray(a['clean_nrmse']) - b['clean_nrmse']).tolist(),
                    candidate_accepted=a['both_accepted'], control_accepted=b['both_accepted'])
            pairs.append(pair)
    totals = {k:sum(v.get('matched_share', {}).get(k, 0) for v in verified) for k in ('operations', 'changed', 'novel', 'successful')}
    technical = len(verified) == len(study.matrix()) and all(v.get('verified', False) for v in verified)
    protected = sum(sum(p.get('protected', [])) for p in pairs)
    candidate, control = methods['matched'], methods['random']
    adoption = (technical and totals['novel'] > 0 and totals['successful'] > 0 and protected == 34
        and candidate['both_accepted'] >= control['both_accepted'] and candidate['mean_clean_nrmse'] <= control['mean_clean_nrmse'])
    return dict(methods=methods, pairs=pairs, expected_outputs=34, protected_outputs=protected,
        technical_pass=technical, matched_share=totals, adoption_pass=adoption, production_default_changed=False,
        claim_scope='Declared historical development regression; no blind or engineering validation.')


def verify():
    if (HERE/'VERIFICATION.json').exists():
        raise RuntimeError('Retain verification.')
    frozen, sealed = base.guards(); scores = read(HERE/'SCORES.json')
    require(scores['all_models_sha256'] == sha(HERE/'ALL_MODELS_FROZEN.json'), 'Scores chain drift.')
    require(sealed['frozen_unix'] <= scores['scored_unix'], 'Scoring preceded model freeze.')
    require(len(scores['rows']) == len(frozen['jobs']), 'Incomplete score denominator.')
    rows = []
    for job, score in zip(frozen['jobs'], scores['rows']):
        require(time.time() < study.budget(), 'Verification budget expired.')
        counts = {}
        try:
            result_path = HERE/'runs'/study.name(job)/'result.json'
            result = read(result_path) if result_path.exists() else {}
            diagnostics = result.get('diagnostics', [{}])[0] if result.get('diagnostics') else {}
            counts = {k:diagnostics.get(k, {}).get('matched_share', 0) for k in ('operations', 'changed', 'novel', 'successful')}
            require(score['job'] == job and score['status'] == 'complete', 'Incomplete cell: ' + score['status'])
            row = base.verify_cell(job, score, frozen)
            require(result['effective_config']['sharing_strategy'] == job['method'], 'Sharing configuration drift.')
            require(0 <= counts['successful'] <= counts['novel'] <= counts['changed'] <= counts['operations'], 'Invalid operation accounting.')
            if job['method'] == 'random':
                require(all(v == 0 for v in counts.values()), 'Control used matched sharing.')
        except Exception as error:
            row = dict(job=job, verified=False, error=repr(error))
        row['matched_share'] = counts
        rows.append(row)
        print(study.name(job), 'verified' if row['verified'] else row['error'], flush=True)
    summary = summarize(scores['rows'], rows)
    write(HERE/'VERIFICATION.json', dict(status='passed' if summary['technical_pass'] else 'failed', rows=rows,
        expected_systems=len(frozen['jobs']), verified_systems=sum(r['verified'] for r in rows),
        verified_unix=time.time(), elapsed_seconds=time.time() - frozen['budget']['start_unix'],
        freeze_sha256=sha(HERE/'FREEZE.json'), models_sha256=sha(HERE/'ALL_MODELS_FROZEN.json'), scores_sha256=sha(HERE/'SCORES.json')))
    write(HERE/'SUMMARY.json', summary)


if __name__ == '__main__':
    evaluate(); verify()
