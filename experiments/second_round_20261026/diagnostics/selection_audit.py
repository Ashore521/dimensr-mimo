"""Audit frozen candidate selection using stored train/validation metrics only.

Run with Python from any directory. No fitting, prediction, test data, or score
files are used. The only output is SELECTION_AUDIT.json beside this script.
"""
import hashlib
import json
import math
from pathlib import Path
from statistics import mean


EXPECTED_INPUTS = {
    'boundary_s0_full': '2b21021aa8e7275933d1c23dd417fea78ba19a1d7cf64e79c49db22ceda1e4f4',
    'boundary_s1_full': '752a0b148e9f26909074466dda8471483333c8191dc7544235d28f1e2a48535c',
    'partial_s0_full': '79ef332d2defc2f2464894cd99093221acffb0f778b5de74320c2a9140e979e9',
    'partial_s1_full': '7bd5bfee1e4b5b8e916ffd739778c3727da8efbee8d64aad5d6ed14ac14843d1',
    'private_s0_full': 'b44734d99109a2227eff338212d8054bc7bf106b4e4cb78c38ebe79721d15c5d',
    'private_s1_full': 'da23f8898d549291f6f8e1f8b7d3ad97309dfe9247546e87198b793930fb7c9e',
    'scales_s0_full': '6c16ee15f223e8af38b27fefb50c39378e76ec6295d5aaf9d87bc3c75439142c',
    'scales_s1_full': 'c0f8d134d8eba9f2133ebcef33fcd65d32569e0bcccf83b3b5071706562015dd',
    'shared_parameter_s0_full': '267bb6483702ccf5853e87db3f01985fa1f30eb00201032119f6832f3b8f1a7f',
    'shared_parameter_s0_no_gp': '87b78de87ecb5b9cc63b551f67a460ebba5e591fb2012456c94e2676b2b3928c',
    'shared_parameter_s0_no_neural': '7546f8e15327a9abda0176e8a915f70d56eddb9ff9bc6c465561e972a00fe16d',
    'shared_parameter_s0_no_sharing': 'c5be30323c1842c38907ce0f30fc3853e8ec9e6165459ab28ddf40d393914d8a',
    'shared_parameter_s1_full': '54f9e6b5aba7127f470861dbd79db1442d15ffda3a1aa61ce6804c16cd9ed7ef',
    'shared_parameter_s1_no_gp': '7e0f6b6e3f02b4e396aee58ef3ca9ce025cd1ce3c2ab19cdf11c831f570263e0',
    'shared_parameter_s1_no_neural': 'b3b7aa5e7811bee7660baa0447d9db01c2c8845f22976b9fdc6ef702d9e0378a',
    'shared_parameter_s1_no_sharing': '1334eed1e805f137da0a8a6f810df1fda91e1337ea84b81cf7cb8aac380a2c45',
    'shared_s0_full': '22c002956ea811d0cbffc7ad48f5376a8b3ad4c9900d30a188c638a67ba9b425',
    'shared_s1_full': '5c94739a26d95e33afff4912120cc23039ad6adb3149ea39188421b155ed3082',
    'units_s0_full': 'e8583b42680720d02cba70a87e01bc9b1f2fa55029cb173d309cdd0c2c6a3b00',
    'units_s1_full': '61f228a5dd43589de65f8fa885164103f29e35e7967591c38d1c5ec82d9b0f80',
}
CANDIDATE_FIELDS = ('system', 'constants', 'errors', 'complexity', 'rademacher',
                    'rademacher_complexity')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label):
    require(math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-13), label)


def selection_indices(candidates):
    current = min(range(len(candidates)), key=lambda i: (
        candidates[i]['violation'], candidates[i]['score'], candidates[i]['complexity']))
    validation_min = min(range(len(candidates)), key=lambda i: (
        candidates[i]['violation'], mean(candidates[i]['validation_errors']),
        candidates[i]['complexity'], candidates[i]['score']))
    return current, validation_min


def describe(candidate, index):
    system = candidate['system']
    nodes = system['nodes']
    def reachable(root):
        seen = {root}
        for child in nodes[root]['children']:
            seen.update(reachable(child))
        return seen
    reach = [reachable(root) for root in system['roots']]
    params = [{nodes[i]['parameter'] for i in part if nodes[i]['parameter'] is not None}
              for part in reach]
    unique_params = set().union(*params)
    def expression(i):
        node = nodes[i]
        if node['parameter'] is not None:
            return 'a' + str(node['parameter'])
        if not node['children']:
            return node['token']
        return node['token'] + '(' + ', '.join(expression(j) for j in node['children']) + ')'
    return dict(candidate_index=index,
                train_nrmse_per_output=candidate['errors'],
                train_mean_nrmse=mean(candidate['errors']),
                validation_nrmse_per_output=candidate['validation_errors'],
                validation_mean_nrmse=mean(candidate['validation_errors']),
                complexity=candidate['complexity'], weighted_score=candidate['score'],
                violation=candidate['violation'],
                complexity_penalty=candidate['score'] - mean(candidate['validation_errors']),
                sharing=dict(shared_nodes=sum(sum(i in part for part in reach) > 1
                                              for i in range(len(nodes))),
                             shared_parameters=sum(sum(p in part for part in params) > 1
                                                   for p in unique_params),
                             unique_parameters=len(unique_params),
                             dag_size=sum(n['parameter'] is None for n in nodes) + len(unique_params)),
                expressions=[expression(r) for r in system['roots']],
                constants=candidate['constants'], system=system)


def audit(name, path):
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    require(digest == EXPECTED_INPUTS[name], name + ': frozen input SHA256 mismatch')
    model = json.loads(raw)
    cfg, candidates = model['config'], model['candidates']
    q = model['n_outputs']
    require(candidates and q == 2, name + ': unexpected candidate/output count')
    thresholds = [v * (1 + cfg['degradation_relative']) + cfg['degradation_absolute']
                  for v in model['reference_validation']]
    for i, c in enumerate(candidates):
        require(len(c['errors']) == len(c['validation_errors']) == q,
                name + ': candidate dimensions')
        require(all(math.isfinite(v) and v >= 0 for v in c['errors'] + c['validation_errors']),
                name + ': invalid stored errors')
        violation = (sum(max(v - t, 0) for v, t in zip(c['validation_errors'], thresholds))
                     if cfg['use_constraints'] else 0.0)
        score = mean(c['validation_errors']) + cfg['selection_complexity_weight'] * c['complexity'] / (q * cfg['max_nodes'])
        close(c['violation'], violation, f'{name}: violation {i}')
        close(c['score'], score, f'{name}: score {i}')
    selected, alternative = selection_indices(candidates)
    require(all(candidates[selected][k] == model['best'][k] for k in CANDIDATE_FIELDS),
            name + ': stored best differs from original selection rule')
    require(candidates[selected]['validation_errors'] == model['validation_nrmse'],
            name + ': stored validation metrics differ from selected candidate')
    close(candidates[selected]['score'], model['diagnostics']['selected_score'], name + ': selected score')
    current, val_min = describe(candidates[selected], selected), describe(candidates[alternative], alternative)
    require(current['sharing'] == model['diagnostics']['sharing'], name + ': sharing metadata')
    close(current['violation'], val_min['violation'], name + ': different constraint tier')
    require(val_min['validation_mean_nrmse'] <= current['validation_mean_nrmse'],
            name + ': validation minimum ordering')
    payload = [{k: c[k] for k in CANDIDATE_FIELDS} for c in candidates]
    pool_digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    case, suffix = name.split('_s', 1)
    seed, method = suffix.split('_', 1)
    return dict(run=name, case=case, seed=int(seed), method=method,
                input=dict(path=path.as_posix(), sha256=digest, bytes=len(raw)),
                selection_config={k: cfg[k] for k in ('use_constraints', 'degradation_relative',
                    'degradation_absolute', 'selection_complexity_weight', 'max_nodes')},
                reference_validation_nrmse=model['reference_validation'],
                validation_constraint_thresholds=thresholds,
                saved_candidate_count=len(candidates),
                feasible_candidate_count=sum(c['violation'] == 0 for c in candidates),
                minimum_violation=min(c['violation'] for c in candidates),
                saved_candidates_match_frozen_pool_hash=pool_digest == model['diagnostics']['frozen_candidate_sha256'],
                final_logged_valid_systems=max(h['valid_systems'] for h in model['history']),
                current_selected=current, validation_min_same_constraint_tier=val_min,
                selected_minus_validation_min=dict(
                    validation_mean_nrmse=current['validation_mean_nrmse'] - val_min['validation_mean_nrmse'],
                    validation_nrmse_per_output=[a - b for a, b in zip(current['validation_nrmse_per_output'], val_min['validation_nrmse_per_output'])],
                    train_mean_nrmse=current['train_mean_nrmse'] - val_min['train_mean_nrmse'],
                    complexity=current['complexity'] - val_min['complexity'],
                    weighted_score=current['weighted_score'] - val_min['weighted_score']))


def summarize(rows):
    result = dict(systems=len(rows), runs=[r['run'] for r in rows],
                  feasible_systems=sum(r['minimum_violation'] == 0 for r in rows),
                  changed_candidate_systems=sum(r['current_selected']['candidate_index'] != r['validation_min_same_constraint_tier']['candidate_index'] for r in rows),
                  gap_above_1e_12_systems=sum(r['selected_minus_validation_min']['validation_mean_nrmse'] > 1e-12 for r in rows),
                  mean_validation_gap=mean(r['selected_minus_validation_min']['validation_mean_nrmse'] for r in rows),
                  maximum_validation_gap=max(r['selected_minus_validation_min']['validation_mean_nrmse'] for r in rows))
    for key in ('current_selected', 'validation_min_same_constraint_tier'):
        result[key] = {metric: mean(r[key][metric] for r in rows)
                       for metric in ('train_mean_nrmse', 'validation_mean_nrmse', 'complexity', 'weighted_score')}
        result[key]['validation_nrmse_per_output'] = [mean(r[key]['validation_nrmse_per_output'][j] for r in rows) for j in range(2)]
    return result


def self_check():
    # A smaller weighted score can lose validation accuracy; feasibility stays first.
    candidates = [dict(violation=0.0, score=.0101, complexity=30, validation_errors=[.0001] * 2),
                  dict(violation=0.0, score=.001, complexity=1, validation_errors=[.0005] * 2),
                  dict(violation=.01, score=0.0, complexity=0, validation_errors=[0.0] * 2)]
    require(selection_indices(candidates) == (1, 0), 'selection self-check')
    for c in candidates:
        c['violation'] += 1
    require(selection_indices(candidates) == (1, 0), 'minimum-violation fallback self-check')
    c = dict(candidates[0], errors=[.1, .2], constants=[1.7],
             system=dict(nodes=[dict(token='c0', parameter=0, children=[]),
                                dict(token='c0', parameter=0, children=[])], roots=[0, 1]))
    require(describe(c, 0)['sharing'] == dict(shared_nodes=0, shared_parameters=1,
            unique_parameters=1, dag_size=1), 'binding identity self-check')


def main():
    self_check()
    here = Path(__file__).resolve().parent
    root = here.parents[2]
    rows = [audit(name, root / 'experiments' / 'first_round' / 'runs' / name / 'model.json')
            for name in EXPECTED_INPUTS]
    full = [r for r in rows if r['method'] == 'full']
    require(len(rows) == 20 and len(full) == 14, 'audit coverage')
    summary = {'all_full': summarize(full)}
    for method in ('full', 'no_gp', 'no_neural', 'no_sharing'):
        group = [r for r in rows if r['case'] == 'shared_parameter' and r['method'] == method]
        require(len(group) == 2, 'shared_parameter ablation coverage')
        summary['shared_parameter_' + method] = summarize(group)
    report = dict(schema_version=1, status='passed',
                  purpose='Frozen-candidate training/validation selection audit; no new performance claim.',
                  input_scope='Exactly the 20 SHA256-pinned model.json files listed below; no test or SCORES files.',
                  validation_min_rule=['minimum stored violation', 'minimum mean validation NRMSE',
                                       'minimum complexity', 'minimum original weighted score', 'first saved index'],
                  self_checks='passed: weighted ranking, constraint priority, no-feasible fallback, explicit parameter identity',
                  checked_all_candidates=['dimensions and finite errors', 'original validation constraint',
                                          'original weighted score'],
                  checked_all_runs=['input SHA256', 'original selected structure and constants',
                                    'selected validation metrics', 'selected sharing', 'same violation tier'],
                  numerical_gap_reporting_tolerance=1e-12,
                  limitations=['Stored metrics audited, not recomputed from raw arrays.',
                               'Validation minima are post hoc candidate-pool diagnostics, not new deployed models.',
                               'No test labels, test metrics, fitting, prediction, or algorithm changes.',
                               'Saved candidate pools do not reconstruct all generated/rejected search structures.',
                               'The 1e-12 gap tolerance suppresses numerical noise; it is not a scientific acceptance threshold.'],
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  summaries=summary, runs=rows)
    output = here / 'SELECTION_AUDIT.json'
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    require(json.loads(output.read_text(encoding='utf-8')) == report, 'output round-trip')
    print(json.dumps(dict(status='passed', audited_systems=len(rows), output=str(output),
                          summaries=summary), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
