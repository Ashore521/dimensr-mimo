"""Independent frozen-result verification and full-denominator development report.

`self-test` uses synthetic in-memory rows only. `verify` requires the complete
source/model freezes and SCORES.json before importing models or generating data.
No training or source mutation is performed; study.evaluate is never called.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, median
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CASES = ('shared_parameter', 'scales', 'partial', 'boundary')
SEEDS = (101, 102, 103, 104, 105)
METHODS = ('original', 'proposed', 'independent')
TOLERANCE = 1e-8
JOINT_FIELDS = ('system', 'constants', 'errors', 'complexity', 'rademacher', 'rademacher_complexity')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def jobs():
    return [dict(case=c, seed=s, method=m, seconds=60) for c in CASES for s in SEEDS for m in METHODS]


def run_name(job):
    return f"{job['case']}_s{job['seed']}_{job['method']}"


def classify(difference):
    return 'win' if difference < -TOLERANCE else 'loss' if difference > TOLERANCE else 'tie'


def compare(cells, method, baseline):
    index = {(r['job']['case'], r['job']['seed'], r['job']['method']): r for r in cells}
    pairs = []
    for case in CASES:
        for seed in SEEDS:
            a, b = index[(case, seed, method)], index[(case, seed, baseline)]
            available = a.get('verified', False) and b.get('verified', False)
            row = dict(case=case, seed=seed, method=method, baseline=baseline,
                       available=available, method_status=a['status'], baseline_status=b['status'])
            if available:
                x, y = a['metrics'], b['metrics']
                differences = [u - v for u, v in zip(x['nrmse'], y['nrmse'])]
                thresholds = [v * 1.10 + TOLERANCE for v in y['nrmse']]
                row.update(method_mean_nrmse=x['mean_nrmse'], baseline_mean_nrmse=y['mean_nrmse'],
                           difference=x['mean_nrmse'] - y['mean_nrmse'],
                           outcome=classify(x['mean_nrmse'] - y['mean_nrmse']),
                           method_nrmse=x['nrmse'], baseline_nrmse=y['nrmse'],
                           per_output_differences=differences,
                           per_output_outcomes=[classify(d) for d in differences],
                           per_output_degradation_thresholds=thresholds,
                           per_output_within_threshold=[u <= v for u, v in zip(x['nrmse'], thresholds)],
                           method_complexity=x['complexity'], baseline_complexity=y['complexity'],
                           complexity_ratio=x['complexity'] / y['complexity'] if y['complexity'] else None)
            pairs.append(row)
    return dict(method=method, baseline=baseline, difference_direction='method minus baseline',
                tolerance=TOLERANCE, overall=pair_summary(pairs),
                strata={case: pair_summary([p for p in pairs if p['case'] == case]) for case in CASES}, pairs=pairs)


def pair_summary(pairs):
    available = [p for p in pairs if p['available']]
    complete = len(available) == len(pairs)
    counts = Counter(p['outcome'] for p in available)
    output_counts = Counter(o for p in available for o in p['per_output_outcomes'])
    result = dict(expected_pairs=len(pairs), available_pairs=len(available), complete=complete,
                  wins=counts['win'], ties=counts['tie'], losses=counts['loss'],
                  unavailable=len(pairs) - len(available),
                  expected_outputs=2 * len(pairs), available_outputs=2 * len(available),
                  output_wins=output_counts['win'], output_ties=output_counts['tie'], output_losses=output_counts['loss'],
                  outputs_within_degradation_threshold=sum(sum(p['per_output_within_threshold']) for p in available),
                  mean_difference=None, median_difference=None, method_mean_nrmse=None,
                  baseline_mean_nrmse=None, relative_mean_reduction=None,
                  mean_per_output_differences=None, median_per_output_differences=None,
                  method_mean_complexity=None, baseline_mean_complexity=None,
                  ratio_of_mean_complexity=None, median_complexity_ratio=None)
    if complete:
        a, b = mean(p['method_mean_nrmse'] for p in pairs), mean(p['baseline_mean_nrmse'] for p in pairs)
        ac, bc = mean(p['method_complexity'] for p in pairs), mean(p['baseline_complexity'] for p in pairs)
        ratios = [p['complexity_ratio'] for p in pairs]
        result.update(mean_difference=mean(p['difference'] for p in pairs),
                      median_difference=median(p['difference'] for p in pairs),
                      method_mean_nrmse=a, baseline_mean_nrmse=b,
                      relative_mean_reduction=(b - a) / b if b else None,
                      mean_per_output_differences=[mean(p['per_output_differences'][j] for p in pairs) for j in range(2)],
                      median_per_output_differences=[median(p['per_output_differences'][j] for p in pairs) for j in range(2)],
                      method_mean_complexity=ac, baseline_mean_complexity=bc,
                      ratio_of_mean_complexity=ac / bc if bc else None,
                      median_complexity_ratio=median(ratios) if all(r is not None for r in ratios) else None)
    return result


def method_summary(cells, method):
    rows = [c for c in cells if c['job']['method'] == method]
    complete = len(rows) == 20 and all(c.get('verified', False) for c in rows)
    result = dict(expected=20, statuses=dict(Counter(c['status'] for c in rows)),
                  verified=sum(c.get('verified', False) for c in rows), complete=complete,
                  mean_nrmse=None, median_nrmse=None, worst_output_nrmse=None,
                  mean_complexity=None, mean_nodes=None, mean_parameters=None,
                  wall_seconds=sum(c.get('resources', {}).get('wall_seconds', 0) for c in rows),
                  sampled_cpu_seconds=sum(c.get('resources', {}).get('process_tree_cpu_seconds', 0) for c in rows),
                  peak_rss_bytes=max((c.get('resources', {}).get('process_tree_peak_rss_bytes', 0) for c in rows), default=0))
    if complete:
        result.update(mean_nrmse=mean(c['metrics']['mean_nrmse'] for c in rows),
                      median_nrmse=median(c['metrics']['mean_nrmse'] for c in rows),
                      worst_output_nrmse=max(c['metrics']['max_nrmse'] for c in rows),
                      mean_complexity=mean(c['metrics']['complexity'] for c in rows),
                      mean_nodes=mean(c['metrics']['nodes'] for c in rows),
                      mean_parameters=mean(c['metrics']['parameters'] for c in rows))
    return result


def recommendation(cells, comparison, isolation, preflight_ok, budget_ok):
    overall = comparison['overall']
    complete = len(cells) == 60 and all(c['status'] == 'complete' for c in cells)
    gates = dict(all_60_complete=complete,
                 all_60_independently_verified=len(cells) == 60 and all(c.get('verified', False) for c in cells),
                 all_60_full_mechanisms=len(cells) == 60 and all(c.get('mechanism_pass', False) for c in cells),
                 all_60_domain_enclosed=len(cells) == 60 and all(c.get('domain_pass', False) for c in cells),
                 frozen_hashes_and_order=preflight_ok,
                 all_20_search_and_pool_isolation=len(isolation) == 20 and all(p['passed'] for p in isolation),
                 protocol_wall_budget=budget_ok,
                 mean_relative_reduction_at_least_10_percent=overall['complete'] and overall['relative_mean_reduction'] is not None and overall['relative_mean_reduction'] >= .10,
                 mean_absolute_reduction_above_1e_8=overall['complete'] and overall['mean_difference'] < -TOLERANCE,
                 all_40_outputs_within_degradation_bound=overall['complete'] and overall['outputs_within_degradation_threshold'] == 40,
                 mean_complexity_ratio_at_most_1_5=overall['complete'] and overall['ratio_of_mean_complexity'] is not None and overall['ratio_of_mean_complexity'] <= 1.50)
    return dict(recommend=all(gates.values()), gates=gates,
                failed_gates=[k for k, v in gates.items() if not v],
                scope='Configuration recommendation for these four known development tasks only.')


def verify_hashes(files, root):
    errors = []
    for name, expected in files.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            errors.append('Manifest path outside root: ' + name)
        elif not path.is_file() or sha(path) != expected:
            errors.append('Frozen file missing/drifted: ' + name)
    return errors


def preflight():
    frozen, models, scores = [read(HERE / name) for name in ('FREEZE.json', 'ALL_MODELS_FROZEN.json', 'SCORES.json')]
    budget = read(HERE / 'BUDGET.json')
    started = read(HERE / 'TRAINING_STARTED.json')['start_unix']
    change = read(HERE / 'NEW_CHANGE.json')
    errors = verify_hashes(frozen['files'], ROOT) + verify_hashes(models['files'], HERE)
    def check(condition, message):
        if not condition:
            errors.append(message)
    check(frozen['jobs'] == jobs(), 'Frozen matrix differs from the exact 60-system protocol.')
    check([r['job'] for r in scores['rows']] == jobs(), 'Scores lack the full matrix or are out of order.')
    check(models['freeze_sha256'] == sha(HERE / 'FREEZE.json'), 'Model freeze chain mismatch.')
    check(scores['all_models_freeze_sha256'] == sha(HERE / 'ALL_MODELS_FROZEN.json'), 'Score freeze chain mismatch.')
    check(not frozen['test_labels_accessed'] and not models['test_labels_accessed'], 'Pre-score test access flag.')
    check(budget['start_unix'] <= frozen['frozen_unix'] <= started <= models['frozen_unix'] <= scores['scored_unix'], 'Freeze/train/test timestamp order.')
    check(frozen['budget'] == budget and budget['limit_seconds'] == 5400 and budget['training_limit_seconds'] == 3600,
          'Budget metadata differs from protocol.')
    check(budget['threads'] == budget['sequential_workers'] == 1, 'Concurrency differs from protocol.')
    check(change['config_overrides'] == {'selection_complexity_weight': 0.0} and change['test_data_used_for_decision'] is False,
          'Unique change contract mismatch.')
    cfg = frozen['configs']
    check(cfg['independent'] == cfg['original'], 'Independent base configuration changed.')
    check(cfg['original'] == read(HERE.parent / 'first_round' / 'FREEZE.json')['config'], 'Original configuration differs from frozen first round.')
    check(cfg['proposed'] == dict(cfg['original'], selection_complexity_weight=0.0), 'Proposed changed more than final selection weight.')
    required = list((ROOT / 'src').rglob('*.py')) + list(HERE.rglob('*.py'))
    required += [HERE / 'PROTOCOL.md', HERE / 'NEW_CHANGE.json', HERE / 'diagnostics' / 'SELECTION_AUDIT.json', HERE / 'diagnostics' / 'FIT_TRACE.json']
    check(all(p.relative_to(ROOT).as_posix() in frozen['files'] for p in required), 'Source/protocol/diagnostic freeze coverage is incomplete.')
    actual_files = {p.relative_to(HERE).as_posix() for p in (HERE / 'runs').rglob('*') if p.is_file()}
    check(actual_files == set(models['files']), 'Model/status freeze does not exactly cover the run directory.')
    for job in jobs():
        folder = HERE / 'runs' / run_name(job)
        try:
            checkpoint = read(folder / 'CELL_FROZEN.json')
            errors.extend(verify_hashes(checkpoint['files'], folder))
            actual = {p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file() and p.name != 'CELL_FROZEN.json'}
            check(actual == set(checkpoint['files']), run_name(job) + ': incomplete cell freeze.')
            check(frozen['frozen_unix'] <= checkpoint['frozen_unix'] <= models['frozen_unix'], run_name(job) + ': cell timestamp order.')
            check(read(folder / 'job.json') == job, run_name(job) + ': job mismatch.')
            check(read(folder / 'config.json') == cfg[job['method']], run_name(job) + ': config mismatch.')
            check(read(folder / 'resources.json')['job'] == job, run_name(job) + ': resource job mismatch.')
        except (OSError, ValueError, KeyError) as error:
            errors.append(run_name(job) + ': ' + repr(error))
    return frozen, models, scores, budget, errors


def numerical_close(actual, expected, label):
    import numpy as np
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), label + ': shape/nonfinite')
    require(np.allclose(a, b, rtol=1e-10, atol=1e-12), label + ': numeric mismatch')
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def metrics(prediction, target, scales):
    import numpy as np
    prediction, target = np.asarray(prediction), np.asarray(target)
    require(prediction.shape == target.shape and np.isfinite(prediction).all(), 'Invalid metric predictions.')
    residuals = prediction - target
    rmse = np.sqrt(np.mean(residuals ** 2, axis=0))
    nrmse = rmse / scales
    denominator = np.sum((target - target.mean(axis=0)) ** 2, axis=0)
    require(np.all(denominator > 0), 'R2 undefined for a constant test target.')
    r2 = 1 - np.sum(residuals ** 2, axis=0) / denominator
    return dict(rmse=rmse.tolist(), nrmse=nrmse.tolist(), r2=r2.tolist(),
                mean_nrmse=float(nrmse.mean()), max_nrmse=float(nrmse.max()))


def mechanism(events, result, independent):
    evidence = []
    for output in (0, 1) if independent else (None,):
        rows = [e for e in events if e['output'] == output]
        nsga = [e for e in rows if e['phase'] == 'nsga2']
        gp_calls = (result['diagnostics'][output]['instrumentation'].get('evaluation_gp_new', 0)
                    if independent else sum(e.get('new_evaluations', 0) for e in rows if e['phase'] == 'gp'))
        item = dict(output=output, policy_delta=sum(e.get('parameter_delta', 0) for e in rows),
                    neural_valid=sum(e.get('neural_valid', 0) for e in rows),
                    gp_new_evaluations=gp_calls, nsga_generations=len(nsga),
                    nsga_new_evaluations=sum(e.get('new_evaluations', 0) for e in nsga))
        item['passed'] = (item['policy_delta'] > 0 and item['neural_valid'] > 0 and gp_calls > 0
                          and len(nsga) >= 2 and item['nsga_new_evaluations'] > 0)
        evidence.append(item)
    return evidence


def selection_check(data, V, Z, scales, independent, output, classes):
    import numpy as np
    cfg = data['config']
    candidates = data['pareto_front'] if independent else data['candidates']
    require(bool(candidates), 'No saved selection candidates.')
    choices, maximum_delta = [], 0.0
    for candidate in candidates:
        if independent:
            tree = classes['Node'].from_dict(candidate['tree'])
            pred = tree.evaluate(V, candidate['constants'])
            rmse = float(np.sqrt(np.mean((pred - Z[:, output]) ** 2)))
            score = rmse / scales[output] + cfg['selection_complexity_weight'] * candidate['complexity'] / cfg['max_nodes']
            maximum_delta = max(maximum_delta, numerical_close(rmse, candidate['validation_rmse'], 'candidate validation RMSE'),
                                numerical_close(score, candidate['selection_score'], 'candidate selection score'))
            choices.append(score)
        else:
            system = classes['System'].from_dict(candidate['system'])
            pred = system.evaluate(V, candidate['constants'])
            errors = np.sqrt(np.mean(((pred - Z) / scales) ** 2, axis=0))
            threshold = np.asarray(data['reference_validation']) * (1 + cfg['degradation_relative']) + cfg['degradation_absolute']
            violation = float(np.maximum(errors - threshold, 0).sum()) if cfg['use_constraints'] else 0.0
            score = float(errors.mean() + cfg['selection_complexity_weight'] * candidate['complexity'] / (Z.shape[1] * cfg['max_nodes']))
            maximum_delta = max(maximum_delta, numerical_close(errors, candidate['validation_errors'], 'candidate validation NRMSE'),
                                numerical_close(violation, candidate['violation'], 'candidate violation'),
                                numerical_close(score, candidate['score'], 'candidate selection score'))
            choices.append((violation, score, candidate['complexity']))
    index = min(range(len(choices)), key=choices.__getitem__)
    selected = candidates[index]
    require(all(selected.get(k) == v for k, v in data['best'].items()), 'Reconstructed selection differs from saved best.')
    matches = [i for i, c in enumerate(candidates) if all(c.get(k) == v for k, v in data['best'].items())]
    require(matches == [index], 'Saved best is not uniquely identified.')
    if independent:
        numerical_close(selected['validation_rmse'], data['validation_rmse'], 'selected validation RMSE')
    else:
        numerical_close(selected['validation_errors'], data['validation_nrmse'], 'selected validation NRMSE')
        payload = [{k: c[k] for k in JOINT_FIELDS} for c in candidates]
        require(digest(payload) == data['diagnostics']['frozen_candidate_sha256'], 'Saved candidates do not reconstruct the frozen pool.')
    return dict(candidate_index=index, candidate_sha256=digest(selected), best_sha256=digest(data['best']),
                pool_sha256=digest(candidates), candidate_count=len(candidates), maximum_metric_delta=maximum_delta)


def verify_run(job, row, frozen, arrays, classes):
    import numpy as np
    folder = HERE / 'runs' / run_name(job)
    resource = read(folder / 'resources.json')
    result = read(folder / 'result.json')
    require(row['resources'] == resource and resource['status'] == result['status'] == 'complete', 'Run status/resource mismatch.')
    require(result['job'] == job and resource['returncode'] == 0, 'Worker failure or job mismatch.')
    require(resource['wall_seconds'] <= job['seconds'], 'System exceeded the 60-second cap.')
    require(result['reload_predictions_equal'] and result['reload_audits_equal'], 'Worker reload checks failed.')
    events = [json.loads(line) for line in (folder / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
    require(events and all(a['monotonic'] <= b['monotonic'] for a, b in zip(events, events[1:])), 'Events are missing or out of order.')
    independent = job['method'] == 'independent'
    filenames = ['model-0.json', 'model-1.json'] if independent else ['model.json']
    require(result['models'] == filenames, 'Unexpected saved model files/order.')
    evidence = mechanism(events, result, independent)
    for verified, recorded in zip(evidence, row['mechanism']['outputs']):
        require(all(verified[k] == v for k, v in recorded.items()), 'Scorer mechanism evidence differs.')
    mechanism_pass = all(e['passed'] for e in evidence)
    require(len(row['mechanism']['outputs']) == len(evidence) and row['mechanism']['full_mechanism_pass'] == mechanism_pass,
            'Scorer mechanism status differs.')
    durations = []
    if independent:
        starts = [e for e in events if e['phase'] == 'output_start']
        ends = [e for e in events if e['phase'] == 'output_complete']
        require([e['output'] for e in starts] == [0, 1] == [e['output'] for e in ends], 'Independent output lifecycle incomplete.')
        require(ends[0]['monotonic'] <= starts[1]['monotonic'], 'Independent outputs overlap.')
        durations = [b['monotonic'] - a['monotonic'] for a, b in zip(starts, ends)]
        require(all(0 <= t <= 30 for t in durations), 'Independent output exceeded its 30-second cap.')
    (X, Y), (V, Z), (T, U) = arrays
    scales = np.std(Y, axis=0)
    # These fixed tasks have nonconstant targets; retain the training scale explicitly.
    require(np.all(scales > 1e-12), 'Unexpected target scale outside fixed-task assumptions.')
    trained, raw_data, checks, audits, test_predictions = [], [], [], [], []
    maximum_delta = 0.0
    model_class = classes['single'] if independent else classes['joint']
    for output, filename in enumerate(filenames):
        data = read(folder / filename)
        raw_data.append(data)
        model = model_class.load(folder / filename)
        trained.append(model)
        base = frozen['configs'][job['method']]
        expected_config = {k: base[k] for k in data['config']}
        expected_config['random_state'] = job['seed']
        if independent:
            expected_config['max_evaluations'] = base['max_evaluations'] // 2
        else:
            expected_config['wall_seconds'] = 55
        require(data['config'] == expected_config and data['config']['use_rnn'] and data['config']['use_gp'], 'Effective model configuration differs.')
        check = selection_check(data, V, Z, scales, independent, output, classes)
        record = result['selections'][output]
        require(record['model'] == filename and all(check[k] == v for k, v in record.items() if k != 'model'), 'Selection identity record mismatch.')
        checks.append(check)
        maximum_delta = max(maximum_delta, check['maximum_metric_delta'])
        selected = data['best']
        expression = classes['Node'].from_dict(selected['tree']) if independent else classes['System'].from_dict(selected['system'])
        for split, A in [('train', X), ('validation', V), ('test', T)]:
            direct = expression.evaluate(A, selected['constants'])
            reloaded = model.predict(A)
            require(np.array_equal(direct, reloaded), 'Raw expression and loaded prediction differ: ' + split)
            if split != 'test':
                prediction_hash = hashlib.sha256(reloaded.astype('<f8').tobytes()).hexdigest()
                require(prediction_hash == result['prediction_hashes'][output][split], 'Original/reloaded prediction hash mismatch: ' + split)
            else:
                test_predictions.append(reloaded)
        if independent:
            train_rmse = float(np.sqrt(np.mean((model.predict(X) - Y[:, output]) ** 2)))
            maximum_delta = max(maximum_delta, numerical_close(train_rmse, selected['train_rmse'], 'best train RMSE'),
                                numerical_close(train_rmse / scales[output], selected['train_nrmse'], 'best train NRMSE'))
        else:
            train_errors = np.sqrt(np.mean(((model.predict(X) - Y) / scales) ** 2, axis=0))
            maximum_delta = max(maximum_delta, numerical_close(train_errors, selected['errors'], 'best train NRMSE'),
                                numerical_close(scales, data['scales'], 'training normalization'))
        audit = model.audit_domain(input_bounds=classes['design'](job['case'])['bounds'])
        require(audit == result['audits'][output] == row['audit'][output], 'Reloaded domain audit differs.')
        audits.append(audit)
    prediction = np.column_stack(test_predictions) if independent else test_predictions[0]
    measured = metrics(prediction, U, scales)
    for key in ('rmse', 'nrmse', 'r2', 'mean_nrmse', 'max_nrmse'):
        maximum_delta = max(maximum_delta, numerical_close(measured[key], row[key], 'test ' + key))
    if independent:
        measured.update(complexity=sum(m.best_.complexity for m in trained),
                        nodes=sum(m.best_.tree.size for m in trained), parameters=sum(len(m.best_.constants) for m in trained),
                        formulas=[str(m.expression_) for m in trained], sharing=None)
    else:
        m = trained[0]
        measured.update(complexity=m.best_.complexity, nodes=m.best_.system.size,
                        parameters=m.best_.system.n_parameters, formulas=m.formulas_, sharing=m.best_.system.sharing)
        require(measured['sharing'] == row['sharing'] == raw_data[0]['diagnostics']['sharing'], 'Sharing metadata differs.')
    for key in ('complexity', 'nodes', 'parameters'):
        numerical_close(measured[key], row[key], key)
    require(measured['formulas'] == row['formulas'], 'Displayed formula differs from saved model.')
    require(row['reload_equal'] and row['saved_original_predictions_equal'] and row['saved_original_audits_equal'], 'Scorer reload flags failed.')
    record = dict(job=job, status='complete', verified=True, resources=resource, metrics=measured,
                  train_scales=scales.tolist(), mechanism_pass=mechanism_pass, mechanism=evidence,
                  domain_pass=all(a['status'] == 'enclosed' for a in audits), domain_audits=audits,
                  selection_checks=checks, maximum_metric_delta=maximum_delta,
                  independent_output_wall_seconds=durations,
                  model_sha256={name: sha(folder / name) for name in filenames})
    if not independent:
        record['search_fingerprint'] = raw_data[0]['search_fingerprint']
        record['frozen_pool_sha256'] = raw_data[0]['diagnostics']['frozen_candidate_sha256']
        record['validation_feasible'] = raw_data[0]['diagnostics']['validation_feasible']
    return record


def isolation_checks(cells):
    index = {(r['job']['case'], r['job']['seed'], r['job']['method']): r for r in cells}
    pairs = []
    for case in CASES:
        for seed in SEEDS:
            a, b = index[(case, seed, 'original')], index[(case, seed, 'proposed')]
            available = a.get('verified', False) and b.get('verified', False)
            search_equal = available and a['search_fingerprint'] == b['search_fingerprint']
            pool_equal = available and a['frozen_pool_sha256'] == b['frozen_pool_sha256']
            pairs.append(dict(case=case, seed=seed, available=available, search_fingerprint_equal=search_equal,
                              frozen_pool_hash_equal=pool_equal, passed=search_equal and pool_equal,
                              original_search_fingerprint=a.get('search_fingerprint'), proposed_search_fingerprint=b.get('search_fingerprint'),
                              original_frozen_pool_sha256=a.get('frozen_pool_sha256'), proposed_frozen_pool_sha256=b.get('frozen_pool_sha256')))
    return pairs


def diagnostic_summary():
    selection = read(HERE / 'diagnostics' / 'SELECTION_AUDIT.json')
    trace = read(HERE / 'diagnostics' / 'FIT_TRACE.json')
    return dict(selection_audit=selection['summaries'], diagnostic_wall_seconds=trace['wall_seconds'],
                fixed_structure_and_search=[dict(seed=r['seed'], status=r['status'],
                    original_structure_recovered=next(f['recovered'] for f in r['summary']['fixed_structure'] if f['mode'] == 'original'),
                    original_structure_nfev=next(f['stats']['constant_nfev'] for f in r['summary']['fixed_structure'] if f['mode'] == 'original'),
                    original_search_fingerprint_equal=r['summary']['fingerprints_equal'],
                    direct_target_proposals=r['summary']['direct_target_proposals'],
                    best_validation_outcome=r['summary']['best_cache_validation']['outcome']) for r in trace['rows']],
                boundary='Known-structure fitting and direct-pattern proposal checks; no algebraic-equivalence proof or general discovery guarantee.')


def format_number(value):
    return '未完成' if value is None else f'{value:.8g}'


def report_text(summary, verification):
    recommendation_result = summary['recommendation']
    verdict = '通过本轮配置推荐门槛' if recommendation_result['recommend'] else '未通过本轮配置推荐门槛，维持原默认配置'
    lines = ['# 第二轮：选择规则的新划分开发复验', '', '## Material Passport', '',
             '- 类型：四个已知合成开发任务的新随机划分；4×5×3=60 个系统。',
             '- 唯一改动：最终验证选择的复杂度权重从 0.01 变为 0；训练搜索配置不变。',
             '- 输入与顺序证据：FREEZE.json、逐单元 CELL_FROZEN.json、ALL_MODELS_FROZEN.json、SCORES.json。',
             '- 本报告由冻结的 verify_report.py 独立重算；逐项证据见 VERIFICATION.json，完整配对见 SUMMARY.json。', '',
             f'结论：**{verdict}**。核验通过 {verification["verified_systems"]}/60，完整机制 {verification["mechanism_pass_systems"]}/60，域筛查 {verification["domain_pass_systems"]}/60，搜索/候选池隔离 {verification["isolation_pass_pairs"]}/20。', '',
             '失败、超时及未完成项保留在分母中；缺少任何系统时，不计算该比较的完整均值或中位数。', '',
             '## 三方法完整结果', '',
             '| 方法 | 核验/20 | 平均 NRMSE | 中位 NRMSE | 最坏输出 NRMSE | 平均复杂度 | 平均节点 | 平均参数 | 墙钟秒 | CPU秒 | 峰值RSS MiB |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for method, row in summary['methods'].items():
        values = [row[k] for k in ('mean_nrmse', 'median_nrmse', 'worst_output_nrmse', 'mean_complexity', 'mean_nodes', 'mean_parameters', 'wall_seconds', 'sampled_cpu_seconds')]
        lines.append(f'| {method} | {row["verified"]}/20 | ' + ' | '.join(format_number(v) for v in values) + f' | {row["peak_rss_bytes"] / 1048576:.3f} |')
    lines += ['', '复杂度跨表示比较仅作描述；联合 DAG 与逐输出树同时报告节点和参数数。CPU/RSS 为监督器每 0.2 秒的进程树采样，CPU 末次采样可能低估。', '']
    for name, comparison in summary['comparisons'].items():
        lines += [f'## {name}', '', '差值方向为前者减后者；负数表示误差降低。改善/持平/退化使用 1e-8 绝对容差。', '',
                  '| 分层 | 完整配对 | 均值差 | 中位数差 | 改善/持平/退化/缺失 | 平均复杂度比 |', '|---|---:|---:|---:|---|---:|']
        for case, row in [('全部', comparison['overall']), *comparison['strata'].items()]:
            lines.append(f'| {case} | {row["available_pairs"]}/{row["expected_pairs"]} | {format_number(row["mean_difference"])} | {format_number(row["median_difference"])} | {row["wins"]}/{row["ties"]}/{row["losses"]}/{row["unavailable"]} | {format_number(row["ratio_of_mean_complexity"])} |')
        lines += ['', '| 任务 | 种子 | 前者均值 | 后者均值 | 差值 | 结果 |', '|---|---:|---:|---:|---:|---|']
        for p in comparison['pairs']:
            lines.append(f'| {p["case"]} | {p["seed"]} | {format_number(p.get("method_mean_nrmse"))} | {format_number(p.get("baseline_mean_nrmse"))} | {format_number(p.get("difference"))} | {p.get("outcome", "unavailable")} |')
        lines.append('')
    lines += ['## 推荐门槛', '', '| 门槛 | 是否通过 |', '|---|---|']
    for key, passed in recommendation_result['gates'].items():
        lines.append(f'| {key} | {"是" if passed else "否"} |')
    main = summary['comparisons']['proposed_minus_original']['overall']
    lines += ['', f'逐输出保护：{main["outputs_within_degradation_threshold"]}/40 项满足 proposed ≤ original×1.10+1e-8；缺失项不计为通过。全部 120 个方法输出及 40 个主要逐输出差见 metrics_per_output.csv。', '',
              '## 诊断与解释边界', '',
              '首轮冻结候选审计发现尺度差异的一个种子与共享参数的两个种子存在选择损失。已知正确共享非线性结构在原拟合配置下可恢复 a=1.7；两次原搜索重放没有提出直接目标结构，较好的验证候选已进入最终池。', '',
              '本轮改动仅处理最终选择的精度与复杂度权衡。共享参数直接结构发现缺口仍未解决；固定结构拟合成功不等于自动发现成功。日志中的机制执行和本轮配对也不构成神经、GP 或共享机制因果收益证明。', '',
              '这是已知任务的新划分开发复验，没有新增未见任务盲测或真实数据验证。enclosed 表示有限浮点区间数值筛查，不是连续域形式证明。不做显著性筛选，不依据本轮测试结果追加改动或种子。', '',
              f'正式阶段从 freeze 开始累计 {format_number(verification.get("wall_seconds_from_freeze"))} 秒，上限 5400 秒；独立核验耗时 {format_number(verification.get("verification_seconds"))} 秒。诊断耗时另列 SUMMARY.json；开发/交互时间未计入正式阶段。']
    if verification['errors']:
        lines += ['', '## 未完成或核验失败', ''] + ['- ' + e for e in verification['errors']]
    return '\n'.join(lines) + '\n'


def write_outputs(summary, verification, cells):
    write(HERE / 'SUMMARY.json', summary)
    write(HERE / 'VERIFICATION.json', verification)
    (HERE / 'REPORT.zh-CN.md').write_text(report_text(summary, verification), encoding='utf-8')
    index = {(c['job']['case'], c['job']['seed'], c['job']['method']): c for c in cells}
    fields = ['case', 'seed', 'method', 'output', 'status', 'verified', 'rmse', 'nrmse', 'r2',
              'difference_vs_original', 'difference_vs_independent', 'original_degradation_threshold',
              'within_original_degradation_threshold', 'complexity', 'nodes', 'parameters',
              'domain_enclosed', 'mechanism_pass', 'formula']
    with (HERE / 'metrics_per_output.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for cell in cells:
            job = cell['job']
            for output in range(2):
                row = {k: job[k] for k in ('case', 'seed', 'method')}
                row.update(output=output, status=cell['status'], verified=cell.get('verified', False),
                           domain_enclosed=cell.get('domain_pass', False), mechanism_pass=cell.get('mechanism_pass', False))
                if cell.get('verified', False):
                    m = cell['metrics']
                    row.update({k: m[k][output] for k in ('rmse', 'nrmse', 'r2')})
                    row.update({k: m[k] for k in ('complexity', 'nodes', 'parameters')})
                    row['formula'] = m['formulas'][output]
                    for reference in ('original', 'independent'):
                        other = index[(job['case'], job['seed'], reference)]
                        if other.get('verified', False):
                            value = other['metrics']['nrmse'][output]
                            row['difference_vs_' + reference] = m['nrmse'][output] - value
                            if reference == 'original' and job['method'] == 'proposed':
                                row['original_degradation_threshold'] = value * 1.10 + TOLERANCE
                                row['within_original_degradation_threshold'] = m['nrmse'][output] <= row['original_degradation_threshold']
                writer.writerow(row)


def verify():
    started = time.time()
    frozen = models = scores = budget = None
    errors, cells = [], []
    try:
        frozen, models, scores, budget, errors = preflight()
        if time.time() >= budget['start_unix'] + budget['limit_seconds']:
            errors.append('Verification started after the 5400-second experiment deadline.')
    except (OSError, ValueError, KeyError, TypeError) as error:
        errors.append('Preflight failed: ' + repr(error))
    preflight_ok = not errors
    if preflight_ok:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from experiments.second_round_20261026 import cases
        from dimensr_mimo import DimenSRMIMORegressor
        from dimensr_mimo._single import DimenSRRegressor
        from dimensr_mimo.graph import System
        from dimensr_mimo._single.expression import Node
        classes = dict(joint=DimenSRMIMORegressor, single=DimenSRRegressor, System=System, Node=Node, design=cases.design)
        cache = {}
    for index, job in enumerate(jobs()):
        folder = HERE / 'runs' / run_name(job)
        resource = read(folder / 'resources.json') if (folder / 'resources.json').exists() else {'status': 'missing'}
        cell = dict(job=job, status=resource['status'], resources=resource, verified=False,
                    mechanism_pass=False, domain_pass=False)
        if preflight_ok and resource['status'] == 'complete':
            try:
                require(time.time() < budget['start_unix'] + budget['limit_seconds'], 'Independent verification exceeded global deadline.')
                key = (job['case'], job['seed'])
                if key not in cache:
                    cache[key] = [cases.generate(*key[:1], split, key[1]) for split in ('train', 'validation', 'test')]
                cell = verify_run(job, scores['rows'][index], frozen, cache[key], classes)
            except Exception as error:
                cell.update(status='verification_failed', error=repr(error))
                errors.append(run_name(job) + ': ' + repr(error))
        elif resource['status'] != 'complete':
            errors.append(run_name(job) + ': ' + resource['status'])
        cells.append(cell)
    isolation = isolation_checks(cells)
    comparisons = {f'{a}_minus_{b}': compare(cells, a, b) for a, b in
                   [('proposed', 'original'), ('original', 'independent'), ('proposed', 'independent')]}
    wall = time.time() - budget['start_unix'] if budget is not None else None
    budget_ok = wall is not None and wall <= 5400
    if budget is not None and models is not None:
        budget_ok = budget_ok and models['frozen_unix'] - read(HERE / 'TRAINING_STARTED.json')['start_unix'] <= 3600
    decision = recommendation(cells, comparisons['proposed_minus_original'], isolation, preflight_ok, budget_ok)
    verification = dict(schema_version=1, status='passed' if all(decision['gates'][k] for k in list(decision['gates'])[:7]) else 'failed',
        expected_systems=60, observed_statuses=dict(Counter(c['status'] for c in cells)),
        verified_systems=sum(c.get('verified', False) for c in cells),
        mechanism_pass_systems=sum(c.get('mechanism_pass', False) for c in cells),
        domain_pass_systems=sum(c.get('domain_pass', False) for c in cells),
        isolation_pass_pairs=sum(p['passed'] for p in isolation), preflight_pass=preflight_ok,
        frozen_source_files=len(frozen['files']) if frozen else 0,
        frozen_model_status_files=len(models['files']) if models else 0,
        metric_tolerance=dict(rtol=1e-10, atol=1e-12), outcome_tolerance=TOLERANCE,
        maximum_metric_difference=max((c.get('maximum_metric_delta', 0.0) for c in cells), default=0.0),
        wall_seconds_from_freeze=wall, verification_seconds=time.time() - started, budget_respected=budget_ok,
        evidence_sha256={name: sha(HERE / name) for name in ('FREEZE.json', 'ALL_MODELS_FROZEN.json', 'SCORES.json', 'PROTOCOL.md', 'NEW_CHANGE.json', 'verify_report.py') if (HERE / name).exists()},
        errors=errors, isolation_pairs=isolation, cells=cells)
    summary = dict(schema_version=1, material_passport=dict(type='known-task synthetic development replication',
                    systems=60, main_pairs=20, main_outputs=40, strata=list(CASES), seeds=list(SEEDS),
                    unique_change={'selection_complexity_weight': 0.0}, independent_full_algorithm_reference=True,
                    blind_unseen_task_evidence=False, neural_gp_causal_benefit_evidence=False),
                   methods={m: method_summary(cells, m) for m in METHODS}, comparisons=comparisons,
                   recommendation=decision, diagnostics=diagnostic_summary() if preflight_ok else None)
    write_outputs(summary, verification, cells)
    # Include report writing in the formal wall-clock accounting.
    verification['wall_seconds_from_freeze'] = time.time() - budget['start_unix'] if budget is not None else None
    verification['verification_seconds'] = time.time() - started
    if verification['wall_seconds_from_freeze'] is not None and verification['wall_seconds_from_freeze'] > 5400:
        verification['budget_respected'] = False
        verification['status'] = 'failed'
        summary['recommendation'] = recommendation(cells, comparisons['proposed_minus_original'], isolation, preflight_ok, False)
    write_outputs(summary, verification, cells)
    print(json.dumps(dict(status=verification['status'], verified_systems=verification['verified_systems'],
          mechanism_pass_systems=verification['mechanism_pass_systems'], isolation_pass_pairs=verification['isolation_pass_pairs'],
          wall_seconds_from_freeze=verification['wall_seconds_from_freeze'], recommendation=summary['recommendation'],
          main_overall=comparisons['proposed_minus_original']['overall'], main_strata=comparisons['proposed_minus_original']['strata']), indent=2))
    return 0 if verification['status'] == 'passed' else 1


def self_test():
    require([classify(v) for v in (-2e-8, -1e-8, 0, 1e-8, 2e-8)] == ['win', 'tie', 'tie', 'tie', 'loss'], 'Outcome tolerance test.')
    cells = []
    for job in jobs():
        value = {'original': 1.0, 'proposed': .8, 'independent': 1.1}[job['method']]
        cells.append(dict(job=job, status='complete', verified=True, mechanism_pass=True, domain_pass=True,
                          metrics=dict(nrmse=[value, value], mean_nrmse=value, max_nrmse=value,
                                       complexity=12 if job['method'] == 'proposed' else 10, nodes=5, parameters=2),
                          search_fingerprint='same', frozen_pool_sha256='same'))
    comparison = compare(cells, 'proposed', 'original')
    require(comparison['overall']['wins'] == 20 and comparison['overall']['outputs_within_degradation_threshold'] == 40, 'Pair/output denominator test.')
    require(all(s['expected_pairs'] == 5 for s in comparison['strata'].values()), 'Stratum denominator test.')
    require(recommendation(cells, comparison, isolation_checks(cells), True, True)['recommend'], 'Passing recommendation test.')
    cells[0]['verified'] = False
    incomplete = compare(cells, 'proposed', 'original')
    require(incomplete['overall']['available_pairs'] == 19 and incomplete['overall']['mean_difference'] is None, 'Do not average successful subsets.')
    require(not recommendation(cells, incomplete, isolation_checks(cells), True, True)['recommend'], 'Incomplete recommendation test.')
    cells[0]['verified'] = True
    cells[1]['metrics']['nrmse'][0] = 1.2
    require(not recommendation(cells, compare(cells, 'proposed', 'original'), isolation_checks(cells), True, True)['recommend'], 'Per-output protection test.')
    cells[1]['metrics']['nrmse'][0] = .8
    cells[1]['search_fingerprint'] = 'changed'
    require(not recommendation(cells, comparison, isolation_checks(cells), True, True)['recommend'], 'Search-isolation test.')
    cells[1]['search_fingerprint'] = 'same'
    require(not recommendation(cells, comparison, isolation_checks(cells), True, False)['recommend'], 'Budget recommendation test.')
    import numpy as np
    target = np.asarray([[0., 2.], [2., 4.]])
    result = metrics(target + 1, target, np.asarray([2., 4.]))
    require(result['rmse'] == [1., 1.] and result['nrmse'] == [.5, .25] and result['r2'] == [0., 0.], 'Independent metric arithmetic test.')
    print('Self-test passed: synthetic rows only; no experiment artifacts or test arrays read/generated.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('self-test', 'verify'))
    arguments = parser.parse_args()
    if arguments.command == 'self-test':
        self_test()
    else:
        sys.exit(verify())
