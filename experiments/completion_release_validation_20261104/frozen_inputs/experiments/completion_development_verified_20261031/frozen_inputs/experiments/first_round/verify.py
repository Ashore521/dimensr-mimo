"""Independent metric, freeze, selection and mechanism reconstruction."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from dimensr_mimo import DimenSRMIMORegressor
from dimensr_mimo._single import DimenSRRegressor
from dimensr_mimo._single.evaluation import target_scale
from dimensr_mimo.graph import System
from cases import generate, design

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    frozen = read(HERE / 'FREEZE.json'); models = read(HERE / 'ALL_MODELS_FROZEN.json')
    scores = read(HERE / 'SCORES.json')
    for p, h in frozen['files'].items():
        assert sha(ROOT / p) == h, p
    for p, h in models['files'].items():
        assert sha(HERE / p) == h, p
    assert models['freeze_sha256'] == sha(HERE / 'FREEZE.json')
    assert scores['all_models_freeze_sha256'] == sha(HERE / 'ALL_MODELS_FROZEN.json')
    assert scores['scored_unix'] >= models['frozen_unix'] >= frozen['frozen_unix']
    assert len(scores['rows']) == len(frozen['jobs'])
    evidence, counts, operators, success, max_metric_delta = [], Counter(), Counter(), Counter(), 0.
    for row, job in zip(scores['rows'], frozen['jobs']):
        assert row['job'] == job
        path = HERE / 'runs' / f"{job['case']}_s{job['seed']}_{job['method']}"
        resource = read(path / 'resources.json')
        counts[resource['status']] += 1
        if resource['status'] != 'complete':
            evidence.append(dict(job=job, status=resource['status'], mechanism_pass=False)); continue
        raw = read(path / 'result.json')
        assert raw['status'] == 'complete'
        events = [json.loads(s) for s in (path / 'events.jsonl').read_text().splitlines()]
        X, Y = generate(job['case'], 'train', job['seed'])
        V, Z = generate(job['case'], 'validation', job['seed'])
        T, U = generate(job['case'], 'test', job['seed'])
        scales = np.array([target_scale(y) for y in Y.T])
        per_output = []
        if job['method'] == 'independent':
            trained = [DimenSRRegressor.load(path / f'model-{j}.json') for j in range(Y.shape[1])]
            predictions = np.column_stack([m.predict(T) for m in trained])
            for j, m in enumerate(trained):
                h = [e for e in events if e['output'] == j]
                stats = raw['diagnostics'][j]['instrumentation']
                per_output.append(dict(output=j, policy_delta=sum(e.get('parameter_delta', 0) for e in h),
                    neural_valid=sum(e.get('neural_valid', 0) for e in h),
                    gp_new_evaluations=stats.get('evaluation_gp_new', 0),
                    nsga_generations=sum(e['phase']=='nsga2' for e in h),
                    nsga_new_evaluations=sum(e.get('new_evaluations', 0) for e in h if e['phase']=='nsga2')))
                train_error = np.sqrt(np.mean((m.predict(X) - Y[:, j])**2)) / scales[j]
                assert np.isclose(train_error, m.best_.train_nrmse, rtol=1e-10, atol=1e-12)
        else:
            m = DimenSRMIMORegressor.load(path / 'model.json'); trained = [m]
            predictions = m.predict(T)
            per_output.append(dict(output=None, policy_delta=sum(e.get('parameter_delta', 0) for e in events),
                neural_valid=sum(e.get('neural_valid', 0) for e in events),
                gp_new_evaluations=sum(e.get('new_evaluations', 0) for e in events if e['phase']=='gp'),
                nsga_generations=sum(e['phase']=='nsga2' for e in events),
                nsga_new_evaluations=sum(e.get('new_evaluations', 0) for e in events if e['phase']=='nsga2')))
            train_errors = np.sqrt(np.mean(((m.predict(X) - Y)/scales)**2, axis=0))
            np.testing.assert_allclose(train_errors, m.best_.errors, rtol=1e-10, atol=1e-12)
            # Reconstruct the final selector from validation values after freeze.
            choices = []
            threshold = m.reference_validation_ * (1 + m.config_.degradation_relative) + m.config_.degradation_absolute
            for c in m.candidates_:
                system = System.from_dict(c['system'])
                values = system.evaluate(V, c['constants'])
                errors = np.sqrt(np.mean(((values - Z)/scales)**2, axis=0))
                np.testing.assert_allclose(errors, c['validation_errors'], rtol=1e-10, atol=1e-12)
                violation = float(np.maximum(errors - threshold, 0).sum()) if m.config_.use_constraints else 0.
                score = float(errors.mean() + m.config_.selection_complexity_weight * c['complexity'] / (Y.shape[1]*m.config_.max_nodes))
                assert np.isclose(violation, c['violation']) and np.isclose(score, c['score'])
                choices.append((violation, score, c['complexity']))
                if job['method'] == 'no_sharing':
                    assert system.sharing['shared_nodes'] == system.sharing['shared_parameters'] == 0
            selected = m.candidates_[min(range(len(choices)), key=choices.__getitem__)]
            assert selected['system'] == m.best_.system.to_dict()
            np.testing.assert_array_equal(selected['constants'], m.best_.constants)
            if job['method'] == 'full':
                operators.update(m.diagnostics_['changed']); success.update(m.diagnostics_['successful'])
        if job['method'] in ('full', 'independent', 'no_sharing'):
            passed = all(e['policy_delta'] > 0 and e['neural_valid'] > 0 and e['gp_new_evaluations'] > 0 and e['nsga_generations'] >= 2 and e['nsga_new_evaluations'] > 0 for e in per_output)
        elif job['method'] == 'no_neural':
            passed = all(e['policy_delta'] == 0 and e['neural_valid'] == 0 and e['gp_new_evaluations'] > 0 and e['nsga_generations'] >= 2 and e['nsga_new_evaluations'] > 0 for e in per_output)
        else:
            passed = all(e['policy_delta'] > 0 and e['neural_valid'] > 0 and e['gp_new_evaluations'] == 0 and e['nsga_generations'] == 0 for e in per_output)
        evidence.append(dict(job=job, status=resource['status'], mechanism_pass=passed, evidence=per_output))
        for index, m in enumerate(trained):
            for name, A in [('train', X), ('validation', V)]:
                h = hashlib.sha256(m.predict(A).astype('<f8').tobytes()).hexdigest()
                assert h == raw['prediction_hashes'][index][name]
        rmse = np.sqrt(np.mean((predictions - U)**2, axis=0)); nrmse = rmse / scales
        r2 = 1 - np.sum((predictions-U)**2,axis=0)/np.sum((U-U.mean(axis=0))**2,axis=0)
        for name, computed in [('rmse', rmse), ('nrmse', nrmse), ('r2', r2)]:
            delta = float(np.max(np.abs(computed-row[name]))); max_metric_delta=max(max_metric_delta,delta)
            np.testing.assert_allclose(computed,row[name],rtol=1e-10,atol=1e-12)
        assert row['saved_original_predictions_equal'] and row['saved_original_audits_equal']
        assert all(a['status']=='enclosed' for a in row['audit'])
    report = dict(frozen_source_files=len(frozen['files']), frozen_model_and_status_files=len(models['files']),
        scheduled=len(frozen['jobs']), statuses=dict(counts), mechanism_pass=sum(e['mechanism_pass'] for e in evidence),
        mechanism_by_method={m:dict(passed=sum(e['mechanism_pass'] for e in evidence if e['job']['method']==m),
                                   total=sum(e['job']['method']==m for e in evidence)) for m in dict.fromkeys(j['method'] for j in frozen['jobs'])},
        changed_operators=dict(operators), successful_operators=dict(success), max_metric_difference=max_metric_delta,
        total_wall_seconds=scores['total_wall_seconds'], budget_respected=scores['total_wall_seconds']<=7200,
        cells=evidence, evidence_files={p.name:sha(p) for p in [HERE/'FREEZE.json',HERE/'ALL_MODELS_FROZEN.json',HERE/'SCORES.json']})
    (HERE / 'VERIFICATION.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='cells'},indent=2))


if __name__ == '__main__':
    main()
