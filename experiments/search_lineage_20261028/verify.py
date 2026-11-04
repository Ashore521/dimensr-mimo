"""Recompute training states and locate compact candidates; never read test labels."""
from collections import Counter
import json
from pathlib import Path
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from dimensr_mimo import audit, DimenSRMIMORegressor
from dimensr_mimo.graph import System
from experiments.native_api_20261026.run import read, write, sha, arrays_hash
from experiments.native_api_20261026.verify import raw, mechanism
from experiments.compound_search_20261027 import cases
from experiments.search_lineage_20261028 import study
from experiments.search_lineage_20261028.trace import digest


def cell(folder, frozen):
    job, model = read(folder/'job.json'), read(folder/'model.json')
    resources, result = read(folder/'resources.json'), read(folder/'result.json')
    assert resources['status'] == result['status'] == 'complete'
    expected_config = dict(frozen['config'], random_state=job['seed'], sharing_strategy=job['method'],
        final_selection='precision', acceptance=cases.design(job['case'])['acceptance'])
    from dimensr_mimo import SearchConfig
    from dataclasses import asdict
    assert result['effective_config'] == model['config'] == json.loads(json.dumps(asdict(SearchConfig(**expected_config))))
    assert resources['wall_seconds'] <= job['seconds']+.5
    assert DimenSRMIMORegressor.load(folder/'model.json').to_dict() == model
    X, Y = cases.training_data(job['case'], 'train', job['seed'])
    V, Z = cases.training_data(job['case'], 'validation', job['seed'])
    assert arrays_hash(X, Y, V, Z) == result['data_sha256'] == frozen['data_sha256'][f"{job['case']}_s{job['seed']}"]
    assert result['test_data_accessed'] is False
    rows = [json.loads(line) for line in (folder/'trace.jsonl').read_text().splitlines()]
    assert [r['serial'] for r in rows] == list(range(1, len(rows)+1))
    graphs, states, shapes = {}, {}, {}
    for row in rows:
        if row['kind'] == 'graph':
            assert digest(row['system']) == row['id'] and row['id'] not in graphs
            graph = System.from_dict(row['system'])
            audit.units(graph, model['feature_units'], model['target_units'], model['constant_units'])
            graphs[row['id']], shapes[row['id']] = row['system'], audit.complexity(graph)
        elif row['kind'] == 'state':
            c = row['candidate']
            assert digest(c) == row['id'] and row['id'] not in states
            pred = raw(graphs[c['system']], c['constants'], X)
            errors = np.sqrt(np.mean(((pred-Y)/model['scales'])**2, axis=0))
            np.testing.assert_allclose(errors, c['errors'], rtol=1e-10, atol=1e-13)
            states[row['id']] = c
    variations = {r['serial']:r for r in rows if r['kind'] == 'variation'}
    evaluations = [r for r in rows if r['kind'] == 'evaluation']
    operations, changed, novel, successful = (Counter() for _ in range(4))
    for row in variations.values():
        assert row['parent'] in states and row['donor'] in states and row['graph'] in graphs
        operations[row['operator']] += 1
        changed[row['operator']] += int(row['graph'] != states[row['parent']]['system'])
    parent_costs = []
    for row in evaluations:
        assert row['graph'] in graphs and (row['state'] is None or row['state'] in states)
        if row['proposal'] is not None:
            proposal = variations[row['proposal']]
            assert proposal['graph'] == row['graph'] and proposal['serial'] < row['serial'] and not row['cached']
            op = proposal['operator']; novel[op] += 1
            successful[op] += int(row['state'] is not None)
            if row['state'] is not None:
                before, after = states[proposal['parent']], states[row['state']]
                parent_costs.append(dict(operator=op, proposal=proposal['serial'], parent=proposal['parent'], child=row['state'],
                    parent_errors=before['errors'], child_errors=after['errors'],
                    graph_size_before=shapes[before['system']]['shared_graph_size'],
                    graph_size_after=shapes[after['system']]['shared_graph_size']))
    d = model['diagnostics']
    for name, counts in zip(('operations', 'changed', 'novel', 'successful'), (operations, changed, novel, successful)):
        assert all(counts[k] == d[name].get(k, 0) for k in counts.keys() | d[name].keys()), name
    assert len(evaluations) == d['proposals']
    assert sum(r['calls_after']-r['calls_before'] for r in evaluations) == d['evaluations']
    assert d['evaluations'] <= frozen['config']['max_evaluations']
    phases = [r for r in rows if r['kind'] == 'phase']
    assert [r['event'] for r in phases] == model['history']
    assert all(s in states for r in phases for s in r['population'])
    key = lambda c:(c['system'], tuple(c['constants']))
    state_keys = {key(c):s for s, c in states.items()}
    final = rows[-1]
    assert final['kind'] == 'final' and final['search_fingerprint'] == model['search_fingerprint']
    expected_pool = [dict(graph=digest(c['system']), constants=c['constants']) for c in model['candidates']]
    assert final['pool'] == expected_pool
    assert final['selected'] == dict(graph=digest(model['best']['system']), constants=model['best']['constants'])
    pool_states = {state_keys[(c['graph'], tuple(c['constants']))] for c in final['pool']}
    for c in model['candidates']:
        s = states[state_keys[(digest(c['system']), tuple(c['constants']))]]
        assert s['errors'] == c['errors']
        pred = raw(c['system'], c['constants'], V)
        np.testing.assert_allclose(np.sqrt(np.mean(((pred-Z)/model['scales'])**2, axis=0)),
            c['validation_errors'], rtol=1e-10, atol=1e-13)
    limits = model['config']['acceptance']
    compact_accurate = [s for s,c in states.items() if all(a<=b for a,b in zip(c['errors'], limits['nrmse']))
        and all(shapes[c['system']][k]<=limits[k] for k in limits if k != 'nrmse')]
    return dict(job=job, verified=True, graphs=len(graphs), states=len(states), evaluations=len(evaluations),
        mechanism=mechanism(model), train_compact_accurate_states=compact_accurate,
        train_compact_accurate_in_final_pool=[s for s in compact_accurate if s in pool_states],
        retained_at_phases=[dict(phase=r['event']['phase'], step=r['event']['step'],
            compact_accurate=[s for s in r['population'] if s in compact_accurate]) for r in phases],
        parent_costs=parent_costs, selected_train=model['best']['errors'], selected_validation=model['validation_nrmse'],
        application_acceptance=model['post_selection'], test_data_accessed=False)


def verify():
    if (HERE/'VERIFICATION.json').exists():
        raise RuntimeError('Keep existing verification.')
    frozen, sealed = read(HERE/'FREEZE.json'), read(HERE/'ALL_MODELS_FROZEN.json')
    study.prior.verify_files(frozen['files'])
    study.prior.verify_files(sealed['files'], HERE)
    assert sealed['freeze_sha256'] == sha(HERE/'FREEZE.json')
    assert frozen['frozen_unix'] <= sealed['frozen_unix']
    rows = []
    for job in frozen['jobs']:
        try:
            assert time.time() < study.supervisor.budget(), 'Verification budget expired.'
            folder = HERE/'runs'/study.prior.name(job)
            seal = read(folder/'CELL_FROZEN.json')
            assert seal['frozen_unix'] <= sealed['frozen_unix']
            study.prior.verify_files(seal['files'], folder)
            assert read(folder/'job.json') == job
            row = cell(folder, frozen)
            assert time.time() < study.supervisor.budget(), 'Cell verification exceeded the deadline.'
        except Exception as error:
            row = dict(job=job, verified=False, error=repr(error))
        rows.append(row)
        print(study.prior.name(job), row.get('error', 'verified'), flush=True)
    finished = time.time()
    budget_met = finished < study.supervisor.budget()
    write(HERE/'VERIFICATION.json', dict(status='passed' if budget_met and all(r['verified'] for r in rows) else 'failed',
        expected=len(frozen['jobs']), verified=sum(r['verified'] for r in rows), rows=rows,
        elapsed_seconds=finished-frozen['budget']['start_unix'], budget_met=budget_met, test_data_accessed=False,
        freeze_sha256=sha(HERE/'FREEZE.json'), models_sha256=sha(HERE/'ALL_MODELS_FROZEN.json')))


if __name__ == '__main__':
    verify()
