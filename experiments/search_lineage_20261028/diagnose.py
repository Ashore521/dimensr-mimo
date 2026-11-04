"""Post-freeze descriptive training diagnostics; no model reselection or test data."""
from collections import Counter
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from experiments.native_api_20261026.run import read, write, sha
from experiments.search_lineage_20261028.trace import digest


def main():
    assert not (HERE/'DIAGNOSIS.json').exists(), 'Preserve prior diagnosis.'
    verification = read(HERE/'VERIFICATION.json')
    assert verification['status'] == 'passed' and verification['verified'] == 10
    rows = []
    for verified in verification['rows']:
        job = verified['job']; name = f"{job['case']}_s{job['seed']}_{job['method']}"
        folder = HERE/'runs'/name
        trace = [json.loads(line) for line in (folder/'trace.jsonl').read_text().splitlines()]
        states = {r['id']:r['candidate'] for r in trace if r['kind'] == 'state'}
        final_graphs = {c['graph'] for c in trace[-1]['pool']}
        accurate = set(verified['train_compact_accurate_states'])
        accurate_graphs = {states[s]['system'] for s in accurate}
        missing_graphs = accurate_graphs-final_graphs
        phase_counts = Counter()
        state_serials = {r['id']:r['serial'] for r in trace if r['kind'] == 'state'}
        for s in accurate:
            following = next((r for r in trace if r['serial'] > state_serials[s] and r['kind'] == 'phase'), None)
            phase_counts[following['event']['phase'] if following else 'after_last_phase'] += 1
        matched_costs = [c for c in verified['parent_costs'] if c['operator'] == 'matched_share']
        parent_protected = [all(a <= b*1.1+1e-8 for a,b in zip(c['child_errors'], c['parent_errors'])) for c in matched_costs]
        model = read(folder/'model.json')
        old = read(HERE.parent/'matched_sharing_20261028/runs'/name/'model.json')
        selected_graph = digest(model['best']['system'])
        selected_origins = [c for c in verified['parent_costs'] if states[c['child']]['system'] == selected_graph]
        rows.append(dict(job=job, search_fingerprint_matches_160=model['search_fingerprint'] == old['search_fingerprint'],
            selected_matches_160=model['best'] == old['best'], states=len(states),
            compact_train_accurate_states=len(accurate), compact_train_accurate_graphs=len(accurate_graphs),
            compact_train_accurate_states_in_pool=len(verified['train_compact_accurate_in_final_pool']),
            compact_train_accurate_graphs_missing_from_pool=sorted(missing_graphs), first_observed_phase=dict(phase_counts),
            minimum_training_max_output_error=min(max(c['errors']) for c in states.values()),
            matched_successes=len(matched_costs), matched_parent_protected=sum(parent_protected),
            matched_parent_degraded=[c for c,p in zip(matched_costs, parent_protected) if not p],
            selected_origins=selected_origins, selected_train=verified['selected_train'],
            selected_validation=verified['selected_validation'], acceptance=verified['application_acceptance']['status']))
    write(HERE/'DIAGNOSIS.json', dict(rows=rows, test_data_accessed=False, model_reselection=False,
        verification_sha256=sha(HERE/'VERIFICATION.json'), script_sha256=sha(Path(__file__)),
        parent_comparison=dict(relative=.1, absolute=1e-8, purpose='descriptive only; no search constraint change')))
    for r in rows:
        print(json.dumps({k:v for k,v in r.items() if k not in ('matched_parent_degraded', 'selected_origins')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
