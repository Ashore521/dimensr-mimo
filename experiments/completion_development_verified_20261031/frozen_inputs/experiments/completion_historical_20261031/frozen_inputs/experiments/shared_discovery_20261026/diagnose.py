"""Read-only structural coverage and binding/survival audit; no test files."""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def feature_frequency(system,root,operator,feature):
    """Detect u(x) or optional amplitude*u(p*x); parameter IDs stay explicit."""
    nodes=system['nodes'];node=nodes[root]
    if node['token']=='mul':
        a,b=node['children']
        if nodes[a]['parameter'] is not None:node=nodes[b]
        elif nodes[b]['parameter'] is not None:node=nodes[a]
    if node['token']!=operator:return None
    arg=nodes[node['children'][0]]
    if arg['token']==feature:return {'scaled':False,'parameter':None}
    if arg['token']=='mul':
        a,b=[nodes[i] for i in arg['children']]
        if a['token']==feature and b['parameter'] is not None:return {'scaled':True,'parameter':b['parameter']}
        if b['token']==feature and a['parameter'] is not None:return {'scaled':True,'parameter':a['parameter']}
    return None


def coverage(system):
    roots=system['roots']
    a=feature_frequency(system,roots[0],'sin','x0')
    b=feature_frequency(system,roots[1],'cos','x1')
    return dict(first_root_family=a is not None,second_root_family=b is not None,
                both_root_families=a is not None and b is not None,
                first_scaled=a is not None and a['scaled'],second_scaled=b is not None and b['scaled'],
                both_unscaled=a is not None and b is not None and not a['scaled'] and not b['scaled'],
                shared_frequency=a is not None and b is not None and a['scaled'] and b['scaled'] and a['parameter']==b['parameter'])


def main():
    out=HERE/'DIAGNOSIS.json'
    if out.exists():raise RuntimeError('Do not overwrite the diagnosis.')
    inputs={};rows=[]
    for seed in (0,1):
        folder=ROOT/'experiments/second_round_20261026/diagnostics/trace_runs'/f'seed{seed}'
        path=folder/'candidates.jsonl';summary_path=folder/'summary.json'
        for p in (path,summary_path):inputs[p.relative_to(ROOT).as_posix()]=sha(p)
        outcomes={r['system_id']:r['outcome'] for r in read(summary_path)['cache_outcomes']}
        proposals={};valid={};counts=Counter();ops=defaultdict(Counter);examples={};unique=set()
        for line in path.read_text().splitlines():
            r=json.loads(line)
            if r['kind']=='proposal':
                proposals[r['proposal']]=r;unique.add(r['system_id'])
                cov=coverage(r['system'])
                for key,yes in cov.items():
                    counts[key]+=yes
                    if yes:examples.setdefault(key,dict(proposal=r['proposal'],phase=r['phase'],system_id=r['system_id'],system=r['system']))
                if r.get('operator'):
                    ops[r['operator']]['proposed']+=1
                    ops[r['operator']]['changed']+=r['system_id']!=r['parent']
                    ops[r['operator']]['eligible']+=r['precheck']=='eligible'
            elif r['kind']=='evaluation' and r.get('candidate') is not None:
                c=r['candidate'];proposal=proposals[r['proposal']];mean=sum(c['errors'])/len(c['errors'])
                op=proposal.get('operator')
                if op and r['reason']=='valid' and not r['refine']:
                    ops[op]['valid_new']+=1
                    parent=valid.get(proposal['parent'])
                    if parent is not None:ops[op]['train_mean_improved_parent']+=mean<parent-1e-12
                    ops[op]['retained_final_pool']+=outcomes.get(r['system_id']) in ('selected','final_pool')
                    ops[op]['selected']+=outcomes.get(r['system_id'])=='selected'
                valid[r['system_id']]=mean
        rows.append(dict(seed=seed,proposal_count=len(proposals),unique_proposal_structures=len(unique),
                         root_coverage=dict(counts),operators={k:dict(v) for k,v in ops.items()},examples=examples))
    current=[]
    for seed in (201,202,203,204,205):
        path=ROOT/f'experiments/guarded_selection_20261026/runs/shared_parameter_s{seed}_original/model.json'
        inputs[path.relative_to(ROOT).as_posix()]=sha(path);data=read(path);counts=Counter()
        for c in data['candidates']:counts.update({k:int(v) for k,v in coverage(c['system']).items()})
        current.append(dict(seed=seed,saved_candidates=len(data['candidates']),coverage=dict(counts)))
    report=dict(status='complete',test_data_access=False,input_sha256=inputs,full_trace_audits=rows,
                new_seed_saved_pool_audits=current,
                limitation='Direct root patterns allow optional amplitude and commuted multiplication; no exhaustive algebraic-equivalence claim. Final-pool membership is not every-generation survival.')
    out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('input_sha256','full_trace_audits')},indent=2))
    for row in rows:print(json.dumps({k:v for k,v in row.items() if k!='examples'},indent=2))


if __name__=='__main__':main()
