"""Summarize sealed observations only; do not regenerate data or initialization."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nodes(tree):
    yield tree
    for child in tree['children']:
        yield from nodes(child)


def size_depth(tree):
    children=[size_depth(child) for child in tree['children']]
    return 1+sum(s for s,d in children),1+max((d for s,d in children),default=0)


def log_square(tree):
    return tree['token']=='log' and any(n['token']=='square' for n in nodes(tree))


def sqrt_sum_squares(tree):
    if tree['token']!='sqrt' or tree['children'][0]['token']!='add':
        return False
    return all(any(n['token']=='square' for n in nodes(child)) for child in tree['children'][0]['children'])


def variable_times_log_square(tree):
    return tree['token']=='mul' and any(
        a['token'].startswith('x') and log_square(b)
        for a,b in (tree['children'],tree['children'][::-1]))


def flags(tree):
    return {name:any(predicate(n) for n in nodes(tree)) for name,predicate in (
        ('log_square',log_square),('sqrt_sum_squares',sqrt_sum_squares),
        ('variable_times_log_square',variable_times_log_square))}


def main():
    verification=json.loads((HERE/'VERIFICATION.json').read_text())
    assert all(sha(HERE/name)==digest for name,digest in verification['outputs'].items())
    rows=[]
    for checked in verification['rows']:
        tag=f"{checked['case']}_o{checked['output']}"
        raw=json.loads((HERE/f'output_{tag}_raw.json').read_text())
        result=json.loads((HERE/f'output_{tag}.json').read_text())
        facets={key:{stage:0 for stage in ('features','proposals','returned','boundary_attempts','boundary_rejected')}
                for key in flags({'token':'x0','children':[]})}
        for stage,records in [('features',raw['features']),('proposals',raw['proposals']),('returned',raw['returned_seeds'])]:
            for record in records:
                for key,found in flags(record['tree']).items():facets[key][stage]+=int(found)
        boundaries=dict(attempted=0,structure_rejected=0,nonfinite_error=0)
        direct_log_rejections=[]
        with (HERE/f'output_{tag}_trace.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                item=json.loads(line)
                if item['kind']!='proposal_boundary':continue
                record=item['record'];boundaries['attempted']+=1
                boundaries['structure_rejected']+=int(record['structure_rejected'])
                boundaries['nonfinite_error']+=int(record['finite_error'] is False)
                for key,found in flags(record['tree']).items():
                    facets[key]['boundary_attempts']+=int(found)
                    facets[key]['boundary_rejected']+=int(found and record['structure_rejected'])
                if record['source_line']==198 and record['structure_rejected'] and flags(record['tree'])['log_square']:
                    direct_log_rejections.append(record)
        assert boundaries==raw['boundary_counts']==result['boundary_counts']
        proposals=result['proposals'];order=result['independent_order_as_original_indices']
        mismatch=[dict(position=i,independent_original_index=j,original_score=proposals[i]['score'],
                       original_other_score=proposals[j]['score'],recomputed_score=proposals[i]['recomputed_score'],
                       recomputed_other_score=proposals[j]['recomputed_score']) for i,j in enumerate(order) if i!=j]
        rows.append(dict(case=checked['case'],output=checked['output'],observation_status=checked['status'],
            counts=raw['counts'],boundary_counts=boundaries,facets=facets,
            root_log_square_features=[dict(tree=r['tree'],constants=r['constants'],
                size=size_depth(r['tree'])[0],depth=size_depth(r['tree'])[1]) for r in raw['features'] if log_square(r['tree'])],
            direct_log_boundary_rejections=direct_log_rejections,
            rank_mismatches=mismatch,top12_match=result['independent_seed_order_match'],
            maximum_score_difference=max(abs(r['score']-r['recomputed_score']) for r in proposals)))
    report=dict(status='observed_with_rank_limitations',expected_calls=4,completed_calls=4,
        full_order_verified=sum(row['observation_status']=='passed' for row in rows),
        top12_verified=sum(row['top12_match'] for row in rows),rows=rows,
        scope='Syntactic facets of the four saved initializer calls; not symbolic equivalence or whole-search coverage.',
        verification_sha256=sha(HERE/'VERIFICATION.json'),script_sha256=sha(Path(__file__)))
    with (HERE/'LINEAGE_SUMMARY.json').open('x',encoding='utf-8') as stream:
        json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}))


if __name__=='__main__':
    main()
