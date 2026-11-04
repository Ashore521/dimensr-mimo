"""Read-only checks of exact paired selections, failure accounting and callback history."""
import hashlib,json,math,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent.parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from experiments.completion_development_verified_20261031 import archive_io
read=lambda p:json.loads(archive_io.disk_path(p).read_text(encoding='utf-8'))
r=read(HERE/'post_review_checks.json');f=read(HERE/'FREEZE.json');score=read(HERE/'SCORES.json')['rows'];summary=read(HERE/'SUMMARY.json');verified=read(HERE/'VERIFICATION.json')
key=lambda j:(j['case'],j['seed'],j['method'])
scoreix={key(x['job']):x for x in score};issues=[]
def check(ok,name):
    if not ok:issues.append(name)
history_rows=0;exact_pairs=[]
for cell in r['systems']:
    j=cell['job'];tag=f"{j['case']}_s{j['seed']}_{j['method']}";folder=HERE/'runs'/tag
    model=read(folder/'model.json');bundle=read(folder/'delivery.json')
    check(np.allclose(cell['train_nrmse'],bundle['train_nrmse'],rtol=1e-9,atol=1e-11) and np.allclose(cell['validation_nrmse'],bundle['validation_nrmse'],rtol=1e-9,atol=1e-11),tag+':bundle_error_fields')
    check(math.isclose(float(np.mean(cell['clean_nrmse'])),scoreix[key(j)]['mean_clean_nrmse'],rel_tol=1e-12,abs_tol=1e-14),tag+':saved_mean_field')
    events=[json.loads(line) for line in (folder/'events.jsonl').read_text(encoding='utf-8').splitlines()]
    check([{k:v for k,v in e.items() if k not in ['output','monotonic']} for e in events]==model['history'],tag+':saved_history')
    history_rows+=len(events)
    if j['method']=='completion':
        oldtag=f"{j['case']}_s{j['seed']}_base";base=read(HERE/'runs'/oldtag/'model.json')
        encoded=lambda b:json.dumps(b,sort_keys=True,allow_nan=False).encode()
        graph_equal=encoded(model['best']['system'])==encoded(base['best']['system'])
        constants_bits_equal=np.asarray(model['best']['constants'],dtype='<f8').tobytes()==np.asarray(base['best']['constants'],dtype='<f8').tobytes()
        best_equal=encoded(model['best'])==encoded(base['best'])
        check(graph_equal and constants_bits_equal and best_equal,tag+':exact_paired_best')
        exact_pairs.append(dict(case=j['case'],seed=j['seed'],graph_json_equal=graph_equal,constant_bits_equal=constants_bits_equal,best_json_equal=best_equal,best_payload_sha256=hashlib.sha256(encoded(model['best'])).hexdigest()))
totals={k:sum(x[k] for x in r['family_discovery'].values()) for k in next(iter(r['family_discovery'].values()))}
check(totals['new_evaluations']==143 and totals['valid']==139 and totals['protected']==totals['validation_retained']==68,'candidate_supplement_totals')
check(totals['compositional_linear_subproblems']==6142 and totals['implicit_linear_subproblems']==609048 and totals['linear_subproblems']==1231284,'linear_work_totals')
check(totals['compositional_returned_roots']==30 and totals['implicit_returned_roots']==203 and totals['compositional_new_systems']==30 and totals['compositional_valid_systems']==29 and totals['compositional_validation_retained']==13,'transform_totals')
check(len(r['completion_failures'])==11 and r['recomputed_adoption_pass'] is False and r['recomputed_technical_pass'] is True,'preserve_scientific_failure')
check('archived_control_identity' not in json.dumps(summary)+json.dumps(verified),'no_fabricated_archive_identity')
supplement={method:{k:sum(x['supplement'][k] for x in r['systems'] if x['job']['method']==method) for k in ['new_evaluations','valid','protected','validation_retained']} for method in ['base','completion']}
result=dict(status='passed' if not issues else 'findings',issues=issues,main_check_receipt_sha256=archive_io.sha(HERE/'post_review_checks.json'),history_rows_compared=history_rows,exact_paired_best=exact_pairs,supplement_totals=supplement,discovery_totals=totals,script_sha256=archive_io.sha(Path(__file__)))
(HERE/'post_review_validation/extra_checks.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='exact_paired_best'},indent=2))
